"""Multilingual name encoder (fine-tuned bi-encoder) for cross-script matching.

Base model: intfloat/multilingual-e5-small (MIT, 118M params). It reads native
scripts directly, so 'ভিশন সিস্টেমস প্রাইভেট লিমিটেড' can land next to
'Vision Systems Private Limited' without hand-written transliteration rules.
It is fine-tuned with an in-batch contrastive loss on (S1 name, S2/S3 name)
true pairs from the *fit* fold only, oversampling native-script names.

Embeddings are L2-normalised float16, one row per record in the order of the
record parquet files (work/emb/{split}_s{src}.npy).
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config

BASE_MODEL = "intfloat/multilingual-e5-small"
MAX_LEN = 32
DIM = 384


def emb_path(split, src):
    return config.WORK_DIR / "emb" / f"{split}_s{src}.npy"


def model_dir():
    return config.WORK_DIR / "models" / "name_encoder"


def _load(path_or_name, device):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path_or_name)
    model = AutoModel.from_pretrained(path_or_name).to(device)
    return tok, model


def _pool(model, batch):
    h = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
    return F.normalize((h * mask).sum(1) / mask.sum(1).clamp(min=1), dim=-1)


def train(pairs_a, pairs_b, epochs=1, batch_size=1024, lr=1e-4, temperature=0.05,
          device="cuda", log=print):
    """Contrastive fine-tuning: a[i] should be closest to b[i] among the batch."""
    tok, model = _load(BASE_MODEL, device)
    model.train()
    n = len(pairs_a)
    steps = epochs * (n // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / max(1, steps)))
    rng = np.random.default_rng(config.SEED)
    labels = torch.arange(batch_size, device=device)
    step, t0 = 0, time.time()
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = order[i:i + batch_size]
            ta = tok([pairs_a[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN,
                     return_tensors="pt").to(device)
            tb = tok([pairs_b[j] for j in idx], padding=True, truncation=True, max_length=MAX_LEN,
                     return_tensors="pt").to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                ea, eb = _pool(model, ta), _pool(model, tb)
                logits = (ea @ eb.T).float() / temperature
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 200 == 0 or step == steps:
                log(f"  step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@torch.no_grad()
def encode(texts, path_or_name=None, batch_size=4096, device="cuda", log=print):
    """float16 (len(texts), DIM) L2-normalised embeddings, sorted by length for speed."""
    tok, model = _load(str(path_or_name or model_dir()), device)
    model.eval().half()
    order = np.argsort([len(t) for t in texts], kind="stable")
    out = np.empty((len(texts), DIM), dtype=np.float16)
    t0 = time.time()
    for k, i in enumerate(range(0, len(texts), batch_size)):
        idx = order[i:i + batch_size]
        b = tok([texts[j] or " " for j in idx], padding=True, truncation=True, max_length=MAX_LEN,
                return_tensors="pt").to(device)
        out[idx] = _pool(model, b).float().cpu().numpy().astype(np.float16)
        if k % 500 == 0:
            log(f"  encoded {min(i + batch_size, len(texts)):,}/{len(texts):,} ({time.time() - t0:.0f}s)")
    return out


def load_embeddings(split, src, mmap=True):
    return np.load(emb_path(split, src), mmap_mode="r" if mmap else None)


def rowwise_cos(A, B, chunk=5_000_000):
    """Elementwise dot of aligned float16 rows -> float32."""
    out = np.empty(len(A), dtype=np.float32)
    for i in range(0, len(A), chunk):
        out[i:i + chunk] = np.einsum("ij,ij->i", A[i:i + chunk].astype(np.float32),
                                     B[i:i + chunk].astype(np.float32))
    return out


def gpu_dense_topk(Q, X, k, min_score=-1.0, device="cuda"):
    """Top-k inner products of dense float16 queries Q against index X on the GPU."""
    if len(Q) == 0 or len(X) == 0:
        return (np.empty(0, np.int64), np.empty(0, np.int32), np.empty(0, np.float32))
    Xt = torch.from_numpy(np.ascontiguousarray(X)).to(device)
    kk = min(k, len(X))
    chunk = max(256, min(16384, int(4e8 // max(1, len(X)))))
    rows, cols, vals = [], [], []
    for s in range(0, len(Q), chunk):
        q = torch.from_numpy(np.ascontiguousarray(Q[s:s + chunk])).to(device)
        v, i = torch.topk(q @ Xt.T, kk, dim=1)
        v, i = v.float().cpu().numpy().ravel(), i.cpu().numpy().ravel()
        r = np.repeat(np.arange(s, s + len(q)), kk)
        keep = v > min_score
        rows.append(r[keep])
        cols.append(i[keep].astype(np.int32))
        vals.append(v[keep])
    return np.concatenate(rows).astype(np.int64), np.concatenate(cols), np.concatenate(vals)
