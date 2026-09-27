"""Cross-encoder reranker for uncertain candidate pairs.

A transformer reads both records together ("name | address" of the S1 entity
and of the S2/S3 record, raw text so native scripts are kept) and outputs a
match logit. It sees token-level interactions the pairwise similarity features
cannot (a transliterated street name, a reordered brand), so it is used only on
the band of pairs stage 1 is unsure about.

Models (BER_RERANK_MODEL):
  e5-small  intfloat/multilingual-e5-small (MIT, 118M) + linear head on the mean-pooled output
  bge-m3    BAAI/bge-reranker-v2-m3 (Apache-2.0, 568M), a pretrained multilingual cross-encoder
Trained on pairs of *one half* of the fit entities (RERANK_HALF); stage 2 learns how
far to trust it on the other half, so its scores are never in-sample for stage 2.
BER_RERANK_ALL=1 scores every pruned candidate pair instead of the uncertain band.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .embed import BASE_MODEL

MAX_LEN = 128
# BER_BAND=wide (default; val 0.9881 vs 0.9845 narrow): stage-1 p in [0.001, 0.99] plus the
# name encoder's top-1 candidate of each record; narrow: p in [0.01, 0.99]
BAND = (0.001, 0.99) if os.environ.get("BER_BAND", "wide") == "wide" else (0.01, 0.99)
RERANK_HALF = 0           # fit entities with hash(s1_id) % 2 == RERANK_HALF train the reranker
KIND = os.environ.get("BER_RERANK_MODEL", "bge-m3")   # val: bge-m3 0.9930 vs e5-small 0.9877
MODELS = {  # name -> (checkpoint, train batch, learning rate, micro-batches per step)
    "e5-small": (BASE_MODEL, 256, 5e-5, 1),
    "bge-m3": ("BAAI/bge-reranker-v2-m3", 128, 2e-5, 2),   # 2 x 64: fits 32 GB with headroom
}
RERANK_ALL = os.environ.get("BER_RERANK_ALL", "1") == "1"


def model_dir():
    kind = "" if KIND == "e5-small" else f"_{KIND}"
    return config.WORK_DIR / "models" / f"reranker{config.tag_suffix()}{kind}"


def in_band(pl, columns, band_only=False):
    """Pairs worth a cross-encoder look: uncertain stage-1 p, or the name encoder's
    top candidate for the record even when stage 1 is sure it is not a match
    (cross-script names the pairwise features cannot see)."""
    if RERANK_ALL and not band_only:
        return pl.lit(True)
    cond = pl.col("p").is_between(*BAND)
    if "ecos_rank_m" in columns and os.environ.get("BER_BAND", "wide") == "wide":
        cond = cond | ((pl.col("ecos_rank_m") == 1) & (pl.col("p") < BAND[1]))
    return cond


def half_expr(pl):
    return (pl.col("s1_id").hash(config.SEED + 7) % 2)


class CrossEncoder(torch.nn.Module):
    """e5-small: mean-pooled encoder + linear head. bge-m3: its own classification head."""

    def __init__(self, path_or_name, kind=KIND):
        super().__init__()
        self.kind = kind
        if kind == "e5-small":
            from transformers import AutoModel
            self.enc = AutoModel.from_pretrained(path_or_name)
            self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)
        else:
            from transformers import AutoModelForSequenceClassification
            self.enc = AutoModelForSequenceClassification.from_pretrained(path_or_name, num_labels=1)

    def forward(self, batch):
        if self.kind != "e5-small":
            return self.enc(**batch).logits[:, 0]
        h = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(-1)


def _tok(path=None):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(path or MODELS[KIND][0])


def train(a, b, y, epochs=1, batch_size=None, lr=None, device="cuda", log=print):
    ckpt, bs, base_lr, accum = MODELS[KIND]
    batch_size, lr = batch_size or bs, lr or base_lr
    tok = _tok()
    model = CrossEncoder(ckpt).to(device).train()
    n = len(a)
    steps = epochs * (n // batch_size)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 20)) * max(0.0, 1 - s / max(1, steps)))
    rng = np.random.default_rng(config.SEED)
    y = np.asarray(y, dtype=np.float32)
    step, t0 = 0, time.time()
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx_all = order[i:i + batch_size]
            opt.zero_grad(set_to_none=True)
            for idx in np.array_split(idx_all, accum):   # same update as one full batch
                batch = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                            max_length=MAX_LEN, return_tensors="pt").to(device)
                target = torch.from_numpy(y[idx]).to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logit = model(batch)
                loss = F.binary_cross_entropy_with_logits(logit.float(), target) * (len(idx) / len(idx_all))
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0 or step == steps:
                log(f"  rerank step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    model.enc.save_pretrained(out)
    tok.save_pretrained(out)
    # head.pt marks a finished model (and holds the e5 head)
    torch.save({"head": model.head.state_dict() if KIND == "e5-small" else None, "kind": KIND},
               out / "head.pt")
    return out


def _load_for_scoring(device):
    saved = torch.load(model_dir() / "head.pt", map_location="cpu")
    model = CrossEncoder(str(model_dir()), kind=saved.get("kind", "e5-small"))
    if model.kind == "e5-small":
        model.head.load_state_dict(saved["head"])
    return model.to(device).eval().half()


@torch.no_grad()
def _score_shard(a, b, idx_all, device, batch_size, out, log):
    tok = _tok(str(model_dir()))
    model = _load_for_scoring(device)
    t0 = time.time()
    for k, i in enumerate(range(0, len(idx_all), batch_size)):
        idx = idx_all[i:i + batch_size]
        batch = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to(device)
        out[idx] = model(dict(batch)).float().cpu().numpy()
        if k % 200 == 0:
            log(f"  [{device}] reranked {min(i + batch_size, len(idx_all)):,}/{len(idx_all):,} "
                f"({time.time() - t0:.0f}s)")


def score(a, b, batch_size=1024, log=print):
    """Match logits for aligned text lists a, b. With several GPUs, each GPU gets its own
    model copy and an interleaved share of the (length-sorted) pairs: no GPU-to-GPU traffic
    (torch DataParallel crashed with illegal memory access on the 2x RTX 5090 box)."""
    from concurrent.futures import ThreadPoolExecutor
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)], kind="stable")
    out = np.empty(len(a), dtype=np.float32)
    n_gpu = max(1, torch.cuda.device_count())
    shards = [order[g::n_gpu] for g in range(n_gpu)]
    with ThreadPoolExecutor(n_gpu) as ex:
        list(ex.map(lambda g: _score_shard(a, b, shards[g], f"cuda:{g}", batch_size, out, log),
                    range(n_gpu)))
    return out
