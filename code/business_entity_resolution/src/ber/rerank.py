"""Cross-encoder reranker for uncertain candidate pairs.

A transformer reads both records together ("name | address" of the S1 entity
and of the S2/S3 record, raw text so native scripts are kept) and outputs a
match logit. It sees token-level interactions the pairwise similarity features
cannot (a transliterated street name, a reordered brand), so it is used only on
the band of pairs stage 1 is unsure about.

Base: intfloat/multilingual-e5-small (MIT) with a linear head on the mean-pooled
output. Trained on fit-universe band pairs of *one half* of the fit entities
(RERANK_HALF); stage 2 learns how far to trust it on the other half, so its
scores are never in-sample for stage 2.
"""
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import config
from .embed import BASE_MODEL

MAX_LEN = 128
# BER_BAND=narrow: stage-1 p in [0.01, 0.99]; wide: [0.001, 0.99] plus the name encoder's top-1
# candidate of each record (cross-script names stage 1 wrongly rules out)
BAND = (0.001, 0.99) if os.environ.get("BER_BAND", "narrow") == "wide" else (0.01, 0.99)
RERANK_HALF = 0           # fit entities with hash(s1_id) % 2 == RERANK_HALF train the reranker


def model_dir():
    return config.WORK_DIR / "models" / f"reranker{config.tag_suffix()}"


def in_band(pl, columns):
    """Pairs worth a cross-encoder look: uncertain stage-1 p, or the name encoder's
    top candidate for the record even when stage 1 is sure it is not a match
    (cross-script names the pairwise features cannot see)."""
    cond = pl.col("p").is_between(*BAND)
    if "ecos_rank_m" in columns and os.environ.get("BER_BAND", "narrow") == "wide":
        cond = cond | ((pl.col("ecos_rank_m") == 1) & (pl.col("p") < BAND[1]))
    return cond


def half_expr(pl):
    return (pl.col("s1_id").hash(config.SEED + 7) % 2)


class CrossEncoder(torch.nn.Module):
    def __init__(self, path_or_name):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(path_or_name)
        self.head = torch.nn.Linear(self.enc.config.hidden_size, 1)

    def forward(self, batch):
        h = self.enc(**batch).last_hidden_state
        mask = batch["attention_mask"].unsqueeze(-1).to(h.dtype)
        return self.head((h * mask).sum(1) / mask.sum(1).clamp(min=1)).squeeze(-1)


def _tok():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE_MODEL)


def train(a, b, y, epochs=1, batch_size=256, lr=5e-5, device="cuda", log=print):
    tok = _tok()
    model = CrossEncoder(BASE_MODEL).to(device).train()
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
            idx = order[i:i + batch_size]
            batch = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                        max_length=MAX_LEN, return_tensors="pt").to(device)
            target = torch.from_numpy(y[idx]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(batch)
            loss = F.binary_cross_entropy_with_logits(logit.float(), target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            if step % 500 == 0 or step == steps:
                log(f"  rerank step {step}/{steps} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    out = model_dir()
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"head": model.head.state_dict()}, out / "head.pt")
    model.enc.save_pretrained(out)
    tok.save_pretrained(out)
    return out


@torch.no_grad()
def score(a, b, batch_size=1024, device="cuda", log=print):
    """Match logits for aligned text lists a, b."""
    tok = _tok()
    model = CrossEncoder(str(model_dir()))
    model.head.load_state_dict(torch.load(model_dir() / "head.pt", map_location="cpu")["head"])
    model = model.to(device).eval().half()
    order = np.argsort([len(x) + len(z) for x, z in zip(a, b)], kind="stable")
    out = np.empty(len(a), dtype=np.float32)
    t0 = time.time()
    for k, i in enumerate(range(0, len(a), batch_size)):
        idx = order[i:i + batch_size]
        batch = tok([a[j] for j in idx], [b[j] for j in idx], padding=True, truncation=True,
                    max_length=MAX_LEN, return_tensors="pt").to(device)
        out[idx] = model(batch).float().cpu().numpy()
        if k % 200 == 0:
            log(f"  reranked {min(i + batch_size, len(a)):,}/{len(a):,} ({time.time() - t0:.0f}s)")
    return out
