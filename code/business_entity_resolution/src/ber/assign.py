"""Decision layer: scores -> final matches.

Each S2/S3 record belongs to at most one S1 entity, so a record keeps only its
highest-probability candidate, and only when p >= tau (exclusive assignment).
"""
import numpy as np
import polars as pl

from .evaluate import macro_f05

KEY = ["s1_id", "m_src", "m_id"]


def exclusive(scored):
    """Best S1 per S2/S3 record (ties broken by s1_id for determinism)."""
    return (scored.sort(["p", "s1_id"], descending=[True, False])
            .unique(["m_src", "m_id"], keep="first"))


def matches(scored, tau):
    return exclusive(scored).filter(pl.col("p") >= tau).select(KEY)


def tune_tau(scored, gt, entities, grid=None):
    """Return (best tau, [(tau, f05)]) maximising macro F0.5."""
    grid = np.round(np.arange(0.2, 0.96, 0.05), 2) if grid is None else grid
    best = exclusive(scored)
    curve = [(float(t), macro_f05(best.filter(pl.col("p") >= t).select(KEY), gt, entities)) for t in grid]
    return max(curve, key=lambda x: x[1])[0], curve
