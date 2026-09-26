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


def expected_f_select(scored, extra=0.5, floor=0.05, beta2=0.25):
    """Per entity, keep the top-j exclusive candidates maximising expected F0.5.

    With p_1 >= p_2 >= ... the entity's exclusive candidates, choosing the top j
    gives E[F] ~ (1+b2) * sum_{i<=j} p_i / (b2 * E[n_true] + j), where
    E[n_true] = sum_i p_i + extra (true matches the candidates missed).
    Predicting nothing is worth P(no match) ~ prod(1 - p_i).
    """
    b = (exclusive(scored).filter(pl.col("p") >= floor)
         .sort(["s1_id", "p"], descending=[False, True])
         .with_columns(pl.col("p").cum_sum().over("s1_id").alias("etp"),
                       pl.int_range(1, pl.len() + 1).over("s1_id").alias("j"),
                       (pl.col("p").sum().over("s1_id") + extra).alias("en")))
    b = b.with_columns(((1 + beta2) * pl.col("etp") / (beta2 * pl.col("en") + pl.col("j"))).alias("ef"))
    best = b.group_by("s1_id").agg(
        pl.col("ef").max().alias("ef_max"),
        pl.col("j").filter(pl.col("ef") == pl.col("ef").max()).first().alias("j_best"),
        (1 - pl.col("p")).product().alias("p_empty"))
    keep = best.filter(pl.col("ef_max") > pl.col("p_empty"))
    return b.join(keep, on="s1_id").filter(pl.col("j") <= pl.col("j_best")).select(KEY)


def tune_rule(scored, gt, entities):
    """Best of the global threshold and expected-F0.5 selection -> (rule dict, score, log)."""
    tau, curve = tune_tau(scored, gt, entities)
    best = ({"rule": "tau", "tau": tau}, max(f for _, f in curve))
    lines = ["tau curve: " + " ".join(f"{t:.2f}:{v:.4f}" for t, v in curve)]
    for extra in (0.0, 0.25, 0.5, 1.0):
        f = macro_f05(expected_f_select(scored, extra), gt, entities)
        lines.append(f"expected-F extra={extra}: {f:.4f}")
        if f > best[1]:
            best = ({"rule": "ef", "extra": extra, "tau": tau}, f)
    return best[0], best[1], lines


def apply_rule(scored, rule):
    if rule.get("rule") == "ef":
        return expected_f_select(scored, rule["extra"])
    return matches(scored, rule["tau"])
