"""Macro F0.5 exactly as the leaderboard defines it, plus blocking diagnostics.

Pair frames are long-format with columns s1_id, m_src, m_id.
`entities` has one row per S1 entity in the evaluation set (column s1_id);
every entity counts, including singletons and ones with no prediction.
"""
import polars as pl

BETA2 = 0.25
KEY = ["s1_id", "m_src", "m_id"]


def f05(tp, n_pred, n_true):
    """Scalar per-entity F0.5 with the singleton rule."""
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0 or tp == 0:
        return 0.0
    p, r = tp / n_pred, tp / n_true
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def per_entity_scores(pred, gt, entities):
    """Return entities with n_pred, n_true, tp and f columns."""
    pred = pred.select(KEY).unique().join(entities.select("s1_id"), on="s1_id", how="semi")
    gt = gt.select(KEY).join(entities.select("s1_id"), on="s1_id", how="semi")
    tp = pred.join(gt, on=KEY, how="inner").group_by("s1_id").len("tp")
    n_pred = pred.group_by("s1_id").len("n_pred")
    n_true = gt.group_by("s1_id").len("n_true")
    df = (
        entities.select("s1_id")
        .join(n_pred, on="s1_id", how="left")
        .join(n_true, on="s1_id", how="left")
        .join(tp, on="s1_id", how="left")
        .with_columns(pl.col("n_pred", "n_true", "tp").fill_null(0).cast(pl.Int32))
    )
    p = pl.col("tp") / pl.col("n_pred")
    r = pl.col("tp") / pl.col("n_true")
    f = (
        pl.when(pl.col("n_true") == 0).then((pl.col("n_pred") == 0).cast(pl.Float64))
        .when(pl.col("tp") == 0).then(0.0)
        .otherwise((1 + BETA2) * p * r / (BETA2 * p + r))
    )
    return df.with_columns(f.alias("f"))


def macro_f05(pred, gt, entities):
    return per_entity_scores(pred, gt, entities)["f"].mean()


def report(pred, gt, entities, by=None):
    """Macro F0.5 overall and (optionally) grouped by a column present in `entities`."""
    s = per_entity_scores(pred, gt, entities)
    if by is not None:
        s = s.join(entities.select("s1_id", by), on="s1_id")
    agg = [
        pl.len().alias("entities"),
        pl.col("f").mean().alias("f05"),
        (pl.col("tp").sum() / pl.col("n_pred").sum()).alias("micro_precision"),
        (pl.col("tp").sum() / pl.col("n_true").sum()).alias("micro_recall"),
        pl.col("f").filter(pl.col("n_true") == 0).mean().alias("singleton_f"),
    ]
    overall = s.select(agg)
    if by is None:
        return overall
    return pl.concat([
        s.group_by(by).agg(agg).sort(by).with_columns(pl.col(by).cast(pl.Utf8)),
        overall.select(pl.lit("ALL").alias(by), pl.all()),
    ])


def blocking_report(cand, gt, entities):
    """Recall ceiling and size of a candidate set.

    oracle_f05 = score if the matcher picked exactly cand ∩ truth for every entity.
    """
    cand = cand.select(KEY).unique().join(entities.select("s1_id"), on="s1_id", how="semi")
    gt = gt.select(KEY).join(entities.select("s1_id"), on="s1_id", how="semi")
    hit = cand.join(gt, on=KEY, how="inner")
    s = per_entity_scores(hit, gt, entities)
    n_true = s["n_true"]
    return {
        "entities": entities.height,
        "candidate_pairs": cand.height,
        "cands_per_entity": cand.height / entities.height,
        "pair_recall": hit.height / max(gt.height, 1),
        "entities_fully_covered": float(((s["tp"] == n_true)).mean()),
        "oracle_f05": float(s["f"].mean()),
    }
