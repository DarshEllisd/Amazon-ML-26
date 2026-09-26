"""Stage 2: group features from stage-1 scores.

Records of one entity resemble each other, so a candidate that looks like the
entity's *other* confident matches is probably right even when its own name is
odd, and a candidate unlike all of them is suspect. For each candidate pair
(s1, m) this computes, against C = the other S2/S3 records confidently assigned
to s1 by stage 1 (exclusive best with p >= CONF):
  grp_n                       |C|
  grp_name_max / grp_name_mean   token_set_ratio(name_m, name_c)
  grp_addr_max / grp_addr_mean   token_set_ratio(addr_m, addr_c)
  grp_ecos_max                encoder cosine(name_m, name_c) (if embeddings exist)
plus how the stage-1 score compares with rivals:
  p1, p1_rank_m, p1_gap_m (vs best other S1 of this record), p1_second_m,
  p1_rank_1 (rank among the entity's candidates), p1_sum_1 (expected size).
Stage-1 scores on the training universe must be out-of-fold.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

KEY = ["s1_id", "m_src", "m_id"]
CONF = 0.5
CHUNK = 5_000_000


def _p_features(scored):
    return scored.with_columns(
        pl.col("p").rank("ordinal", descending=True).over("m_src", "m_id").cast(pl.Int16).alias("p1_rank_m"),
        pl.col("p").rank("ordinal", descending=True).over("s1_id").cast(pl.Int16).alias("p1_rank_1"),
        pl.col("p").sum().over("s1_id").alias("p1_sum_1"),
        pl.len().over("m_src", "m_id").cast(pl.Int16).alias("p1_n_m"),
    ).with_columns(
        # best score this record gets from any *other* S1
        pl.when(pl.col("p1_rank_m") == 1)
        .then(pl.col("p").filter(pl.col("p1_rank_m") == 2).max().over("m_src", "m_id"))
        .otherwise(pl.col("p").max().over("m_src", "m_id")).fill_null(0).alias("p1_other_m"),
    ).with_columns((pl.col("p") - pl.col("p1_other_m")).alias("p1_gap_m")).rename({"p": "p1"})


def build(scored, m, emb_m=None, log=print):
    """scored: s1_id, m_src, m_id, p (stage-1, all candidates of the universe).
    m: records with m_src, m_id, name_core, addr_nr (+ row for embeddings).
    emb_m: optional function (m_src array, row array) -> float16 embeddings."""
    best = (scored.sort(["p", "s1_id"], descending=[True, False]).unique(["m_src", "m_id"], keep="first")
            .filter(pl.col("p") >= CONF).select("s1_id", pl.col("m_src").alias("c_src"),
                                                 pl.col("m_id").alias("c_id")))
    cols = ["m_src", "m_id", "name_core", "addr_nr"] + (["row"] if "row" in m.columns else [])
    rec = m.select(cols)
    pairs = (scored.select(KEY).join(best, on="s1_id")
             .filter(~((pl.col("m_src") == pl.col("c_src")) & (pl.col("m_id") == pl.col("c_id")))))
    log(f"  group pairs {pairs.height:,}")
    parts = []
    for i in range(0, pairs.height, CHUNK):
        x = (pairs.slice(i, CHUNK)
             .join(rec, on=["m_src", "m_id"], how="left")
             .join(rec.rename({c: f"{c}_c" for c in cols[2:]} | {"m_src": "c_src", "m_id": "c_id"}),
                   on=["c_src", "c_id"], how="left"))
        f = {
            "gn": process.cpdist(x["name_core"].to_list(), x["name_core_c"].to_list(),
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
            "ga": process.cpdist(x["addr_nr"].to_list(), x["addr_nr_c"].to_list(),
                                 scorer=fuzz.token_set_ratio, workers=-1, dtype=np.float32),
        }
        if emb_m is not None and "row" in x.columns:
            a = emb_m(x["m_src"].to_numpy(), x["row"].to_numpy())
            b = emb_m(x["c_src"].to_numpy(), x["row_c"].to_numpy())
            f["ge"] = np.einsum("ij,ij->i", a.astype(np.float32), b.astype(np.float32))
        parts.append(x.select(KEY).with_columns(**{k: pl.Series(v) for k, v in f.items()}))
    g = pl.concat(parts) if parts else pl.DataFrame(schema={**{k: pl.Int64 for k in KEY}})
    aggs = [pl.len().cast(pl.Int16).alias("grp_n"),
            pl.col("gn").max().alias("grp_name_max"), pl.col("gn").mean().alias("grp_name_mean"),
            pl.col("ga").max().alias("grp_addr_max"), pl.col("ga").mean().alias("grp_addr_mean")]
    if "ge" in g.columns:
        aggs += [pl.col("ge").max().alias("grp_ecos_max"), pl.col("ge").mean().alias("grp_ecos_mean")]
    g = g.group_by(KEY).agg(aggs) if g.height else g
    out = _p_features(scored).join(g, on=KEY, how="left").with_columns(pl.col("grp_n").fill_null(0))
    return out


def feature_columns(df):
    return [c for c in df.columns if c not in KEY and c != "label"]
