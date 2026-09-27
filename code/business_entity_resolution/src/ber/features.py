"""Pair features for candidate (S1, S2/S3) pairs. All are country-agnostic.

Input: candidates from ber.blocking (s1_id, m_src, m_id, cos, acos, rank_name,
rank_addr, b1) and the record frames with the normalised columns plus
`region` and `addr_nr`. Output: one float32/int row per candidate pair.
"""
import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

KEY = ["s1_id", "m_src", "m_id"]
CHUNK = 4_000_000
# last filtering stage before the model: drop pairs whose name+address similarity is far
# below the best candidate of the same S2/S3 record (val: -70% pairs, F0.5 unchanged)
PRUNE_SIM_GAP = -60

_NUM = r"\b\d+\b"


def record_frame(df, id_cols):
    """Per-record columns used by the features (computed once per record)."""
    return df.select(
        *id_cols, "country", "name_core", "legal", "native", "addr_nr", "region",
        pl.col("addr_nr").str.extract_all(_NUM).list.unique().alias("nums"),
        (pl.col("addr_nr") == "").alias("addr_empty"),
        pl.col("name_core").str.len_chars().cast(pl.Int16).alias("name_len"),
        # how common this exact name / address is among the country's records of this side
        pl.len().over("country", "name_core").cast(pl.Int32).alias("name_freq"),
        pl.len().over("country", "addr_nr").cast(pl.Int32).alias("addr_freq"),
    )


def _cpdist(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def _chunk_features(c, r1, rm):
    x = c.join(r1, on="s1_id", how="left").join(rm, on=["m_src", "m_id"], how="left", suffix="_m")
    a, b = x["name_core"].to_list(), x["name_core_m"].to_list()
    aa, ab = x["addr_nr"].to_list(), x["addr_nr_m"].to_list()
    feats = {
        "n_ratio": _cpdist(a, b, fuzz.ratio),
        "n_tset": _cpdist(a, b, fuzz.token_set_ratio),
        "n_tsort": _cpdist(a, b, fuzz.token_sort_ratio),
        "n_partial": _cpdist(a, b, fuzz.partial_ratio),
        "n_jw": _cpdist(a, b, JaroWinkler.normalized_similarity),
        "a_tset": _cpdist(aa, ab, fuzz.token_set_ratio),
        "a_ratio": _cpdist(aa, ab, fuzz.ratio),
        "a_partial": _cpdist(aa, ab, fuzz.partial_ratio),
    }
    x = x.with_columns(**{k: pl.Series(v) for k, v in feats.items()})
    both_legal = (pl.col("legal") != "") & (pl.col("legal_m") != "")
    n_common = pl.col("nums").list.set_intersection(pl.col("nums_m")).list.len()
    return x.select(
        *KEY, "cos", "acos", "rank_name", "rank_addr", pl.col("b1").cast(pl.Int8),
        *[c for c in ("ecos", "rank_emb") if c in x.columns],
        *feats.keys(),
        (both_legal & (pl.col("legal") == pl.col("legal_m"))).cast(pl.Int8).alias("legal_eq"),
        (both_legal & (pl.col("legal") != pl.col("legal_m"))).cast(pl.Int8).alias("legal_diff"),
        ((pl.col("legal") == "") != (pl.col("legal_m") == "")).cast(pl.Int8).alias("legal_one"),
        n_common.cast(pl.Int8).alias("num_common"),
        (n_common / pl.max_horizontal(pl.col("nums").list.len(), pl.col("nums_m").list.len(), 1))
        .cast(pl.Float32).alias("num_jacc"),
        pl.col("nums").list.len().cast(pl.Int8).alias("nums_1"),
        pl.col("nums_m").list.len().cast(pl.Int8).alias("nums_m"),
        ((pl.col("region") == pl.col("region_m")) & (pl.col("region") != "")).cast(pl.Int8).alias("region_eq"),
        (pl.col("region_m") == "").cast(pl.Int8).alias("region_m_empty"),
        pl.col("addr_empty").cast(pl.Int8), pl.col("addr_empty_m").cast(pl.Int8),
        pl.col("native_m").cast(pl.Int8).alias("native_m"),
        "name_len", "name_len_m",
        "name_freq", "addr_freq",
    )


def _sim_expr():
    return pl.col("n_tset") + pl.col("a_tset")


def _emb_competition(f):
    """Encoder-similarity competition features (only when candidates carry ecos)."""
    if "ecos" not in f.columns:
        return f
    return f.with_columns(
        (pl.col("ecos") - pl.col("ecos").max().over("m_src", "m_id")).alias("ecos_gap_m"),
        pl.col("ecos").rank("min", descending=True).over("m_src", "m_id").cast(pl.Int16).alias("ecos_rank_m"),
    )


def _competition(f, s1_agg):
    """How this pair compares with the other candidates of the same record / entity.

    The chunk must hold every candidate of its S2/S3 records; entity-level
    aggregates come precomputed in `s1_agg` (s1_id, sim_max_1, n_cand_1).
    """
    return f.with_columns(_sim_expr().alias("sim")).join(s1_agg, on="s1_id", how="left").with_columns(
        (pl.col("sim") - pl.col("sim").max().over("m_src", "m_id")).alias("sim_gap_m"),
        (pl.col("sim").rank("min", descending=True).over("m_src", "m_id")).cast(pl.Int16).alias("sim_rank_m"),
        (pl.col("sim") - pl.col("sim_max_1")).alias("sim_gap_1"),
        (pl.col("cos") - pl.col("cos").max().over("m_src", "m_id")).alias("cos_gap_m"),
        (pl.col("acos") - pl.col("acos").max().over("m_src", "m_id")).alias("acos_gap_m"),
        pl.len().over("m_src", "m_id").cast(pl.Int16).alias("n_cand_m"),
        # other candidates of this record with (nearly) the same name as this S1: look-alikes
        ((pl.col("n_tset") >= 95).sum().over("m_src", "m_id")).cast(pl.Int16).alias("n_same_name_m"),
    ).drop("sim_max_1")


def _entity_aggregates(cands, r1, rm, log):
    """Pass 1: best name+address similarity and candidate count per S1 entity."""
    sims = []
    for i in range(0, cands.height, CHUNK):
        c = cands.slice(i, CHUNK).select("s1_id", "m_src", "m_id")
        x = (c.join(r1.select("s1_id", "name_core", "addr_nr"), on="s1_id", how="left")
             .join(rm.select("m_src", "m_id", "name_core", "addr_nr"), on=["m_src", "m_id"],
                   how="left", suffix="_m"))
        sim = (_cpdist(x["name_core"].to_list(), x["name_core_m"].to_list(), fuzz.token_set_ratio)
               + _cpdist(x["addr_nr"].to_list(), x["addr_nr_m"].to_list(), fuzz.token_set_ratio))
        sims.append(c.select("s1_id").with_columns(pl.Series("sim", sim)))
    log("  entity aggregates done")
    return (pl.concat(sims).group_by("s1_id")
            .agg(pl.col("sim").max().alias("sim_max_1"), pl.len().cast(pl.Int32).alias("n_cand_1")))


def build(cands, s1, m, log=print, sink=None):
    """Features for every candidate pair.

    Candidates are processed in buckets of whole S2/S3 records (so per-record
    competition features are exact); features are computed on all candidates,
    then pairs below PRUNE_SIM_GAP are dropped. With `sink`, each bucket's frame is passed
    to sink(frame) and nothing is returned (bounded memory for test);
    otherwise the concatenated frame is returned.
    """
    r1 = record_frame(s1, ["s1_id"])
    rm = record_frame(m, ["m_src", "m_id"]).drop("country")
    s1_agg = _entity_aggregates(cands, r1, rm, log)
    n_buckets = max(1, -(-cands.height // CHUNK))
    bucket = (pl.col("m_id").hash(7) + pl.col("m_src").cast(pl.UInt64)) % n_buckets
    out = []
    for b in range(n_buckets):
        c = cands.filter(bucket == b)
        f = _competition(_chunk_features(c, r1, rm), s1_agg)
        f = _emb_competition(f)
        keep = pl.col("sim_gap_m") >= PRUNE_SIM_GAP
        if "ecos_rank_m" in f.columns:     # never prune what the name encoder ranks top-2
            keep = keep | (pl.col("ecos_rank_m") <= 2)
        f = f.filter(keep)
        if sink is None:
            out.append(f)
        else:
            sink(f)
        log(f"  features bucket {b + 1}/{n_buckets}")
    return pl.concat(out) if sink is None else None


def feature_columns(df):
    return [c for c in df.columns if c not in KEY and c != "label"]
