"""Loading raw TSVs, the parquet cache, and writing submission files.

IDs are stored as (src: int8, id: int64) — the numeric part of `S{src}-{id}` —
because S2 and S3 numeric parts may collide and strings cost ~10x the memory.
"""
import polars as pl

from . import config


def read_tsv(path):
    """Read a challenge TSV with every column as a string; no quoting, empty stays ''."""
    return pl.read_csv(
        path, separator="\t", quote_char=None, infer_schema_length=0,
        missing_utf8_is_empty_string=True, encoding="utf8",
    )


def split_entity_id(col, src):
    """Expression turning 'S{src}-123' into Int64 123 (errors if the prefix is wrong)."""
    prefix = f"S{src}-"
    return pl.col(col).str.strip_prefix(prefix).cast(pl.Int64, strict=True)


def format_id(src, ids):
    """Vectorised inverse of split_entity_id as a polars expression/series."""
    return pl.lit(f"S{src}-") + ids.cast(pl.Utf8)


def load_records(split, src, columns=None):
    return pl.read_parquet(config.records_path(split, src), columns=columns)


def load_gt_pairs():
    """Long-format ground truth: s1_id, m_src (2/3), m_id."""
    return pl.read_parquet(config.gt_pairs_path())


def load_gt_entities():
    """One row per train S1 entity: s1_id, n_matches."""
    return pl.read_parquet(config.gt_entities_path())


def write_id_lists(path, s1_ids, lists, header):
    """Write a submission-style TSV.

    s1_ids: iterable of int S1 ids; lists: matching iterable of iterables of
    (src, id) tuples. Rows for every s1 id are written, empty lists as ''.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{header}\n")
        for s1, lst in zip(s1_ids, lists):
            ids = ",".join(dict.fromkeys(f"S{s}-{i}" for s, i in lst))
            f.write(f"S1-{s1}\t{ids}\n")


def write_pair_table(path, s1_ids, pairs, header):
    """Vectorised submission writer for large pair sets.

    s1_ids: Series of every int S1 id (one output row each, '' when no pairs);
    pairs: frame with s1_id, m_src, m_id. Lists are deduplicated and sorted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lists = (
        pairs.select("s1_id", "m_src", "m_id").unique().sort("s1_id", "m_src", "m_id")
        .select("s1_id", (pl.lit("S") + pl.col("m_src").cast(pl.Utf8) + pl.lit("-")
                          + pl.col("m_id").cast(pl.Utf8)).alias("m"))
        .group_by("s1_id", maintain_order=True).agg(pl.col("m").str.join(",").alias(header))
    )
    out = (
        pl.DataFrame({"s1_id": s1_ids}).join(lists, on="s1_id", how="left")
        .select((pl.lit("S1-") + pl.col("s1_id").cast(pl.Utf8)).alias("source1_entity_id"),
                pl.col(header).fill_null(""))
    )
    out.write_csv(path, separator="\t", quote_style="never", line_terminator="\n")
