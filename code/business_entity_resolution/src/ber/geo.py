"""Coarse region (state / région) detection, country-agnostic.

A region is the address component (comma-separated part) with the highest
document frequency among a country's records: every region is shared by more
addresses than any city, street or house number inside it. No per-country
lists are needed, so unseen countries (France in test) work unchanged.
"""
import polars as pl


def _norm_part(expr):
    return expr.str.strip_chars().str.to_lowercase()


def address_parts(df):
    """Explode records into (row_idx, country, part) with normalised parts."""
    return (
        df.select(
            pl.int_range(pl.len(), dtype=pl.UInt32).alias("row_idx"),
            "country",
            pl.col("address").str.split(",").alias("part"),
        )
        .explode("part")
        .with_columns(_norm_part(pl.col("part")))
        .filter(pl.col("part") != "")
        .unique(["row_idx", "part"])
    )


def part_frequencies(df):
    """Fraction of a country's records whose address contains each part."""
    parts = address_parts(df)
    n = df.group_by("country").len("n_country")
    return (
        parts.group_by("country", "part").len("n")
        .join(n, on="country")
        .with_columns((pl.col("n") / pl.col("n_country")).alias("df"))
        .select("country", "part", "df")
    )


def assign_region(df, freqs=None, min_df=0.002):
    """Add a `region` column: the most frequent address part (>= min_df), else ''.

    `freqs` defaults to frequencies computed on `df` itself; pass the S1 table's
    frequencies to put S2/S3 records into the same vocabulary.
    """
    if freqs is None:
        freqs = part_frequencies(df)
    freqs = freqs.filter(pl.col("df") >= min_df)
    best = (
        address_parts(df)
        .join(freqs, on=["country", "part"], how="inner")
        .sort(["row_idx", "df"], descending=[False, True])
        .unique("row_idx", keep="first")
        .select("row_idx", pl.col("part").alias("region"))
    )
    return (
        df.with_row_index("row_idx")
        .join(best, on="row_idx", how="left")
        .drop("row_idx")
        .with_columns(pl.col("region").fill_null(""))
    )


def learn_region_lexicon(linked, min_count=20, min_purity=0.9):
    """Map noisy S2/S3 address parts to canonical S1 regions.

    `linked` has columns country, address (of the S2/S3 record) and region (of
    the S1 record it is linked to). A part maps to a region when it co-occurs
    with that region in >= min_purity of its >= min_count occurrences — this
    learns 'texas'->'tx', 'ಕರ್ನಾಟಕ'->'karnataka', city->state, etc.
    """
    counts = (
        address_parts(linked)
        .join(linked.select(pl.int_range(pl.len(), dtype=pl.UInt32).alias("row_idx"), "region"),
              on="row_idx")
        .filter(pl.col("region") != "")
        .group_by("country", "part", "region").len("n")
    )
    return (
        counts.with_columns(
            pl.col("n").sum().over("country", "part").alias("total"))
        .filter((pl.col("total") >= min_count) & (pl.col("n") / pl.col("total") >= min_purity))
        .select("country", "part", "region", pl.col("total").alias("support"))
    )


def assign_region_by_lexicon(df, lexicon):
    """Add `region` using the lexicon part with the largest support ('' if none)."""
    best = (
        address_parts(df)
        .join(lexicon, on=["country", "part"], how="inner")
        .sort(["row_idx", "support"], descending=[False, True])
        .unique("row_idx", keep="first")
        .select("row_idx", "region")
    )
    return (
        df.with_row_index("row_idx")
        .join(best, on="row_idx", how="left")
        .drop("row_idx")
        .with_columns(pl.col("region").fill_null(""))
    )
