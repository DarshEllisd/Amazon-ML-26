"""Step 0: convert every raw TSV into the parquet cache (int ids, string fields).

    python src/prepare_data.py
"""
import time

import polars as pl

from ber import config
from ber.io import read_tsv, split_entity_id


def convert_source(split, src):
    t = time.time()
    df = read_tsv(config.raw_path(split, src))
    assert df.columns == ["entity_id", "business_name", "business_address", "country"], df.columns
    out = df.select(
        split_entity_id("entity_id", src).alias("id"),
        pl.col("business_name").alias("name"),
        pl.col("business_address").alias("address"),
        pl.col("country"),
    )
    # ids must round-trip exactly (no leading zeros lost) and be unique
    roundtrip = (pl.lit(f"S{src}-") + out["id"].cast(pl.Utf8)) == df["entity_id"]
    assert out.select(roundtrip.all()).item(), f"{split} s{src}: id does not round-trip"
    assert out["id"].n_unique() == out.height, f"{split} s{src}: duplicate ids"
    path = config.records_path(split, src)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(path)
    print(f"{split} s{src}: {out.height:,} rows -> {path.name} ({time.time() - t:.0f}s)")


def convert_ground_truth():
    gt = read_tsv(config.gt_raw_path())
    assert gt.columns == ["source1_entity_id", "matched_entity_ids"], gt.columns
    gt = gt.with_columns(split_entity_id("source1_entity_id", 1).alias("s1_id"))
    pairs = (
        gt.filter(pl.col("matched_entity_ids") != "")
        .select("s1_id", pl.col("matched_entity_ids").str.split(",").alias("m"))
        .explode("m")
        .select(
            "s1_id",
            pl.col("m").str.slice(1, 1).cast(pl.Int8).alias("m_src"),
            pl.col("m").str.slice(3).cast(pl.Int64).alias("m_id"),
        )
    )
    entities = gt.select(
        "s1_id",
        pl.when(pl.col("matched_entity_ids") == "").then(0)
        .otherwise(pl.col("matched_entity_ids").str.count_matches(",") + 1)
        .cast(pl.Int16).alias("n_matches"),
    )
    assert pairs.select(pl.col("m_src").is_in([2, 3]).all()).item()
    assert entities["n_matches"].sum() == pairs.height
    pairs.write_parquet(config.gt_pairs_path())
    entities.write_parquet(config.gt_entities_path())
    print(f"ground truth: {entities.height:,} entities, {pairs.height:,} pairs")


def main():
    for split in config.SPLITS:
        for src in config.SOURCES:
            convert_source(split, src)
    convert_ground_truth()


if __name__ == "__main__":
    main()
