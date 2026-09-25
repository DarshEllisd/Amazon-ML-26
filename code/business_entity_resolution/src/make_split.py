"""Step 1: geographic train/validation split of the training data.

Whole regions (states) are held out with every S1, S2 and S3 record in them,
so validation keeps the real candidate density and distractor rate.

    python src/make_split.py

Writes work/split/s{1,2,3}.parquet with columns id, region, fold where fold is
'fit' or 'val', and s1 additionally has `mini` (a small val subset for fast
iteration: the smallest held-out region per country).
"""
import random

import polars as pl

from ber import config
from ber.geo import assign_region, assign_region_by_lexicon, learn_region_lexicon
from ber.io import load_gt_pairs, load_records

VAL_FRACTION = 0.10
MIN_REGION_SHARE = 0.005  # noisy pseudo-regions ('unit 1', 'atlanta') stay in fit


def choose_val_regions(s1):
    """Per country, shuffle real regions and take them until >= VAL_FRACTION of S1."""
    rng = random.Random(config.SEED)
    sizes = (
        s1.group_by("country", "region").len("n")
        .with_columns((pl.col("n") / pl.col("n").sum().over("country")).alias("share"))
        .filter((pl.col("region") != "") & (pl.col("share") >= MIN_REGION_SHARE))
        .sort("country", "region")
    )
    val, mini = [], []
    for country in sizes["country"].unique().sort():
        rows = sizes.filter(pl.col("country") == country).rows(named=True)
        rng.shuffle(rows)
        taken, share = [], 0.0
        for r in rows:
            if share >= VAL_FRACTION:
                break
            # skip regions that alone would overshoot badly (e.g. Maharashtra ~22%)
            if share + r["share"] > 2 * VAL_FRACTION:
                continue
            taken.append(r)
            share += r["share"]
        val += [(country, r["region"]) for r in taken]
        mini += [(country, r["region"]) for r in sorted(taken, key=lambda r: r["n"])[:1]]
        print(f"{country}: val regions {[r['region'] for r in taken]} = {share:.1%} of S1")
    return [f"{c}|{r}" for c, r in val], [f"{c}|{r}" for c, r in mini]


def region_key():
    return pl.col("country") + "|" + pl.col("region")


def main():
    s1 = assign_region(load_records("train", 1))
    val_regions, mini_regions = choose_val_regions(s1)
    s1 = s1.with_columns(
        region_key().is_in(val_regions).alias("is_val"),
        region_key().is_in(mini_regions).alias("mini"),
    )

    gt = load_gt_pairs()
    out_dir = config.WORK_DIR / "split"
    out_dir.mkdir(parents=True, exist_ok=True)
    s1.select("id", "region", pl.when("is_val").then(pl.lit("val")).otherwise(pl.lit("fit")).alias("fold"),
              "mini").write_parquet(out_dir / "s1.parquet")

    for src in (2, 3):
        rec = load_records("train", src)
        owner = (
            gt.filter(pl.col("m_src") == src)
            .join(s1.select(pl.col("id").alias("s1_id"), "region", "is_val"), on="s1_id")
            .select(pl.col("m_id").alias("id"), "region", "is_val")
        )
        rec = rec.join(owner, on="id", how="left")
        matched = rec.filter(pl.col("is_val").is_not_null())
        lexicon = learn_region_lexicon(matched.select("country", "address", "region"))
        # unmatched distractors: region from their own address via the lexicon
        unmatched = assign_region_by_lexicon(
            rec.filter(pl.col("is_val").is_null()).select("id", "country", "address"), lexicon)
        unmatched = unmatched.with_columns(
            region_key().is_in(val_regions).alias("is_val"))
        # no usable address: hold out a random VAL_FRACTION so val keeps its share
        no_region = unmatched["region"] == ""
        unmatched = unmatched.with_columns(
            pl.when(no_region)
            .then((pl.col("id").hash(config.SEED) % 1000) < int(1000 * VAL_FRACTION))
            .otherwise(pl.col("is_val")).alias("is_val"))
        both = pl.concat([matched.select("id", "region", "is_val"),
                          unmatched.select("id", "region", "is_val")])
        assert both.height == rec.height
        both.select("id", "region",
                    pl.when("is_val").then(pl.lit("val")).otherwise(pl.lit("fit")).alias("fold")
                    ).write_parquet(out_dir / f"s{src}.parquet")
        print(f"s{src}: lexicon {lexicon.height:,} parts; unmatched {unmatched.height:,} "
              f"(no region {no_region.mean():.1%}); val share {both['is_val'].mean():.1%}")

    val = s1.filter("is_val")
    print(f"s1 val: {val.height:,} entities; mini: {s1['mini'].sum():,}")


if __name__ == "__main__":
    main()
