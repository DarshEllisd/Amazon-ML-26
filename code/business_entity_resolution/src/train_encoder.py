"""Step 3: fine-tune the multilingual name encoder and embed every record's name.

    python src/train_encoder.py            # train on fit-fold pairs, then encode all
    python src/train_encoder.py --encode   # only (re-)encode with the saved model

Training uses true pairs whose S1 entity is in the *fit* fold only, so the
validation regions stay unseen. Writes work/models/name_encoder/ and
work/emb/{split}_s{src}.npy (rows aligned with work/parquet/{split}_s{src}).
"""
import argparse
import os
import time

import numpy as np
import polars as pl

from ber import config, embed
from ber.io import load_gt_pairs, load_records

N_PAIRS = 3_000_000
NATIVE_REPEAT = 2      # native-script pairs are the hard, under-served case


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def training_pairs(all_train=False):
    s1 = load_records("train", 1, ["id", "name"])
    if not all_train:   # validation experiments: fit fold only
        fit = pl.read_parquet(config.WORK_DIR / "split" / "s1.parquet").filter(pl.col("fold") == "fit")
        s1 = s1.join(fit.select("id"), on="id", how="semi")
    gt = load_gt_pairs().join(s1.select(pl.col("id").alias("s1_id")), on="s1_id", how="semi")
    ms = pl.concat([
        load_records("train", s, ["id", "name"]).join(
            pl.read_parquet(config.norm_path("train", s), columns=["id", "native"]), on="id")
        .with_columns(pl.lit(s, pl.Int8).alias("m_src")) for s in (2, 3)])
    pairs = (gt.join(s1.rename({"id": "s1_id", "name": "a"}), on="s1_id")
             .join(ms.rename({"id": "m_id", "name": "b"}), on=["m_src", "m_id"])
             .filter((pl.col("a") != "") & (pl.col("b") != "")))
    native = pairs.filter("native")
    latin = pairs.filter(~pl.col("native"))
    n_latin = max(0, min(latin.height, N_PAIRS - NATIVE_REPEAT * native.height))
    sample = pl.concat([native] * NATIVE_REPEAT + [latin.sample(n_latin, seed=config.SEED)])
    log(f"pairs: {pairs.height:,} fit-fold true pairs ({native.height:,} native-script); "
        f"training on {sample.height:,}")
    return sample["a"].to_list(), sample["b"].to_list()


def encode_all():
    embed.emb_path("train", 1).parent.mkdir(parents=True, exist_ok=True)   # tagged dir (BER_TAG)
    for split in config.SPLITS:
        for src in config.SOURCES:
            names = load_records(split, src, ["name"])["name"]
            uniq = names.unique(maintain_order=True)
            inv = (names.to_frame().with_row_index("i")
                   .join(uniq.to_frame().with_row_index("u"), on="name", how="left")
                   .sort("i")["u"].to_numpy())
            log(f"{split} s{src}: {len(names):,} names, {len(uniq):,} distinct")
            e = embed.encode(uniq.to_list(), log=log)
            path = embed.emb_path(split, src)
            tmp = path.with_name(path.stem + ".tmp.npy")
            np.save(tmp, e[inv])
            os.replace(tmp, path)          # readers never see a half-written file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", action="store_true", help="skip training")
    ap.add_argument("--all-train", action="store_true",
                    help="final model: train on every training pair (val regions included)")
    a = ap.parse_args()
    if not a.encode:
        pa, pb = training_pairs(a.all_train)
        embed.train(pa, pb, log=log)
    encode_all()


if __name__ == "__main__":
    main()
