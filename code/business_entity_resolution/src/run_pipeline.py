"""Blocking -> features -> scoring -> assignment on one "universe" of records.

    python src/run_pipeline.py --run mini  --stage all   # fast dev loop (58k S1)
    python src/run_pipeline.py --run fit   --stage all   # training universe (fit-fold regions)
    python src/run_pipeline.py --run val   --stage all   # hold-out numbers
    python src/run_pipeline.py --run test  --stage all   # writes output/*.tsv

Universes: `test` is the whole test set. The train universes come from
work/split: `val` = held-out regions, `mini` = its smallest region per country,
`fit` = a random ~10% (by S1) of the fit-fold regions per country. Each
universe keeps all S1/S2/S3 records of its regions, distractors included.
`predict` needs a model trained by `--run fit --stage train`.

Requires src/prepare_data.py, src/make_split.py and src/normalize_data.py.
"""
import argparse
import random
import time

import polars as pl

from ber import config
from ber.io import load_gt_pairs

FIT_FRACTION = 0.10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------- universes

def _fit_regions():
    s1 = pl.read_parquet(config.WORK_DIR / "split" / "s1.parquet").filter(pl.col("fold") == "fit")
    s1 = s1.join(pl.read_parquet(config.norm_path("train", 1), columns=["id", "country"]), on="id")
    sizes = (s1.filter(pl.col("region") != "").group_by("country", "region").len("n")
             .with_columns((pl.col("n") / pl.col("n").sum().over("country")).alias("share"))
             .filter(pl.col("share") >= 0.005).sort("country", "region"))
    rng = random.Random(config.SEED)
    chosen = []
    for country in sizes["country"].unique().sort():
        rows = sizes.filter(pl.col("country") == country).rows(named=True)
        rng.shuffle(rows)
        share = 0.0
        for r in rows:
            if share >= FIT_FRACTION:
                break
            if share + r["share"] > 2 * FIT_FRACTION:
                continue
            chosen.append(r["region"])
            share += r["share"]
    return chosen


def load_universe(run):
    """Return (split, s1, m) with normalised columns; ids renamed s1_id / (m_src, m_id)."""
    split = "test" if run == "test" else "train"
    s1 = pl.read_parquet(config.norm_path(split, 1))
    ms = [pl.read_parquet(config.norm_path(split, s)).with_columns(pl.lit(s, pl.Int8).alias("m_src"))
          for s in (2, 3)]
    if run != "test":
        sd = config.WORK_DIR / "split"
        sp1 = pl.read_parquet(sd / "s1.parquet")
        if run == "val":
            keep1 = sp1.filter(pl.col("fold") == "val")
            regions = None
        elif run == "mini":
            keep1 = sp1.filter(pl.col("mini"))
            regions = keep1["region"].unique().to_list()
        elif run == "fit":
            regions = _fit_regions()
            keep1 = sp1.filter((pl.col("fold") == "fit") & pl.col("region").is_in(regions))
        else:
            raise ValueError(run)
        fold = "fit" if run == "fit" else "val"
        s1 = s1.join(keep1.select("id"), on="id", how="semi")
        for i, s in enumerate((2, 3)):
            sp = pl.read_parquet(sd / f"s{s}.parquet").filter(pl.col("fold") == fold)
            if regions is not None:
                sp = sp.filter(pl.col("region").is_in(regions))
            ms[i] = ms[i].join(sp.select("id"), on="id", how="semi")
    s1 = s1.rename({"id": "s1_id"})
    m = pl.concat(ms).rename({"id": "m_id"})
    return split, s1, m


def universe_gt(s1, m):
    gt = load_gt_pairs()
    return (gt.join(s1.select("s1_id"), on="s1_id", how="semi")
            .join(m.select("m_src", "m_id"), on=["m_src", "m_id"], how="semi"))


# ---------------------------------------------------------------- stages

def stage_block(run):
    from ber.blocking import generate
    from ber.evaluate import blocking_report

    split, s1, m = load_universe(run)
    log(f"{run}: {s1.height:,} S1, {m.height:,} S2/S3")
    d = config.run_dir(run)
    cdir = d / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.parquet"):
        old.unlink()
    _, s1r, mr, lexicon = generate(
        s1, m, log=log, sink=lambda country, c: c.write_parquet(cdir / f"{country}.parquet"))
    cands = pl.scan_parquet(cdir / "*.parquet")
    s1r.select("s1_id", "region", "addr_nr").write_parquet(d / "s1_region.parquet")
    mr.select("m_src", "m_id", "region", "addr_nr").write_parquet(d / "m_region.parquet")
    lexicon.write_parquet(d / "lexicon.parquet")
    n_cands = cands.select(pl.len()).collect().item()
    log(f"candidates {n_cands:,} ({n_cands / s1.height:.1f} per S1)")
    if split == "train":
        cands = cands.collect()
        gt = universe_gt(s1, m)
        ents = s1.select("s1_id", "country")
        for country in ["ALL"] + sorted(ents["country"].unique().to_list()):
            e = ents if country == "ALL" else ents.filter(pl.col("country") == country)
            r = blocking_report(cands, gt, e)
            log(f"  {country:6s} " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v:,}"
                                                for k, v in r.items()))
        # recall by pass and for native-script records
        hit = gt.join(cands, on=["s1_id", "m_src", "m_id"], how="left").join(
            m.select("m_src", "m_id", "native"), on=["m_src", "m_id"])
        log("  recall b1={:.4f} b2={:.4f} b6={:.4f} native={:.4f} latin={:.4f}".format(
            hit["b1"].fill_null(False).mean(), (hit["rank_name"] < 99).mean(), (hit["rank_addr"] < 99).mean(),
            hit.filter("native")["cos"].is_not_null().mean(),
            hit.filter(~pl.col("native"))["cos"].is_not_null().mean()))


def load_with_regions(run):
    split, s1, m = load_universe(run)
    d = config.run_dir(run)
    s1 = s1.join(pl.read_parquet(d / "s1_region.parquet"), on="s1_id")
    m = m.join(pl.read_parquet(d / "m_region.parquet"), on=["m_src", "m_id"])
    return split, s1, m


def _load_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_DIR / "m1.lgb"))


def _score(model, f):
    p = model.predict(f.select(model.feature_name()).to_numpy().astype("float32"), num_threads=12)
    return f.select("s1_id", "m_src", "m_id").with_columns(pl.Series("p", p.astype("float32")))


def stage_features(run):
    """Train universes: write features.parquet. Test: score each bucket as it is
    built (the full test feature table would not fit in RAM or on disk) and
    write scored.parquet directly."""
    from ber.features import build

    split, s1, m = load_with_regions(run)
    d = config.run_dir(run)
    files = sorted((d / "candidates").glob("*.parquet"))
    if split == "test":
        model, scored = _load_model(), []
        for fp in files:
            log(f"scoring {fp.stem}")
            cands = pl.read_parquet(fp)
            build(cands, s1.filter(pl.col("country") == fp.stem), m.filter(pl.col("country") == fp.stem),
                  log=log, sink=lambda f: scored.append(_score(model, f)))
            del cands
        scored = pl.concat(scored)
        scored.write_parquet(d / "scored.parquet")
        log(f"scored {scored.height:,} pairs")
        return
    feats = build(pl.concat([pl.read_parquet(fp) for fp in files]), s1, m, log=log)
    gt = universe_gt(s1, m).with_columns(pl.lit(1, pl.Int8).alias("label"))
    feats = feats.join(gt, on=["s1_id", "m_src", "m_id"], how="left").with_columns(
        pl.col("label").fill_null(0))
    log(f"positives {feats['label'].sum():,} / {feats.height:,}")
    feats.write_parquet(d / "features.parquet")
    log(f"{feats.width} columns, {feats.height:,} rows")


MODEL_DIR = config.WORK_DIR / "models"
MAX_NEG = 8_000_000
LGB_PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=12, seed=config.SEED)
N_ROUNDS = 600


def stage_train(run):
    import lightgbm as lgb
    from ber.features import feature_columns

    assert run == "fit", "train on the fit universe"
    f = pl.read_parquet(config.run_dir(run) / "features.parquet")
    pos, neg = f.filter(pl.col("label") == 1), f.filter(pl.col("label") == 0)
    if neg.height > MAX_NEG:
        neg = neg.sample(MAX_NEG, seed=config.SEED)
    f = pl.concat([pos, neg])
    cols = feature_columns(f)
    # hold out 5% of S1 entities for early stopping
    is_es = (pl.col("s1_id").hash(config.SEED) % 20) == 0
    tr, es = f.filter(~is_es), f.filter(is_es)
    log(f"train {tr.height:,} rows ({tr['label'].mean():.1%} positive), {len(cols)} features")
    dtr = lgb.Dataset(tr.select(cols).to_numpy().astype("float32"), tr["label"].to_numpy(), feature_name=cols)
    des = lgb.Dataset(es.select(cols).to_numpy().astype("float32"), es["label"].to_numpy(), reference=dtr)
    model = lgb.train(LGB_PARAMS, dtr, N_ROUNDS, valid_sets=[des],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / "m1.lgb"))
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{c}={g:.0f}" for c, g in imp[:12]))


def stage_predict(run):
    import json

    from ber.assign import matches, tune_tau
    from ber.evaluate import report
    from ber.io import write_pair_table

    split, s1, m = load_universe(run)
    d = config.run_dir(run)
    if split == "test":
        scored = pl.read_parquet(d / "scored.parquet")
    else:
        scored = _score(_load_model(), pl.read_parquet(d / "features.parquet"))
        scored.write_parquet(d / "scored.parquet")
    tau_path = MODEL_DIR / "m1_tau.json"
    if split == "train":
        gt = universe_gt(s1, m)
        ents = s1.select("s1_id", "country")
        tau, curve = tune_tau(scored, gt, ents)
        log("tau curve: " + " ".join(f"{t:.2f}:{v:.4f}" for t, v in curve))
        if run == "val":
            tau_path.write_text(json.dumps({"tau": tau}))
        print(report(matches(scored, tau), gt, ents, by="country"))
    else:
        tau = json.loads(tau_path.read_text())["tau"]
        pred = matches(scored, tau)
        cand = scored.select("s1_id", "m_src", "m_id")
        assert pred.join(cand, on=["s1_id", "m_src", "m_id"], how="anti").height == 0
        write_pair_table(config.OUTPUT_DIR / "matching_results.tsv", s1["s1_id"], pred, "matched_entity_ids")
        write_pair_table(config.OUTPUT_DIR / "candidate_pairs.tsv", s1["s1_id"], cand, "candidate_entity_ids")
        per = pred.group_by("s1_id").len().join(s1.select("s1_id", "country"), on="s1_id", how="right")
        log(f"tau={tau}; wrote {pred.height:,} matches, {cand.height:,} candidates")
        print(per.group_by("country").agg(pl.col("len").fill_null(0).mean().alias("matches_per_s1"),
                                          pl.col("len").is_null().mean().alias("empty_share")))


STAGES = {"block": stage_block, "features": stage_features, "train": stage_train, "predict": stage_predict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["mini", "val", "fit", "test"])
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    a = ap.parse_args()
    if a.stage == "all":
        stages = ["block", "features"] + (["train"] if a.run == "fit" else []) + (["predict"] if a.run != "fit" else [])
    else:
        stages = [a.stage]
    for s in stages:
        t = time.time()
        log(f"== {a.run}: {s}")
        STAGES[s](a.run)
        log(f"== {s} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
