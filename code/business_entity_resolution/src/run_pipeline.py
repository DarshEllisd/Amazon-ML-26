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

Test runs one country per subprocess (block + features), so each process
holds only that country's records; results go to work/runs/test/country/<C>/
and `predict` merges them. `--country C` runs a single country directly.

Requires src/prepare_data.py, src/make_split.py and src/normalize_data.py.
"""
import argparse
import os
import random
import subprocess
import sys
import time

import polars as pl

from ber import config
from ber.io import load_gt_pairs

FIT_FRACTION = 0.10
COUNTRY = None  # set by --country: restrict the universe to one country


def _is_train_run(run):
    """Universes that models are trained on."""
    return run.startswith("fit") or run == "all"


def rdir(run):
    """Directory of this run's artefacts (per country when --country is set)."""
    d = config.run_dir(run)
    return d / "country" / COUNTRY if COUNTRY else d


def _read_norm(split, src):
    # `row` = position in the record parquet, used to look up name embeddings
    lf = pl.scan_parquet(config.norm_path(split, src)).with_row_index("row")
    if COUNTRY:
        lf = lf.filter(pl.col("country") == COUNTRY)
    return lf.collect()


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
    s1 = _read_norm(split, 1)
    ms = [_read_norm(split, s).with_columns(pl.lit(s, pl.Int8).alias("m_src")) for s in (2, 3)]
    if run not in ("test", "all"):   # "all" = every training record (final model)
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
        elif run == "fitall":
            regions = None
            keep1 = sp1.filter(pl.col("fold") == "fit")
        else:
            raise ValueError(run)
        fold = "fit" if run.startswith("fit") else "val"
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
    d = rdir(run)
    cdir = d / "candidates"
    cdir.mkdir(parents=True, exist_ok=True)
    for old in cdir.glob("*.parquet"):
        old.unlink()
    _, s1r, mr, lexicon = generate(
        s1, m, log=log, sink=lambda country, c: c.write_parquet(cdir / f"{country}.parquet"),
        split=split)
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
        if "rank_emb" in hit.columns:
            log("  recall b5={:.4f}".format((hit["rank_emb"] < 99).mean()))
        rate = lambda x: float("nan") if x is None else x   # e.g. no native-script records in the US
        log("  recall b1={:.4f} b2={:.4f} b6={:.4f} native={:.4f} latin={:.4f}".format(*map(rate, (
            hit["b1"].fill_null(False).mean(), (hit["rank_name"] < 99).mean(), (hit["rank_addr"] < 99).mean(),
            hit.filter("native")["cos"].is_not_null().mean(),
            hit.filter(~pl.col("native"))["cos"].is_not_null().mean()))))


def load_with_regions(run):
    split, s1, m = load_universe(run)
    d = rdir(run)
    if (d / "s1_region.parquet").exists():
        r1, rm = pl.read_parquet(d / "s1_region.parquet"), pl.read_parquet(d / "m_region.parquet")
    else:   # blocked one country per subprocess (fitall): merge the per-country files
        r1 = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("country/*/s1_region.parquet"))])
        rm = pl.concat([pl.read_parquet(p) for p in sorted(d.glob("country/*/m_region.parquet"))])
    s1 = s1.join(r1, on="s1_id")
    m = m.join(rm, on=["m_src", "m_id"])
    return split, s1, m


def _load_model():
    import lightgbm as lgb
    return lgb.Booster(model_file=str(MODEL_DIR / f"{MODEL_NAME}.lgb"))


def _score(model, f):
    p = model.predict(f.select(model.feature_name()).to_numpy().astype("float32"), num_threads=LGB_THREADS)
    keep = ["s1_id", "m_src", "m_id"] + [c for c in ("ecos_rank_m",) if c in f.columns]
    return f.select(keep).with_columns(pl.Series("p", p.astype("float32")))


def stage_features(run):
    """Train universes: write features.parquet. Test: score each bucket as it is
    built (the full test feature table would not fit in RAM or on disk) and
    write scored.parquet directly."""
    from ber.features import build

    split, s1, m = load_with_regions(run)
    d = rdir(run)
    files = sorted((d / "candidates").glob("*.parquet")) + sorted(d.glob("country/*/candidates/*.parquet"))
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
LGB_THREADS = min(config.N_JOBS, 32)   # LightGBM slows down with too many threads
MODEL_NAME = "m1"     # set by --model
RERANK_MAX = int(os.environ.get("BER_RERANK_MAX", 3_000_000))
MAX_NEG = int(os.environ.get("BER_MAX_NEG", 8_000_000))
LGB_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=255, min_data_in_leaf=100,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  verbose=-1, num_threads=LGB_THREADS, seed=config.SEED)
N_ROUNDS = int(os.environ.get("BER_ROUNDS", 3000))


def stage_train(run):
    import lightgbm as lgb
    from ber.features import feature_columns

    assert _is_train_run(run), "train on a fit universe"
    f = pl.read_parquet(rdir(run) / "features.parquet")
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
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_DIR / f"{MODEL_NAME}.lgb"))
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    log("top features: " + ", ".join(f"{c}={g:.0f}" for c, g in imp[:12]))
    # out-of-fold stage-1 scores on this universe: the training input of stage 2
    rounds = model.best_iteration or N_ROUNDS
    full = pl.read_parquet(rdir(run) / "features.parquet")
    half = lambda df: (df["s1_id"].hash(config.SEED + 1) % 2).to_numpy()
    h_tr, h_full, oof = half(f), half(full), []
    for k in (0, 1):
        sel = h_tr != k
        mk = lgb.train(LGB_PARAMS, lgb.Dataset(f.select(cols).to_numpy().astype("float32")[sel],
                                               f["label"].to_numpy()[sel]), rounds)
        part = full.filter(pl.Series(h_full == k))
        oof.append(part.select("s1_id", "m_src", "m_id", *[c for c in ("ecos_rank_m",) if c in part.columns])
                   .with_columns(pl.Series(
            "p", mk.predict(part.select(cols).to_numpy().astype("float32"),
                            num_threads=LGB_THREADS).astype("float32"))))
        log(f"  out-of-fold half {k} scored {part.height:,}")
    pl.concat(oof).write_parquet(rdir(run) / "scored_oof.parquet")


def _emb_lookup(split):
    """(m_src array, row array) -> name embeddings, or None without embeddings."""
    from ber.embed import emb_path, load_embeddings
    import numpy as np
    if os.environ.get("BER_NO_EMB") or not all(emb_path(split, s).exists() for s in (2, 3)):
        return None
    mm = {s: load_embeddings(split, s) for s in (2, 3)}

    def get(src, rows):
        out = np.empty((len(rows), mm[2].shape[1]), dtype=np.float16)
        for s in (2, 3):
            sel = np.flatnonzero(src == s)
            order = np.argsort(rows[sel])
            out[sel[order]] = mm[s][rows[sel][order]]
        return out
    return get


def _pair_texts(split, pairs):
    """'name | address' raw texts of both sides of each pair (native scripts kept)."""
    from ber.io import load_records
    rec1 = load_records(split, 1, ["id", "name", "address"]).join(
        pairs.select(pl.col("s1_id").alias("id")).unique(), on="id", how="semi")
    recm = pl.concat([load_records(split, s, ["id", "name", "address"])
                      .with_columns(pl.lit(s, pl.Int8).alias("m_src")) for s in (2, 3)]).join(
        pairs.select("m_src", pl.col("m_id").alias("id")).unique(), on=["m_src", "id"], how="semi")
    txt = pl.col("name") + pl.lit(" | ") + pl.col("address")
    x = (pairs.join(rec1.select(pl.col("id").alias("s1_id"), txt.alias("a")), on="s1_id", how="left")
         .join(recm.select("m_src", pl.col("id").alias("m_id"), txt.alias("b")), on=["m_src", "m_id"],
               how="left"))
    return x["a"].fill_null("").to_list(), x["b"].fill_null("").to_list()


def _rerank_frame(split, scored):
    """Cross-encoder logits `xe` for the uncertain band of stage-1 scores (or None)."""
    from ber import rerank
    if os.environ.get("BER_NO_RERANK") or not (rerank.model_dir() / "head.pt").exists():
        return None
    band = scored.filter(rerank.in_band(pl, scored.columns)).select("s1_id", "m_src", "m_id")
    log(f"  reranking {band.height:,} band pairs of {scored.height:,}")
    a, b = _pair_texts(split, band)
    return band.with_columns(pl.Series("xe", rerank.score(a, b, log=log)))


def stage_rerank(run):
    """Train the cross-encoder on one half of the fit entities, score the other half."""
    from ber import rerank
    assert _is_train_run(run), "train the reranker on a fit universe"
    split = "train"
    d = rdir(run)
    oof = pl.read_parquet(d / "scored_oof.parquet")
    band = oof.filter(rerank.in_band(pl, oof.columns))
    gt = load_gt_pairs().with_columns(pl.lit(1.0).alias("y"))
    band = band.join(gt, on=["s1_id", "m_src", "m_id"], how="left").with_columns(pl.col("y").fill_null(0.0))
    tr = band.filter(rerank.half_expr(pl) == rerank.RERANK_HALF)
    if tr.height > RERANK_MAX:
        tr = tr.sample(RERANK_MAX, seed=config.SEED)
    log(f"rerank: band {band.height:,} pairs; training on {tr.height:,} ({tr['y'].mean():.1%} positive)")
    a, b = _pair_texts(split, tr)
    rerank.train(a, b, tr["y"].to_list(), log=log)
    other = oof.filter(rerank.half_expr(pl) != rerank.RERANK_HALF)
    xe = _rerank_frame(split, other)
    xe.write_parquet(d / "rerank_oof.parquet")


def _group_frame(split, scored, m, extra=None):
    from ber import group
    return group.build(scored, m, _emb_lookup(split), log=log, extra=extra)


def stage_group(run):
    """Train the stage-2 group model on out-of-fold stage-1 scores."""
    import lightgbm as lgb
    from ber.group import feature_columns

    assert _is_train_run(run), "train stage 2 on a fit universe"
    split, s1, m = load_with_regions(run)
    d = rdir(run)
    scored = pl.read_parquet(d / "scored_oof.parquet")
    extra = None
    if (d / "rerank_oof.parquet").exists() and not os.environ.get("BER_NO_RERANK"):
        from ber import rerank
        extra = pl.read_parquet(d / "rerank_oof.parquet")
    g = _group_frame(split, scored, m, extra)
    if extra is not None:     # rows whose entity trained the reranker would carry in-sample xe
        g = g.filter(rerank.half_expr(pl) != rerank.RERANK_HALF)
    gt = universe_gt(s1, m).with_columns(pl.lit(1, pl.Int8).alias("label"))
    g = g.join(gt, on=["s1_id", "m_src", "m_id"], how="left").with_columns(pl.col("label").fill_null(0))
    cols = feature_columns(g)
    is_es = (pl.col("s1_id").hash(config.SEED) % 20) == 0
    tr, es = g.filter(~is_es), g.filter(is_es)
    log(f"stage 2: {tr.height:,} rows, {len(cols)} features: {cols}")
    dtr = lgb.Dataset(tr.select(cols).to_numpy().astype("float32"), tr["label"].to_numpy(), feature_name=cols)
    des = lgb.Dataset(es.select(cols).to_numpy().astype("float32"), es["label"].to_numpy(), reference=dtr)
    params = dict(LGB_PARAMS, num_leaves=63)
    model = lgb.train(params, dtr, N_ROUNDS, valid_sets=[des],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    model.save_model(str(MODEL_DIR / f"{MODEL_NAME}_g.lgb"))
    imp = sorted(zip(cols, model.feature_importance("gain")), key=lambda x: -x[1])
    log("stage 2 top features: " + ", ".join(f"{c}={v:.0f}" for c, v in imp[:10]))


def _apply_group(split, scored, m):
    """Replace stage-1 p by the stage-2 probability when a stage-2 model exists."""
    import lightgbm as lgb
    path = MODEL_DIR / f"{MODEL_NAME}_g.lgb"
    if not path.exists() or os.environ.get("BER_NO_GROUP"):
        return scored
    model = lgb.Booster(model_file=str(path))
    extra = _rerank_frame(split, scored) if "xe" in model.feature_name() else None
    g = _group_frame(split, scored, m, extra)
    if "xe" in model.feature_name() and "xe" not in g.columns:
        g = g.with_columns(pl.lit(None, pl.Float32).alias("xe"))
    p = model.predict(g.select(model.feature_name()).to_numpy().astype("float32"), num_threads=LGB_THREADS)
    log(f"stage 2 applied to {g.height:,} pairs")
    return g.select("s1_id", "m_src", "m_id", pl.col("p1"), pl.Series("p", p.astype("float32")))


def stage_predict(run):
    global COUNTRY
    import json

    from ber.assign import apply_rule, tune_rule
    from ber.evaluate import report
    from ber.io import write_pair_table

    d = rdir(run)
    if run == "test":
        split = "test"
        s1 = pl.read_parquet(config.norm_path("test", 1), columns=["id", "country"]).rename({"id": "s1_id"})
        parts = sorted((config.run_dir(run) / "country").glob("*/scored.parquet"))
        log("merging " + ", ".join(p.parent.name for p in parts))
        merged = []
        for p in parts:
            sc = pl.read_parquet(p)
            if (MODEL_DIR / f"{MODEL_NAME}_g.lgb").exists():
                COUNTRY = p.parent.name
                _, _, mc = load_with_regions(run)
                sc = _apply_group(split, sc, mc)
                COUNTRY = None
            merged.append(sc.select("s1_id", "m_src", "m_id", "p"))
        scored = pl.concat(merged)
    else:
        split, s1, m = load_with_regions(run)
        scored = _score(_load_model(), pl.read_parquet(d / "features.parquet"))
        scored.write_parquet(d / "scored.parquet")
        scored = _apply_group(split, scored, m)
        scored.write_parquet(d / f"scored_final_{MODEL_NAME}.parquet")
    tau_path = MODEL_DIR / f"{MODEL_NAME}_tau.json"
    if split == "train":
        gt = universe_gt(s1, m)
        ents = s1.select("s1_id", "country")
        rule, score, lines = tune_rule(scored, gt, ents)
        for line in lines:
            log(line)
        log(f"decision rule {rule} -> {score:.4f}")
        if run == "val":
            tau_path.write_text(json.dumps(rule))
        print(report(apply_rule(scored, rule), gt, ents, by="country"))
    else:
        rule = json.loads(tau_path.read_text())
        tau = rule["tau"]
        pred = apply_rule(scored, rule)
        cand = scored.select("s1_id", "m_src", "m_id")
        assert pred.join(cand, on=["s1_id", "m_src", "m_id"], how="anti").height == 0
        write_pair_table(config.OUTPUT_DIR / "matching_results.tsv", s1["s1_id"], pred, "matched_entity_ids")
        write_pair_table(config.OUTPUT_DIR / "candidate_pairs.tsv", s1["s1_id"], cand, "candidate_entity_ids")
        per = pred.group_by("s1_id").len().join(s1.select("s1_id", "country"), on="s1_id", how="right")
        log(f"rule={rule}; wrote {pred.height:,} matches, {cand.height:,} candidates")
        print(per.group_by("country").agg(pl.col("len").fill_null(0).mean().alias("matches_per_s1"),
                                          pl.col("len").is_null().mean().alias("empty_share")))


STAGES = {"block": stage_block, "features": stage_features, "train": stage_train, "rerank": stage_rerank,
          "group": stage_group, "predict": stage_predict}


def _countries(split):
    return sorted(pl.scan_parquet(config.norm_path(split, 1)).select(pl.col("country").unique())
                  .collect()["country"].to_list())


def main():
    global COUNTRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=["mini", "val", "fit", "fitall", "all", "test"])
    ap.add_argument("--stage", required=True, choices=list(STAGES) + ["all"])
    ap.add_argument("--country", help="test only: process a single country in this process")
    ap.add_argument("--model", default="m1", help="model name under work/models (train/predict)")
    a = ap.parse_args()
    global MODEL_NAME
    COUNTRY, MODEL_NAME = a.country, a.model
    if a.stage == "all":
        fit = _is_train_run(a.run)
        stages = ["block", "features"] + (["train", "rerank", "group"] if fit else []) + (["predict"] if not fit else [])
    else:
        stages = [a.stage]
    if a.run in ("test", "fitall", "all") and COUNTRY is None:
        # one country per subprocess bounds memory; fitall only needs it for blocking
        per_country = [s for s in stages if s in (("block", "features") if a.run == "test" else ("block",))]
        for country in _countries("test" if a.run == "test" else "train"):
            for s in per_country:
                log(f"== {a.run}/{country}: {s} (subprocess)")
                subprocess.run([sys.executable, "-X", "faulthandler", __file__, "--run", a.run,
                                "--stage", s, "--country", country, "--model", MODEL_NAME], check=True)
        stages = [s for s in stages if s not in per_country]
    for s in stages:
        t = time.time()
        log(f"== {a.run}: {s}")
        STAGES[s](a.run)
        log(f"== {s} done in {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
