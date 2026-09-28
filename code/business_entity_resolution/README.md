# Business Entity Resolution: pipeline (team momos)

Amazon ML Challenge 2026. For every Source 1 business, finds its matching Source 2 / Source 3
records. It writes `output/matching_results.tsv` (the leaderboard file) and
`output/candidate_pairs.tsv` (the exact pairs the matching model scores). The methodology is in
`Documentation_template.md` at the package root.

**Results:** held-out validation macro F0.5 is **0.993** (US 0.993, India 0.993; 285k S1 entities in
held-out regions). The public leaderboard score is **0.9827**, with France using the smaller e5 reranker (below). The
candidate set is ~5.7 pairs per S1 entity on validation (7.4 on test) after learned pruning, where the
average entity has 3.5 true matches.

## Setup

Python 3.9. The pipeline is tested on Ubuntu 24.04 (64 cores, 125 GB RAM, RTX 5090) and on Windows 11.

```bash
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128   # CUDA build (RTX 50xx needs cu128)
pip install -r requirements.txt
```

**Hardware for the full run:**
- a CUDA GPU: about 12 GB of GPU memory is used, for top-k search, encoder training and reranking
- 64 GB RAM recommended; the observed peak is about 30 GB, because blocking runs one country per process

The only downloads are the pretrained weights of `intfloat/multilingual-e5-small` (MIT licence, the
name encoder) and `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M parameters, the cross-encoder). They come
from the Hugging Face hub on first use and are cached under `HF_HOME`. No other external data or
service is used.

**Paths:**
- The raw data is read from `../../6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/` by default.
- Override the locations with `BER_DATA_DIR` (raw TSVs), `BER_WORK_DIR` (intermediate files, default `work/`)
  and `BER_OUTPUT_DIR` (default `../../output/`).

## Reproduce the submission

Run this from this directory. It regenerates both output files from the raw data and then runs the
official validator. The validator comes from the challenge's `student_resource` kit and isn't part of this
package. Point `BER_VALIDATOR` at `validate_submission.py`, or the check is skipped; the output files are
written either way.

```bash
bash src/reproduce.sh              # ~6 h on the machine above
bash src/reproduce.sh --validate   # also the held-out validation experiment (+~2.5 h)
```

| Step | Command (inside `reproduce.sh`) | Output |
|---|---|---|
| 0. Parquet cache | `python src/prepare_data.py` | `work/parquet/` |
| 1. Validation split | `python src/make_split.py` | `work/split/` |
| 2. Normalise names/addresses | `python src/normalize_data.py` | `work/norm/` |
| 3. Name encoder | `BER_TAG=all python src/train_encoder.py --all-train` | `work/models/name_encoder_all/`, `work/emb_all/` |
| 4. Train all models | `BER_TAG=all python src/run_pipeline.py --run all --stage all --model final` | `work/models/final*.lgb`, `work/models/reranker_all/` |
| 5. Predict test | `BER_TAG=all python src/run_pipeline.py --run test --stage all --model final` | `output/*.tsv` |

**Reproducibility:** the submitted `output/` files were produced by this code: `run_pipeline.py --run test`
with the models described here. A full rerun gives the same candidate set byte for byte. On different GPUs it can
differ in a handful of borderline matches, because of floating-point differences in reranker scoring (batch
composition changes with the number of GPUs). Moving France's reranker from 1 to 2 GPUs changed 1 of
1,732,544 rows.

`run_pipeline.py` runs the stages `block → features → train → rerank → group` on a training
universe, and `block → features → predict` on `val`/`test`. Any single stage can be re-run with
`--stage`.

**Universes:**
- `all`: every training record (used for the final models)
- `fitall`: the fit-fold regions
- `val`: the held-out regions
- `mini`: a small val subset
- `test`

Large universes are blocked one country per subprocess to bound memory.

The final settings are the code defaults:
- cross-encoder `bge-reranker-v2-m3` scoring every pruned pair (`BER_RERANK_MODEL`, `BER_RERANK_ALL`)
- **France** uses the `final_e5` models instead: the e5-small reranker on the uncertain band, with its own stage 2 and rule
  (`DEFAULT_COUNTRY_MODELS` / `BER_COUNTRY_MODELS="France=final_e5:e5-small"`). France has no labels, so this was chosen on
  the public leaderboard: 0.98268 vs 0.98232 with bge for France, with identical US/India predictions.
- learned pruning at stage-1 p ≥ 1e-4 (`BER_PRUNE_P1`)
- expected-F0.5 decision rule, extra = 0.5 (`FINAL_RULE` in `run_pipeline.py`, selected on validation)

`BER_JOBS` caps the CPU workers. LightGBM uses at most 32 threads, because more was much slower on
64 cores.

## Pipeline

1. **Normalisation** (`ber/normalize.py`):
   - ASCII fold and transliteration of native scripts (anyascii)
   - junk wrappers, leet/OCR fixes and legal-form extraction for names
   - canonical street types and ordinals, and `number|token` address keys
2. **Regions** (`ber/geo.py`, `ber/blocking.py`):
   - S1 region = its most frequent address component
   - S2/S3 regions come from a lexicon learned on *pseudo-links* (exact address-key matches with near-identical names). No labels are used, so it also works for France (région ↔ département).
3. **Blocking** (`ber/blocking.py`, `ber/topk.py`, `ber/embed.py`). Every pass runs within (country, region) and gives top-k per S2/S3 record, on the GPU:
   - B1: address keys
   - B2: name char-3-gram TF-IDF
   - B6: address word TF-IDF
   - B5: fine-tuned multilingual name-encoder embeddings

   Records without a region search the whole country.
4. **Stage 1** (`ber/features.py` + LightGBM):
   - ~45 pairwise features: rapidfuzz name/address similarities, number overlap, legal-form agreement, TF-IDF and encoder cosines
   - competition features: gap and rank against the other candidates of the same record
5. **Learned pruning:** pairs with stage-1 p < 1e-4 are dropped. The remaining pairs are `candidate_pairs.tsv`, and nothing after this point sees any other pair.
6. **Cross-encoder reranker** (`ber/rerank.py`): `BAAI/bge-reranker-v2-m3` (Apache-2.0, 568M), a pretrained multilingual cross-encoder, reads `name | address` of both records. It is fine-tuned on the uncertain band plus some easy pairs of one half of the entities, then scores every pruned pair. Stage 2 learns how far to trust it on the other half. Scoring runs one model copy per GPU.
7. **Stage 2** (`ber/group.py` + LightGBM):
   - out-of-fold stage-1 scores and rival-S1 gaps
   - the reranker logit
   - similarity of each record to the entity's *other* confident matches
8. **Decision** (`ber/assign.py`):
   - each S2/S3 record goes to at most its single best S1
   - per entity, choose the prefix of candidates that maximises expected F0.5; this handles singletons too

## Layout

- `src/ber/`: library modules
  - `config`, `io`, `evaluate` (exact macro F0.5), `geo`, `normalize`, `topk`, `blocking`, `embed`, `features`, `rerank`, `group`, `assign`
- `src/prepare_data.py`, `src/make_split.py`, `src/normalize_data.py`, `src/train_encoder.py`, `src/run_pipeline.py`, `src/reproduce.sh`: entry points

## Validation protocol

Whole regions are held out, with all their S1/S2/S3 records and distractors:
- US: `ny, nd, az`
- India: `kerala, tamil nadu`

That's 285k S1 entities, about 12% per country. The fit and val folds have matching singleton rates,
mean match counts and distractor rates.

Holding out Tamil Nadu and Kerala also withholds most Tamil- and Malayalam-script names from
training, so validation understates India. The final models train on every region.
