# Business Entity Resolution — pipeline

Amazon ML Challenge 2026. Links every Source 1 business to its matching Source 2 / Source 3
records. See `../../PLAN.md` for the full approach.

## Setup

```bash
pip install -r requirements.txt
```

By default the raw data is read from
`../../6ab10eb3b23ba_student_resource/student_resource/dataset/{train,test}/`.
Override paths with the environment variables `BER_DATA_DIR`, `BER_WORK_DIR` (intermediate
files, default `work/`) and `BER_OUTPUT_DIR` (default `../../output/`).

## Run (from this directory)

| Step | Command | Output | Time* |
|---|---|---|---|
| 0. Parquet cache | `python src/prepare_data.py` | `work/parquet/` | ~25 s |
| 1. Validation split | `python src/make_split.py` | `work/split/` | ~30 s |

\*16 GB RAM / 12-core laptop.

Tests: `python -m pytest -q tests`

## Layout

- `src/ber/config.py`: paths and constants
- `src/ber/io.py`: TSV/parquet loading, submission writer (IDs kept as `(src, int)`)
- `src/ber/evaluate.py`: exact macro F0.5 (with the singleton rule), blocking recall/oracle report
- `src/ber/geo.py`: country-agnostic region detection (most frequent address component), region lexicon learned from linked records
- `src/prepare_data.py`, `src/make_split.py`: pipeline entry points

## Validation protocol

Whole regions are held out (with all their S1/S2/S3 records, including distractors), about 10% of S1
per country: US `ny, ky, ia`; India `kerala, tamil nadu`, giving 253k S1 entities. `mini` (`ia` + `kerala`,
58k entities) is used for fast iteration. The fit and val folds have matching singleton rates (5.5–5.6%),
mean matches (3.46) and distractor rates (25–27%).
