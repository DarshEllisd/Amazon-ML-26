#!/bin/bash
# End-to-end reproduction of output/matching_results.tsv and output/candidate_pairs.tsv
# from the raw challenge data. Run from code/business_entity_resolution/:
#
#     bash src/reproduce.sh            # final submission (models trained on all training data)
#     bash src/reproduce.sh --validate # additionally: the held-out validation experiment
#
# Data location: BER_DATA_DIR (default ../../6ab10eb3b23ba_student_resource/student_resource/dataset),
# outputs: BER_OUTPUT_DIR (default ../../output), intermediates: BER_WORK_DIR (default work/).
# Needs a CUDA GPU (tested: RTX 5090 32 GB; ~12 GB used) and ~64 GB RAM (peak ~30 GB observed).
set -euo pipefail
PY="python -u -W ignore"

# 0-2. parquet cache, geographic validation split, name/address normalisation   (~5 min)
$PY src/prepare_data.py
$PY src/make_split.py
$PY src/normalize_data.py

if [ "${1:-}" = "--validate" ]; then
  # Validation experiment: every model trained on the fit-fold regions only, scored on the
  # held-out regions (US ny/nd/az, India kerala/tamil nadu). Prints macro F0.5 per country and
  # tunes the decision rule (written to work/models/val_tau.json).                  (~2.5 h)
  $PY src/train_encoder.py
  $PY src/run_pipeline.py --run fitall --stage all --model val
  $PY src/run_pipeline.py --run val    --stage all --model val
fi

# 3. multilingual name encoder fine-tuned on every training pair, all names embedded (~25 min)
export BER_TAG=all
$PY src/train_encoder.py --all-train

# 4. final models on every training record: blocking, features, stage-1 LightGBM
#    (+ out-of-fold scores), learned pruning, cross-encoder reranker, stage-2 group model (~2.5 h)
$PY src/run_pipeline.py --run all --stage all --model final

# 5. test: blocking + features + stage 1 per country, pruning, reranker, stage 2, decision
#    rule -> output/matching_results.tsv and output/candidate_pairs.tsv                  (~1 h)
$PY src/run_pipeline.py --run test --stage all --model final

# 6. official validator
R=../../6ab10eb3b23ba_student_resource/student_resource
python "$R/utils/validate_submission.py" --matching "${BER_OUTPUT_DIR:-../../output}/matching_results.tsv" \
  --candidate "${BER_OUTPUT_DIR:-../../output}/candidate_pairs.tsv" --test-dir "${BER_DATA_DIR:-$R/dataset}/test" --check-ids
