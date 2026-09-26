#!/bin/bash
# Unattended remainder of the run (survives the laptop disconnecting).
# Progress: ~/ml-chall/logs/FINAL_STATUS (one line per step); per-step logs in ~/ml-chall/logs/.
cd ~/ml-chall && . ./env.sh && cd repo/code/business_entity_resolution
L=~/ml-chall/logs
S=$L/FINAL_STATUS
step() { echo "$(date '+%F %T') $*" >> "$S"; }
fail() { step "FAILED: $*"; exit 1; }
PY="python -u -W ignore"
export BER_MAX_NEG=30000000
score() { grep "decision rule" "$1" | tail -1 | sed 's/.*-> //'; }

step "started"

# 1. final name encoder on every training pair, on GPU 1, in parallel with the m3 steps
#    (reuse a run that is already going or finished)
if pgrep -f "[t]rain_encoder.py --all-train" > /dev/null || grep -q "exit=0" $L/enc_all.log 2>/dev/null; then
  step "final encoder already running/finished; waiting for it later"
else
  ( export BER_TAG=all CUDA_VISIBLE_DEVICES=1
    $PY src/train_encoder.py --all-train > $L/enc_all.log 2>&1; echo "exit=$?" >> $L/enc_all.log ) &
  step "final encoder training started (GPU 1)"
fi

# 2. m3: every fit-fold region, narrow reranker band, scored on val
rm -rf work/runs/fitall
step "m3 started"
( $PY src/run_pipeline.py --run fitall --stage all --model m3 &&
  $PY src/run_pipeline.py --run val --stage predict --model m3 ) > $L/m3.log 2>&1 || fail "m3 (see logs/m3.log)"
step "m3 done: val F0.5 $(score $L/m3.log)"

# 3. same models with the wide reranker band (reranker + stage 2 retrained), val scored
cp work/models/m3.lgb work/models/m3w.lgb
step "m3w (wide band) started"
( export BER_BAND=wide
  $PY src/run_pipeline.py --run fitall --stage rerank --model m3w &&
  $PY src/run_pipeline.py --run fitall --stage group --model m3w &&
  $PY src/run_pipeline.py --run val --stage predict --model m3w ) > $L/m3w.log 2>&1 || fail "m3w (see logs/m3w.log)"
N=$(score $L/m3.log); W=$(score $L/m3w.log)
if python -c "import sys; sys.exit(0 if float('$W') > float('$N') else 1)"; then BAND=wide; RULE=m3w; else BAND=narrow; RULE=m3; fi
step "m3w done: val F0.5 $W (narrow $N) -> using band=$BAND"

# 4. final models on every training record (val regions included), final encoder
until grep -q "exit=" $L/enc_all.log 2>/dev/null; do sleep 30; done
grep -q "exit=0" $L/enc_all.log || fail "final encoder (see logs/enc_all.log)"
step "final encoder done"
export BER_TAG=all BER_BAND=$BAND
$PY src/run_pipeline.py --run all --stage all --model final > $L/final_train.log 2>&1 || fail "final training (see logs/final_train.log)"
cp work/models/${RULE}_tau.json work/models/final_tau.json
step "final models trained; decision rule from $RULE: $(cat work/models/final_tau.json)"

# 5. test predictions
$PY src/run_pipeline.py --run test --stage all --model final > $L/final_test.log 2>&1 || fail "test run (see logs/final_test.log)"
step "test predictions written: $(grep 'wrote' $L/final_test.log | tail -1)"

# 6. official validator
R=../../6ab10eb3b23ba_student_resource/student_resource
python $R/utils/validate_submission.py --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv --test-dir $R/dataset/test --check-ids > $L/validator.log 2>&1
step "validator: $(tail -1 $L/validator.log)"
step "ALL DONE"
