#!/bin/bash
# Unattended remainder of the run (survives the laptop disconnecting). RESUMABLE:
# every finished step leaves logs/done_<step>; re-running this script skips those.
# Progress: ~/ml-chall/logs/FINAL_STATUS; per-step logs: ~/ml-chall/logs/<step>.log
cd ~/ml-chall && . ./env.sh && cd repo/code/business_entity_resolution
L=~/ml-chall/logs
S=$L/FINAL_STATUS
step() { echo "$(date '+%F %T') $*" >> "$S"; }
fail() { step "FAILED: $1 (see logs/$1.log)"; exit 1; }
PY="python -u -W ignore"
RP="$PY src/run_pipeline.py"
export BER_MAX_NEG=30000000
score() { grep "decision rule" "$1" | tail -1 | sed 's/.*-> //'; }

# run NAME CMD... : skip if done, else run with its own log, mark done or stop
run() {
  local name=$1; shift
  [ -f $L/done_$name ] && return 0
  step "start $name"
  ( "$@" ) > $L/$name.log 2>&1 || fail $name
  touch $L/done_$name
  step "done  $name"
}

step "=== orchestrator (re)started"

# final encoder (every training pair) on GPU 1, in the background
if [ ! -f $L/done_enc_all ]; then
  ( export BER_TAG=all CUDA_VISIBLE_DEVICES=1
    if [ ! -f work/models/name_encoder_all/model.safetensors ]; then
      $PY src/train_encoder.py --all-train > $L/enc_all.log 2>&1 || exit 1
    else
      $PY src/train_encoder.py --encode > $L/enc_all.log 2>&1 || exit 1
    fi
    touch $L/done_enc_all ) &
  step "final encoder running in background (GPU 1)"
fi

# m3: every fit-fold region, narrow reranker band, scored on val
run m3_block    $RP --run fitall --stage block    --model m3
run m3_features $RP --run fitall --stage features --model m3
run m3_train    $RP --run fitall --stage train    --model m3
run m3_rerank   $RP --run fitall --stage rerank   --model m3
run m3_group    $RP --run fitall --stage group    --model m3
run m3_val      $RP --run val    --stage predict  --model m3
step "m3 val F0.5: $(score $L/m3_val.log)"

# m3w: reranker + stage 2 retrained with the wide band (stage 1 shared with m3)
cp -n work/models/m3.lgb work/models/m3w.lgb
run m3w_rerank env BER_BAND=wide $RP --run fitall --stage rerank  --model m3w
run m3w_group  env BER_BAND=wide $RP --run fitall --stage group   --model m3w
run m3w_val    env BER_BAND=wide $RP --run val    --stage predict --model m3w
if [ ! -f $L/band ]; then
  N=$(score $L/m3_val.log); W=$(score $L/m3w_val.log)
  if python -c "import sys; sys.exit(0 if float('$W') > float('$N') else 1)"; then echo wide > $L/band; else echo narrow > $L/band; fi
  step "val F0.5 wide $W vs narrow $N -> band=$(cat $L/band)"
fi
BAND=$(cat $L/band); RULE=$([ "$BAND" = wide ] && echo m3w || echo m3)

# final models on every training record with the final encoder
until [ -f $L/done_enc_all ] || ! pgrep -f "[t]rain_encoder.py" > /dev/null; do sleep 30; done
[ -f $L/done_enc_all ] || fail enc_all
export BER_TAG=all BER_BAND=$BAND
run final_block    $RP --run all --stage block    --model final
run final_features $RP --run all --stage features --model final
run final_train    $RP --run all --stage train    --model final
run final_rerank   $RP --run all --stage rerank   --model final
run final_group    $RP --run all --stage group    --model final
cp work/models/${RULE}_tau.json work/models/final_tau.json
step "final decision rule (from $RULE): $(cat work/models/final_tau.json)"

# test predictions + official validator
run test_block    $RP --run test --stage block    --model final
run test_features $RP --run test --stage features --model final
run test_predict  $RP --run test --stage predict  --model final
step "$(grep 'wrote' $L/test_predict.log | tail -1)"
R=../../6ab10eb3b23ba_student_resource/student_resource
run validator python $R/utils/validate_submission.py --matching ../../output/matching_results.tsv \
  --candidate ../../output/candidate_pairs.tsv --test-dir $R/dataset/test --check-ids
step "validator: $(tail -1 $L/validator.log)"
step "=== ALL DONE"
