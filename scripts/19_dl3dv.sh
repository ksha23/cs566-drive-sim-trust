#!/bin/bash
# Stage 19: DL3DV validation of our Difix reimplementation [1] against the official trainer and the released weights
# (docs/difix_reimpl_validation.md). DL3DV is gated; the account is authorized for DL3DV/DL3DV-10K-Sample only.
#   download          11 sample scenes (COLMAP + images_4)                                      (CPU)
#   fit 30|60         sparse-reconstruction + underfitting pairs at stride S for every scene      (one GPU job per S)
#   json              pair JSONs + split                                                         (CPU)
#   ours448 | official448 | oursref    matched training runs (one GPU job each, <= 60 min, resumable)
#   eval              every model on the held-out TEST scenes + the VAL scene, metrics            (GPU job(s))
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
DL=$DATA/dl3dv; SRC=$DL/DL3DV-10K-Sample; DX=$DATA/difix_reimpl; H=src/fixers/difix_reimpl
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  download) $PY src/dl3dv/download.py ;;
  fit) for s in $(ls -d $SRC/*/ | sort); do $PY src/dl3dv/fit_pairs.py $s --stride $2 2>&1 | grep -E "frames|DONE|done already|Error"; done ;;
  json) $PY src/dl3dv/build_json.py ;;
  ours448)  # our trainer, matched to the official one: no reference view, 448x768, 1801 steps, same data/settings
    $H/run_ours.sh $DL/ours448 1801 --json $DL/dl3dv_pairs.json --minutes 57 --val_every 300 --size 448x768 --no_ref >> $DL/ours448.log 2>&1 ;;
  official448)  # official trainer (patched copy, see the validation doc), same data/settings. Its max_train_steps does not
    # stop the loop, so a watcher stops it once the step-1801 checkpoint is written.
    setsid timeout 3500 $H/run_official.sh $DL/official448 1801 --checkpointing_steps 600 --data_height 448 --data_width 768 --no_mv \
      --dataset_path=$DL/dl3dv_pairs_noref.json >> $DL/official448.log 2>&1 &
    P=$!; until [ -f $DL/official448/checkpoints/model_1801.pkl ] || ! kill -0 $P 2>/dev/null; do sleep 20; done
    sleep 60; kill -- -$P 2>/dev/null; wait $P; echo "official448 stopped; checkpoints:" $(ls $DL/official448/checkpoints) ;;
  oursref)  # our trainer with the reference view at 576x1024 (the Difix-ref setting the official trainer cannot fit in 24 GB)
    $H/run_ours.sh $DL/oursref 1801 --json $DL/dl3dv_pairs.json --minutes 57 --val_every 300 >> $DL/oursref.log 2>&1 ;;
  eval) bash src/dl3dv/eval_job.sh ${2:-all} ;;
esac
