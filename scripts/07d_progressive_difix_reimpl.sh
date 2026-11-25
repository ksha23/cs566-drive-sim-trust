#!/bin/bash
# Stage 7d (one GPU job per scene): the progressive update with OUR Difix reimplementation [1] (reference view = the
# lane's real frame at the unshifted pose), same schedule as stage 7, then its metrics (method prog_difix_reimpl).
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
CK=${CK:-$DATA/difix_reimpl/ours_matched/best.pt}
for s in ${TEST:-scene_16 scene_17}; do
  $GJ bash -c "for c in clip_0 clip_1 clip_2; do
      [ -f $DATA/splats/$s/\$c/prog_difix_reimpl/log.json ] || $PY src/progressive/progressive.py $s \$c difix_reimpl $CK; done
    $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/eval/eval_${s}_progdx.json --methods prog_difix_reimpl
    scripts/lib/lanes_scene.sh $s $DATA/fixed results/lanes/ntl_${s}_progdx.json prog_difix_reimpl" >> logs/prog_dx_$s.log 2>&1
done
