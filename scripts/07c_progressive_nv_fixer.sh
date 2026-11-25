#!/bin/bash
# Stage 7c (one GPU job per scene): the progressive update with NVIDIA Fixer [4] as the restorer (frozen; served from
# its own env), same schedule as stage 7, then its metrics (method prog_nv_fixer).
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
for s in ${TEST:-scene_16 scene_17}; do
  $GJ bash -c "for c in clip_0 clip_1 clip_2; do
      [ -f $DATA/splats/$s/\$c/prog_nv_fixer/log.json ] || $PY src/progressive/progressive.py $s \$c nv_fixer -; done
    $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/eval/eval_${s}_prognv.json --methods prog_nv_fixer
    scripts/lib/lanes_scene.sh $s $DATA/fixed results/lanes/ntl_${s}_prognv.json prog_nv_fixer" >> logs/prog_nv_$s.log 2>&1
done
