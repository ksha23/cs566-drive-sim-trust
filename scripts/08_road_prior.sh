#!/bin/bash
# Stage 8 (one GPU job; stretch): ViSE-style road-surface prior. Refit scene_16's lanes with --road_prior, render at the other lanes'
# poses (no fixer) as pseudo-method road_prior, and score it (image metrics + CLRNet lanes). One GPU job.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"; s=scene_16
for c in clip_0 clip_1 clip_2; do
  d=$DATA/splats/$s/$c
  [ -f $d/splat_road.pt ] || $PY src/splats/fit_splat.py $s $c $d --iters 12000 --name splat_road --road_prior 0.1 > logs/road_${s}_$c.log 2>&1
  grep -E "road surface|ON-PATH" logs/road_${s}_$c.log | cut -c1-150
  $PY src/splats/render_method.py $s $c $d/splat_road.pt road_prior
done
$PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/eval/eval_${s}_road.json --methods road_prior
scripts/lib/lanes_scene.sh $s $DATA/fixed results/lanes/ntl_${s}_road.json road_prior
