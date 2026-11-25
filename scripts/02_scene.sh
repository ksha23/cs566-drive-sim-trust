#!/bin/bash
# Stage 2 (one GPU job per scene): fit the 3 lane splats (12k iters). For TRAIN scenes also fit the cycle-reconstruction
# splats (8k iters, fitted only on renders at +/-3 m), then render all pair variants; for val/test scenes only the
# eval pairs ('full' variant).
# usage: $GJ scripts/02_scene.sh <scene> <train|eval>
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
s=$1; role=${2:-train}
for c in clip_0 clip_1 clip_2; do
  d=$DATA/splats/$s/$c
  [ -f $d/splat.pt ] || $PY src/splats/fit_splat.py $s $c $d --iters 12000 > logs/fit_${s}_${c}.log 2>&1
  grep ON-PATH logs/fit_${s}_${c}.log | cut -c1-120
  if [ $role == train ] && [ ! -f $d/splat_cycle.pt ]; then
    $PY src/splats/fit_splat.py $s $c $d --iters 8000 --name splat_cycle --cycle_from $d/splat.pt --cycle_shift 3.0 > logs/cycle_${s}_${c}.log 2>&1
  fi
done
V=full; [ $role == train ] && V=full,identity,underfit,cycle
$PY src/pairs/render_pairs.py $s $DATA/splats $DATA/pairs --variants $V > logs/pairs_$s.log 2>&1; tail -1 logs/pairs_$s.log
