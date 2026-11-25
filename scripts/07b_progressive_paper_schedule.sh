#!/bin/bash
# Stage 7b (one GPU job): the progressive update with a schedule closer to Difix3D+ Alg. 1 -- pose steps of 1.5 m
# (1.5, 3, 4.5, 6 m) with 1.5k refinement iterations after each step -- using Difix-FT on scene_16, method name
# prog_difix_ft_paper; then its metrics.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"; s=scene_16
$GJ bash -c "for c in clip_0 clip_1 clip_2; do
    [ -f $DATA/splats/$s/\$c/prog_difix_ft_paper/log.json ] || $PY src/progressive/progressive.py $s \$c difix_ft $DATA/difix_ft/best_unet.pt --offsets 1.5,3,4.5,6 --iters 1500 --tag prog_difix_ft_paper; done
  $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/eval/eval_${s}_c.json --methods prog_difix_ft_paper
  scripts/lib/lanes_scene.sh $s $DATA/fixed results/lanes/ntl_${s}_c.json prog_difix_ft_paper" > logs/prog_paper.log 2>&1
