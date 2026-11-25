#!/bin/bash
# Stage 6b (one GPU job): evaluate additional fixer runs on the test scenes (default: the +HF+GAN U-Net seeds).
# usage: RUNS="hfgan_s0 hfgan_s1" scripts/06b_eval_extra_unets.sh
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
RUNS=${RUNS:-hfgan_s0 hfgan_s1}; M=""; for r in $RUNS; do M="$M ours_$r"; done
$GJ bash -c "for s in ${TEST:-scene_16 scene_17}; do
    for r in $RUNS; do $PY src/fixers/apply.py $DATA/pairs \$s $DATA/fixed --methods ours --ckpt $DATA/fixers/\$r/best.pt --tag ours_\$r; done
    $PY src/eval/evaluate.py $DATA/pairs \$s $DATA/fixed results/eval/eval_\${s}_gan.json --methods \$(echo $M | tr ' ' ',')
    scripts/lib/lanes_scene.sh \$s $DATA/fixed results/lanes/ntl_\${s}_gan.json $M
  done" > logs/eval_gan.log 2>&1
