#!/bin/bash
# Stage 7: Difix3D+-style progressive update of each test lane's splat with a fixer (ours / Difix-FT, plus a
# no-fixer control on scene_16), then score the refined splats themselves (no post-fix) at the other lanes' poses.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
OURS=${OURS:-base_s0}   # the val-selected run at the time (before stage 4b's GAN runs existed)
for s in ${TEST:-scene_16 scene_17}; do
  FX="ours:$DATA/fixers/$OURS/best.pt difix_ft:$DATA/difix_ft/best_unet.pt"; [ $s == scene_16 ] && FX="$FX none:-"
  $GJ bash -c "for fx in $FX; do for c in clip_0 clip_1 clip_2; do
      [ -f $DATA/splats/$s/\$c/prog_\${fx%%:*}/log.json ] || $PY src/progressive/progressive.py $s \$c \${fx%%:*} \${fx#*:}; done; done" >> logs/prog_$s.log 2>&1
done
$GJ bash -c "for s in ${TEST:-scene_16 scene_17}; do
    M=\$(for d in $DATA/fixed/prog_*; do [ -d \$d/\$s ] && basename \$d; done | paste -sd,)
    $PY src/eval/evaluate.py $DATA/pairs \$s $DATA/fixed results/eval/eval_\${s}_b.json --methods \$M
    scripts/lib/lanes_scene.sh \$s $DATA/fixed results/lanes/ntl_\${s}_b.json \$(echo \$M | tr ',' ' ')
  done" > logs/prog_eval.log 2>&1
