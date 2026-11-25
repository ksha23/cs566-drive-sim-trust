#!/bin/bash
# Stage 6: every fixer on the held-out TEST scenes (one GPU job per scene): zero-shot Difix / Difix-ref (near and
# >= 2 s reference), fine-tuned Difix, our U-Nets; then image/hallucination/YOLO metrics and CLRNet lane agreement.
# Extra fixers (e.g. nv_fixer, difix_reimpl) are evaluated with the same two commands at the end, method names only.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
mkdir -p results/eval results/lanes
RUNS=$(ls $DATA/fixers | tr "\n" " "); OM=""; for r in $RUNS; do OM="$OM ours_$r"; done
for s in ${TEST:-scene_16 scene_17}; do
  $GJ bash -c "
    $PY src/fixers/apply.py $DATA/pairs $s $DATA/fixed --methods difix,difix_ref,difix_ref_far
    $PY src/fixers/apply.py $DATA/pairs $s $DATA/fixed --methods difix_ft --ckpt $DATA/difix_ft/best_unet.pt
    for r in $RUNS; do $PY src/fixers/apply.py $DATA/pairs $s $DATA/fixed --methods ours --ckpt $DATA/fixers/\$r/best.pt --tag ours_\$r; done
    $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/eval/eval_${s}_a.json --methods raw,difix,difix_ref,difix_ref_far,difix_ft$(echo $OM | sed 's/ /,/g; s/^/,/')
    scripts/lib/lanes_scene.sh $s $DATA/fixed results/lanes/ntl_${s}_a.json raw difix difix_ref difix_ref_far difix_ft $OM
  " > logs/eval_fixers_$s.log 2>&1
done
