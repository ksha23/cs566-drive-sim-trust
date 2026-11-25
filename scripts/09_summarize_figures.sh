#!/bin/bash
# Stage 9: summary tables with bootstrap CIs (CPU), trust-vs-offset chart + panels (CPU), fly-through videos (GPU job).
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
OURS=ours_$(awk '/^selected/{print $2}' results/training/selected_ours.txt)
E=$(ls results/eval/eval_scene_1[67]_*.json results/faithfusion/eval_scene_16_ff.json results/difix_reimpl/eval_scene_1[67]_*.json | grep -v summary | tr "\n" " ")
L=$(ls results/lanes/ntl_scene_1[67]_*.json results/faithfusion/ntl_scene_16_ff.json results/difix_reimpl/ntl_scene_1[67]_*.json | tr "\n" " ")
CUDA_VISIBLE_DEVICES= $PY src/eval/summarize.py results/eval/summary_test $E $L > /dev/null
CUDA_VISIBLE_DEVICES= $PY src/eval/summarize.py results/eval/summary_scene_16 $(ls results/eval/eval_scene_16_*.json results/faithfusion/eval_scene_16_ff.json results/difix_reimpl/eval_scene_16_*.json | grep -v summary) $(ls results/lanes/ntl_scene_16_*.json results/faithfusion/ntl_scene_16_ff.json results/difix_reimpl/ntl_scene_16_*.json) > /dev/null
CUDA_VISIBLE_DEVICES= $PY src/eval/summarize.py results/eval/summary_scene_17 $(ls results/eval/eval_scene_17_*.json results/difix_reimpl/eval_scene_17_*.json | grep -v summary) $(ls results/lanes/ntl_scene_17_*.json results/difix_reimpl/ntl_scene_17_*.json) > /dev/null
CUDA_VISIBLE_DEVICES= $PY src/eval/results_tables.py > results/eval/tables.md
CUDA_VISIBLE_DEVICES= $PY src/eval/figures.py results/eval/summary_test.csv $(echo $E | tr " " ",") $DATA/pairs $DATA/fixed results/figures --ours $OURS
if [ "$1" == video ]; then
  D=$DATA/splats/scene_16/clip_0; R=$(awk '/^selected/{print $2}' results/training/selected_ours.txt)
  $GJ bash -c "$PY src/eval/flythrough.py $D/splat.pt scene_16 clip_0 $DATA/fixers/$R/best.pt results/figures/flythrough_raw_vs_ours.mp4 6.3 'our fixer (small, from scratch)';
    $PY src/eval/flythrough.py $D/splat.pt scene_16 clip_0 splat:$D/prog_difix_ft_r1.pt results/figures/flythrough_raw_vs_progressive_difixft.mp4 6.3 'progressive update (Difix-FT), no post-fix'" > logs/video.log 2>&1
fi
