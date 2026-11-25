#!/bin/bash
# Stage 23: regenerate every summary, table, figure and the headline numbers from the per-image results (CPU only,
# about 30 minutes). Run after stages 18 and 22 (and their 'collect' step when part ran on a cluster).
#   results/eval_more      pooled lateral metrics over the eight roads (more_scenes_summary.py)
#   results/hugsim         closed-loop scores and route deviation
#   results/heading        heading / camera-height tables and figure
#   results/faithfusion    information-gain AUROC over the eight roads; EIG- vs flat-weighted update per road
#   results/final          report tables and figures, robustness checks, key_numbers.json (talk and webpage)
# usage: bash scripts/23_regenerate.sh
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
export CUDA_VISIBLE_DEVICES=
set -e
ROADS=scene_3,scene_4,scene_10,scene_11,scene_14,scene_15,scene_16,scene_17
t() { echo "$(date "+%H:%M") $*" >&2; "$@"; }
t $PY src/eval/more_scenes_summary.py > /dev/null
t $PY src/hugsim/summarize_closed_loop.py --out results/hugsim > /dev/null
t $PY src/hugsim/route_deviation.py --out results/hugsim > /dev/null
t bash scripts/15_heading_height.sh summary
t scripts/lib/ff_py.sh -m progressive.faithfusion.halluc_auroc --scenes $ROADS \
    --methods difix,difix_ref,nv_fixer,difix_ft,difix_reimpl_ref --out results/faithfusion > logs/ff_auroc.log
t $PY src/progressive/faithfusion/summary.py > /dev/null
t $PY src/eval/final_report.py > /dev/null
t $PY src/eval/midterm_checks.py > /dev/null
t $PY src/eval/key_numbers.py > /dev/null
echo "$(date '+%H:%M') done: results/final/key_numbers.json"
