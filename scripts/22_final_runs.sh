#!/bin/bash
# Stage 22: runs added for the final report (the plan in Section 4 of the mid-term report). Resumable: each step
# leaves logs/final/<step>.done and is skipped once done. Every GPU step goes through $GJ exactly once (never nested:
# the lock would deadlock).
#   ref       released Difix-ref on scene_3, scene_4, scene_10, scene_11 (fix, metrics, lanes; stage 18 'ref')
#   heading   heading + camera height on the six roads beyond scene_16/17 (render, six fixers, lanes, metrics; stage 15)
#   ff        FaithFusion maps on those six roads; flat- and EIG-weighted progressive updates + metrics on scene_17 and
#             the six (scene_16 was done by stage 14)
#   closed    on-path reference views (src/hugsim/render_refs.py), then our Difix re-implementation in the closed loop
#   collect   for steps run on a cluster (scripts/euler/: heading fixers on CL_HEADING, FaithFusion updates on CL_FF):
#             once their outputs are copied back, the 0-deviation links, CLRNet lanes and metrics here
#             (CL_HEADING= or CL_FF= collects only the other half)
# usage: bash scripts/22_final_runs.sh <step>...      (progress: logs/final/queue.log)
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
L=logs/final; mkdir -p $L
step() {
  n=$1; shift
  [ -f $L/$n.done ] && return 0
  echo "$(date '+%m-%d %H:%M') start $n"
  if "$@" > $L/$n.log 2>&1; then touch $L/$n.done; echo "$(date '+%m-%d %H:%M') done $n"
  else echo "$(date '+%m-%d %H:%M') FAILED $n (see $L/$n.log)"; fi
}
REF_SCENES="scene_3 scene_4 scene_10 scene_11"
NEW_SCENES="scene_14 scene_15 scene_10 scene_3 scene_11 scene_4"
HM="difix difix_ref nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"

run_ref() { for s in $REF_SCENES; do step ref_$s bash scripts/18_more_scenes.sh ref $s; done; }
run_heading() {
  for s in $NEW_SCENES; do
    step heading_render_$s $GJ $PY src/splats/render_deviation.py $s
    for m in $HM; do step heading_fix_${s}_$m $GJ bash scripts/15_heading_height.sh fix $s $m; done
    step heading_lanes_$s $GJ bash scripts/15_heading_height.sh lanes $s
    step heading_eval_$s $GJ bash scripts/15_heading_height.sh eval $s
  done
}
run_ff() {  # FaithFusion (stage 14) on the remaining roads: maps, flat- and EIG-weighted progressive updates, metrics
  FF=scripts/lib/ff_py.sh
  for s in $NEW_SCENES; do step ff_maps_$s $GJ $FF -m progressive.faithfusion.maps $s; done
  for s in scene_17 $NEW_SCENES; do
    for arm in flat eig; do for c in clip_0 clip_1 clip_2; do
      step ff_prog_${s}_${arm}_$c $GJ $FF -m progressive.faithfusion.prog_ff $s $c $arm
    done; done
    ff_eval $s
  done
}
ff_eval() {
  step ff_eval_$1 $GJ bash -c "set -e
    $PY src/eval/evaluate.py $DATA/pairs $1 $DATA/fixed results/faithfusion/eval_${1}_ff.json --methods prog_ff_flat,prog_ff_eig
    scripts/lib/lanes_scene.sh $1 $DATA/fixed results/faithfusion/ntl_${1}_ff.json prog_ff_flat prog_ff_eig"
}
run_collect() {  # after outputs computed elsewhere (scripts/euler/) are copied back: 0-deviation links, lanes, metrics
  for s in $CL_HEADING; do
    for m in $HM; do step heading_links_${s}_$m link_zero $s $m; done
    step heading_lanes_$s $GJ bash scripts/15_heading_height.sh lanes $s
    step heading_eval_$s $GJ bash scripts/15_heading_height.sh eval $s
  done
  for s in $CL_FF; do ff_eval $s; done
}
link_zero() {  # as scripts/15_heading_height.sh: the 0-deviation outputs are the lateral-eval outputs, linked
  for d in $DATA/fixed/$2/$1/render/*; do
    t=$DATA/heading/fixed/$2/$1/render/$(basename $d)/y+0_z+0.0; mkdir -p $(dirname $t); [ -e $t ] || ln -s $d/full $t
  done
}
CL_HEADING=${CL_HEADING-"scene_3 scene_11 scene_4"}; CL_FF=${CL_FF-"scene_10 scene_3 scene_11 scene_4"}  # set empty to skip
run_closed() {  # our Difix re-implementation in the HUGSIM closed loop (stage 12): 25 scenarios x 3 LTF seeds
  step cl_refs $GJ bash -c "source $SMOKE/hugsim_env.sh; cd $SMOKE/HUGSIM && pixi run python $REPO/src/hugsim/render_refs.py \
    --scenes scene-0064,scene-0071,scene-0167,scene-0383,scene-0661"
  SC=$(ls $DATA/hugsim/scenarios/nuscenes/scene-{0383,0071,0661,0064,0167}-*.yaml | tr "\n" " ")
  A="--difix_reimpl $DATA/difix_reimpl/ours_matched/best.pt --refs $DATA/hugsim/refs"
  for k in 1 2 3 4 5 6; do  # each job stops starting runs after BUDGET_S; finished runs are skipped
    step cl_reimpl_$k env SEEDS="0 1 2" $GJ bash scripts/12_hugsim_closed_loop.sh difix_reimpl "$SC" $A
  done
}
for st in "$@"; do run_$st; done
