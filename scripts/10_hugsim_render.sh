#!/bin/bash
# Stage 10 (GPU job): render HUGSIM's released nuScenes scenes at 0/+-1/+-2/+-3 m (+ projected GT vehicle boxes).
# usage: $GJ scripts/10_hugsim_render.sh
source "$(dirname "$0")/lib/env.sh"
source $SMOKE/hugsim_env.sh; cd $SMOKE/HUGSIM
for s in ${SCENES:-scene-0064 scene-0167 scene-0661 scene-0071 scene-0383}; do
  real=""; [ $s == scene-0383 ] && real="--real_root $SMOKE/data/data/scene-0383"
  timeout 1500 pixi run python $REPO/src/hugsim/render_offsets.py --scene $s --n_frames 30 $real
done
