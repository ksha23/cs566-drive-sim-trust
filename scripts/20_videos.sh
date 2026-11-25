#!/bin/bash
# Stage 20: demo videos for the project webpage (results/videos/, index.json). Not used for any metric.
#   sweeps   (one GPU job) continuous lane-change / heading / camera-height sweeps on scene_16 lane 0 + fixers on them
#   make     (CPU) compose all videos and teaser stills from existing outputs: scripts/lib/py.sh src/viz/make_videos.py all
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
V=$DATA/viz; S=scene_16; C=clip_0; SP=$DATA/splats/$S/$C
fix() {  # name: difix, our Difix-ref reimpl., our U-Net, NVIDIA Fixer on the sweep renders
  M=$V/$1/meta.json
  $PY src/fixers/apply.py $V x $V/fixed --meta $M --methods difix --skip_existing
  $PY src/fixers/apply.py $V x $V/fixed --meta $M --methods difix_reimpl --tag difix_reimpl_ref --ckpt $DATA/difix_reimpl/ours_matched/best.pt --skip_existing
  $PY src/fixers/apply.py $V x $V/fixed --meta $M --methods ours --tag ours_hfgan_s0 --ckpt $DATA/fixers/hfgan_s0/best.pt --skip_existing
  $PY src/fixers/apply.py $V x $V/fixed --meta $M --methods nv_fixer --skip_existing
}
case $1 in
  sweeps)
    [ -f $V/lane_$S/meta.json ] || $PY src/viz/sweeps.py $S $C lane lane_$S --alt road=$SP/splat_road.pt --alt prog=$SP/prog_difix_reimpl_r1.pt
    [ -f $V/yaw_$S/meta.json ] || $PY src/viz/sweeps.py $S $C yaw yaw_$S
    [ -f $V/height_$S/meta.json ] || $PY src/viz/sweeps.py $S $C height height_$S
    for n in lane_$S yaw_$S height_$S; do fix $n; done ;;
  make) $PY src/viz/make_videos.py all ;;
esac
