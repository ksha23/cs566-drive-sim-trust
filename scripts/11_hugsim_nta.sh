#!/bin/bash
# Stage 11 (GPU job): NTA-IoU (ReconDreamer's metric, re-implemented) on the HUGSIM offset renders.
# Extra args go to nta_iou.py, e.g. --methods raw,difix,difix_ref,difix_ft,ours --unet ours=<ckpt> --difix_ft <unet.pt>
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
$PY src/hugsim/nta_iou.py --scenes ${SCENES:-scene-0064,scene-0167,scene-0661,scene-0071} --out results/hugsim "$@"
$PY src/hugsim/plot_nta.py --res results/hugsim
