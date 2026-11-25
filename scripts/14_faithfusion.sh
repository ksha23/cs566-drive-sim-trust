#!/bin/bash
# Stage 14: FaithFusion [2] arm (src/progressive/faithfusion, docs/faithfusion.md).
#   setup  clone FaithFusion, build its modified rasterizer into a venv over the project env (CPU only)
#   maps   EIG + reprojection-consistency + coverage maps for the test scenes' cross-lane eval renders (1 GPU job)
#   auroc  do those maps predict hallucinated pixels of each fixer? (CPU)
#   prog   EIG-weighted vs. flat-weighted progressive update with Difix-ref as restorer, scene_16 (1 GPU job per arm)
#   eval   score the refined splats like every other method (1 GPU job)
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
FF=scripts/lib/ff_py.sh; step=${1:-all}; S=${SCENE:-scene_16}
if [[ $step == setup ]]; then
  mkdir -p $DATA/faithfusion && cd $DATA/faithfusion
  [ -d FaithFusion ] || git clone --depth 1 https://github.com/wangyuanbiubiubiu/FaithFusion.git   # tested at dfd8c5d
  [ -d env ] || $SMOKE/envs/difix/bin/python -m venv --system-site-packages env
  CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=8.6 \
    env/bin/pip install --no-build-isolation --no-deps FaithFusion/diff
fi
if [[ $step == maps || $step == all ]]; then
  $GJ bash -c "$FF -m progressive.faithfusion.maps scene_16 && $FF -m progressive.faithfusion.maps scene_17" > logs/ff_maps.log 2>&1
fi
if [[ $step == auroc || $step == all ]]; then
  CUDA_VISIBLE_DEVICES= $FF -m progressive.faithfusion.halluc_auroc --out results/faithfusion > logs/ff_auroc.log 2>&1
fi
if [[ $step == prog || $step == all ]]; then
  for arm in flat eig; do
    $GJ bash -c "for c in clip_0 clip_1 clip_2; do
        [ -f $DATA/splats/$S/\$c/prog_ff_$arm/log.json ] || $FF -m progressive.faithfusion.prog_ff $S \$c $arm; done" >> logs/ff_prog_$arm.log 2>&1
  done
fi
if [[ $step == eval || $step == all ]]; then
  $GJ bash -c "$PY src/eval/evaluate.py $DATA/pairs $S $DATA/fixed results/faithfusion/eval_${S}_ff.json --methods prog_ff_flat,prog_ff_eig &&
      scripts/lib/lanes_scene.sh $S $DATA/fixed results/faithfusion/ntl_${S}_ff.json prog_ff_flat prog_ff_eig" > logs/ff_eval.log 2>&1
fi
