#!/bin/bash
#SBATCH -p interactive
#SBATCH -J cs566-lanesenv
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler16
#SBATCH -c 8
#SBATCH --mem=24G
#SBATCH -t 01:00:00
#SBATCH --requeue
#SBATCH -o logs/lanesenv_%j.log
# CLRNet lane-detector env on a Slurm cluster, as on the main machine (README, setup step 2): a venv over the project
# env with mmcv-lite 1.7.2, addict and yapf 0.32.0, and CLRNet's NMS op built for the cluster's GPUs (sm_86/89/90).
# Expects $CS566D_DATA/lanes/CLRNet (source at 7269e9d) and lanes/weights/culane_dla34.pth copied from the main machine.
set -e
source "${CS566D_ROOT:-$HOME/cs566/D}/repo/scripts/euler/env.sh"
module load gcc/12.2.0 nvidia/cuda/12.1.0
L=$DATA/lanes
[ -x $L/env/bin/python ] || $PY -m venv $L/env
# the project env is itself a venv here, so --system-site-packages would not reach it: a .pth file adds its packages
# after the lane env's own (as the venv-over-env on the main machine does)
$PY -c "import site; print(site.getsitepackages()[0])" > "$($L/env/bin/python -c 'import site; print(site.getsitepackages()[0])')/project_env.pth"
$L/env/bin/python -c "import torch, cv2; print('lane env sees torch', torch.__version__)"
$L/env/bin/python -m pip install -q --no-build-isolation mmcv==1.7.2 addict==2.4.0 yapf==0.32.0
cd $L/CLRNet && TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0" $L/env/bin/python setup.py -q build_ext --inplace
cd $REPO && ls $DATA/pairs/scene_4/real/clip_0/*.png | head -4 > $L/smoke.txt
$L/env/bin/python src/eval/lanes/clrnet_infer.py --list $L/smoke.txt --root $DATA/pairs --out $L/smoke.json
$L/env/bin/python -c "import json; d = json.load(open('$L/smoke.json')); print('CLRNet ok:', len(d), 'images,', sum(len(v.get('lanes', [])) for v in d.values()), 'lanes')"
