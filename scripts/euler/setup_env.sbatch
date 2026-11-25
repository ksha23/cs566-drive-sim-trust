#!/bin/bash
#SBATCH -p interactive
#SBATCH -J cs566-env
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/env_%j.log
# Optional: the environments for running part of stage 22 on a Slurm cluster (here UW-Madison Euler), CPU only.
# Project env (Python 3.10 via uv, envs/difix_env.txt), our Difix re-implementation venv, FaithFusion's rasterizer,
# Difix3D's code, and the public Hugging Face weights. Paths: R below; the scripts read CS566D_DATA / CS566D_SMOKE.
set -euo pipefail
R=${CS566D_ROOT:-$HOME/cs566/D}; S=$R/smoke; D=$R/data; REPO=$R/repo
module load gcc/12.2.0 nvidia/cuda/12.1.0
export CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
export HF_HOME=$S/hf UV_CACHE_DIR=$S/uv_cache PIP_CACHE_DIR=$S/pip_cache UV_PYTHON_INSTALL_DIR=$S/uv_python
mkdir -p $S/envs $D/difix_reimpl $D/faithfusion
python3 -m pip install --user -q uv; U=$HOME/.local/bin/uv
# 1. project env
[ -x $S/envs/difix/bin/python ] || { $U python install 3.10; $U venv --seed -p 3.10 $S/envs/difix; }
grep -E '==' $REPO/envs/difix_env.txt > $S/difix_env_pins.txt
$U pip install -p $S/envs/difix/bin/python -r $S/difix_env_pins.txt --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://docs.gsplat.studio/whl/pt24cu121
# 2. released Difix code (model definition, pipeline), pinned as in README
[ -d $S/Difix3D ] || git clone -q https://github.com/nv-tlabs/Difix3D $S/Difix3D
git -C $S/Difix3D checkout -q c76edc5
# venvs layered over the project env: here the project env is itself a venv, so --system-site-packages would expose
# the bare interpreter; a .pth file adds the project env's packages instead (same effect as on a conda base)
SP=$($S/envs/difix/bin/python -c "import site; print(site.getsitepackages()[0])")
over() { [ -x $1/bin/python ] || $S/envs/difix/bin/python -m venv $1
         echo "$SP" > "$($1/bin/python -c "import site; print(site.getsitepackages()[0])")/_project_env.pth"; }
# 3. our Difix re-implementation: venv over the project env
over $D/difix_reimpl/venv
$D/difix_reimpl/venv/bin/pip install -q -r $REPO/envs/difix_reimpl_venv_local.txt
# 4. FaithFusion's modified rasterizer (tested at dfd8c5d), venv over the project env
[ -d $D/faithfusion/FaithFusion ] || git clone -q https://github.com/wangyuanbiubiubiu/FaithFusion.git $D/faithfusion/FaithFusion
git -C $D/faithfusion/FaithFusion checkout -q dfd8c5d
over $D/faithfusion/env
TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0" MAX_JOBS=8 \
  $D/faithfusion/env/bin/pip install -q --no-build-isolation --no-deps $D/faithfusion/FaithFusion/diff
# 5. public weights
$S/envs/difix/bin/python -c "
from huggingface_hub import snapshot_download as s
for r in ('nvidia/difix', 'nvidia/difix_ref', 'stabilityai/sd-turbo'): print(s(r))"
$S/envs/difix/bin/python -c "import torch, gsplat, diffusers; print('env ok', torch.__version__, gsplat.__version__, diffusers.__version__)"
$D/faithfusion/env/bin/python -c "import modified_diff_gaussian_rasterization; print('faithfusion rasterizer ok')"
echo SETUP_DONE
