#!/bin/bash
# Run python in the project env (torch 2.4 + gsplat 1.4 + diffusers + Difix deps; see envs/difix_env.txt),
# with src/ on PYTHONPATH so modules import as packages (splats.*, fixers.*, eval.*, ...).
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export PATH=$SMOKE/envs/difix/bin:$PATH
export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
exec python "$@"
