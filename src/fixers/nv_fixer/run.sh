#!/bin/bash
# Run a python script in the NVIDIA Fixer env (Python 3.12, torch 2.6 cu126, cosmos-predict2 1.0.9; see docs/nv_fixer.md).
#   NV_FIXER_ROOT (default $CS566D_DATA/nv_fixer) holds repo/ (nv-tlabs/Fixer clone), env/ (pixi env), weights/ (nvidia/Fixer)
source "$(dirname "${BASH_SOURCE[0]}")/../../../scripts/lib/env.sh"
FIX=${NV_FIXER_ROOT:-$DATA/nv_fixer}
export HF_HOME=$FIX/cache/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$FIX/cache
exec $FIX/env/.pixi/envs/default/bin/python "$@"
