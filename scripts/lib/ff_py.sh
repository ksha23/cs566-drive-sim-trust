#!/bin/bash
# Python in the FaithFusion env: a venv on top of the project (difix) env plus FaithFusion's modified
# diff-gaussian-rasterization (FisherRF-style squared-gradient backward), built by scripts/14_faithfusion.sh setup.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
export PYTHONPATH=$REPO/src${PYTHONPATH:+:$PYTHONPATH}
exec $DATA/faithfusion/env/bin/python "$@"
