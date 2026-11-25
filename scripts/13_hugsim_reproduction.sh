#!/bin/bash
# Stage 13 (CPU): closed-loop summary + reproduction check of LTF vs. the HUGSIM paper's Table 13 (nuScenes rows).
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
CUDA_VISIBLE_DEVICES= $PY src/hugsim/summarize_closed_loop.py --out results/hugsim
CUDA_VISIBLE_DEVICES= $PY src/hugsim/reproduction_check.py
