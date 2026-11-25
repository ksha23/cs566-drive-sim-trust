#!/bin/bash
# Stage 3 (CPU): on-path held-out quality per lane (+ cycle splats) and pair counts -> results/onpath, results/pairs.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
CUDA_VISIBLE_DEVICES= $PY src/eval/collect_splat_quality.py
