#!/bin/bash
# Our Difix trainer. usage: run_ours.sh <out_dir> <steps> [extra train.py args]
DX=${CS566D_DATA:-$HOME/cs566/data/D}/difix_reimpl; SMOKE=${CS566D_SMOKE:-$HOME/cs566/smoke/D}
export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
OUT=$1; STEPS=$2; shift 2
$DX/venv/bin/python $(dirname $0)/train.py --json $DX/driving_pairs.json --out $OUT --steps $STEPS "$@"
