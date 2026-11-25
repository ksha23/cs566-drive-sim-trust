#!/bin/bash
# CLRNet lanes + NTL-IoU for one scene's eval subset (GPU; ~13 ms/img). CLRNet runs in its own env ($DATA/lanes/env).
# usage: scripts/lib/lanes_scene.sh <scene> <fixed_root> <out_rows_json> <method>...
#   method "raw" = the splat render in the pairs dir; any other name = <fixed_root>/<name>/<meta path>
# Real-image detections are cached per scene in $DATA/lanes/out/<scene>/real.json.
set -e
source "$(dirname "$0")/env.sh"; cd "$REPO"
S=$1; FIX=$2; OUT=$3; shift 3
PAIRS=${PAIRS:-$DATA/pairs}; LOUT=$DATA/lanes/out/$S; mkdir -p $LOUT
# eval subset: same selection as src/fixers/apply.py:eval_items (stride 2)
$PY - "$PAIRS" "$S" "$LOUT" <<'PY'
import sys
from fixers.apply import eval_items
P, S, L = sys.argv[1:4]
it = eval_items(P, S, 2)
open(f"{L}/real.txt", "w").write("\n".join(sorted({f"{P}/{m['real']}" for m in it})))
open(f"{L}/renders.txt", "w").write("\n".join(m["path"] for m in it))
PY
LP=$DATA/lanes/env/bin/python
export XDG_CACHE_HOME=$SMOKE/cache TORCH_HOME=$SMOKE/cache/torch
[ -f $LOUT/real.json ] || $LP src/eval/lanes/clrnet_infer.py --list $LOUT/real.txt --root $PAIRS --out $LOUT/real.json
ARGS=""
for m in "$@"; do
  if [ "$m" = raw ]; then R=$PAIRS; else R=$FIX/$m; fi
  sed "s|^|$R/|" $LOUT/renders.txt > $LOUT/list_$m.txt
  $LP src/eval/lanes/clrnet_infer.py --list $LOUT/list_$m.txt --root $R --out $LOUT/$m.json
  ARGS="$ARGS --method $m=$LOUT/$m.json"
done
$PY src/eval/lanes/ntl_iou.py --pairs $PAIRS --scene $S --real $LOUT/real.json $ARGS --out $OUT
