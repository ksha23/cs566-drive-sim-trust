#!/bin/bash
# Quick check without datasets or a GPU (README, "Quick check"; a few minutes on a laptop CPU):
#   1. regenerate the final tables and robustness checks in results/final/ from the per-image result files that ship
#      in results/, and compare every CSV and tables.md with the shipped copy (numbers to a relative 1e-9; CSV float
#      formatting differs between pandas versions; the PNG figures are not compared)
#   2. score the bundled example frame (examples/scene_17_frame, Figure 2 of the report) with the metrics of the results
# usage: bash scripts/quick_check.sh        (PYTHON=... to choose the interpreter; requirements-cpu.txt)
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=$PWD/src${PYTHONPATH:+:$PYTHONPATH} CUDA_VISIBLE_DEVICES=
PY=${PYTHON:-python3}
echo "== 1. tables and checks from results/ (results/final/)"
T=$(mktemp -d); cp results/final/*.csv results/final/tables.md "$T"/
$PY src/eval/final_report.py > /dev/null
$PY src/eval/midterm_checks.py > /dev/null
$PY - "$T" <<'PY'
import glob, os, sys
import numpy as np, pandas as pd
T = sys.argv[1]; fs = sorted(glob.glob(f"{T}/*")); bad = []
for f in fs:
    g = os.path.join("results/final", os.path.basename(f))
    if f.endswith(".md"):
        same = open(f).read() == open(g).read()
    else:
        a, b = pd.read_csv(f), pd.read_csv(g)
        same = a.shape == b.shape and list(a.columns) == list(b.columns) and all(
            np.allclose(a[c], b[c], rtol=1e-9, atol=1e-12, equal_nan=True)
            if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c])
            else a[c].astype(str).equals(b[c].astype(str)) for c in a.columns)
    if not same:
        bad.append(os.path.basename(f))
print(f"results/final: {len(fs) - len(bad)} of {len(fs)} tables and CSVs reproduced"
      + (f"; DIFFERENT: {', '.join(bad)}" if bad else ""))
PY
rm -rf "$T"
echo "== 2. metrics of the example frame against the real image (examples/scene_17_frame)"
E=examples/scene_17_frame
for m in raw_render difix_ft our_reimplementation; do
  echo "-- $m"
  $PY src/tools/pair_metrics.py $E/$m.png $E/real.png --fg $E/moving_mask.png --cpu
done
