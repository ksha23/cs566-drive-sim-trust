#!/bin/bash
# Stage 1: download + extract Para-Lane scenes. Para-Lane (CC BY-NC) is a 335-part split ZIP on ModelScope; each scene
# is a contiguous byte range, so we fetch only its parts plus the two parts holding the ZIP index (333, 334).
# Part ranges per scene: configs/paralane_scene_index.json.
# usage: scripts/01_download_paralane.sh scene_20 scene_22 ...
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
P=$DATA/paralane/parts; mkdir -p $P
parts=$($PY - "$@" <<'PY'
import json, sys
idx = json.load(open("configs/paralane_scene_index.json"))
s = {333, 334}
for sc in sys.argv[1:]:
    a, b = idx[f"0315/{sc}"]["parts"]; s |= set(range(a, b + 1))
print(" ".join(map(str, sorted(s))))
PY
)
URL="https://www.modelscope.ai/api/v1/datasets/nizqleo/Paralane/repo?Revision=master&FilePath=paralane_v_1_0.zip.parts"
[ -s $P/SHA256SUMS ] || curl -sL -o $P/SHA256SUMS "$URL/SHA256SUMS"
for i in $parts; do
  f=$(printf "part-%05d" $i); [ -s $P/$f ] && continue
  curl -sL --retry 5 -o $P/$f "$URL/paralane_v_1_0.zip.$f" && echo "got $f"
done
$PY scripts/lib/extract_scenes.py "$@" && ls $P/part-* | grep -v -E "part-0033[34]$" | xargs -r rm -f   # keep only the index parts
