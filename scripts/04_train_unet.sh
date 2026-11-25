#!/bin/bash
# Stage 4: our fixer (smaller version, from scratch) = compact residual U-Net, 1800 steps each:
# {base (L1+LPIPS), +HF (Sobel high-frequency loss)} x seeds {0,1}; +HF+GAN was implemented but not trained (time).
# Grouped into two GPU jobs (<= 60 min); running two concurrently OOMs the 24 GB card.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
TRAIN=scene_20,scene_21,scene_22,scene_23,scene_24; VAL=scene_19; STEPS=${STEPS:-1800}
run() {  # name w_hf w_gan seed
  echo "grep -q DONE logs/unet_$1_s$4.log 2>/dev/null || $PY src/fixers/unet/train.py --pairs $DATA/pairs --scenes $TRAIN \
--val_scenes $VAL --out $DATA/fixers/$1_s$4 --total_steps $STEPS --minutes 19 --w_hf $2 --w_gan $3 --seed $4 >> logs/unet_$1_s$4.log 2>&1;"
}
$GJ bash -c "$(run base 0 0 0) $(run base 0 0 1)"
$GJ bash -c "$(run hf 0.5 0 0) $(run hf 0.5 0 1)"
# val-based selection over all trained configs (incl. stage 4b's +HF+GAN): config with the best mean val LPIPS over
# seeds, seed 0 -> results/training/selected_ours.txt (re-run this block after stage 4b)
mkdir -p results/training
$PY - <<'PY' | tee results/training/selected_ours.txt
import json, glob, os, collections
from common.config import DATA
v = collections.defaultdict(list)
for f in glob.glob(f"{DATA}/fixers/*/train_log.json"):
    run = f.split("/")[-2]; v[run.rsplit("_s", 1)[0]].append(json.load(open(f))["best"]["val_lpips"])
for k in sorted(v):
    print(f"# {k}: mean best val LPIPS {sum(v[k]) / len(v[k]):.4f} over {len(v[k])} seeds")
print("selected", min(v, key=lambda k: sum(v[k]) / len(v[k])) + "_s0")
PY
for d in $DATA/fixers/*; do cp $d/train_log.json results/training/unet_$(basename $d).json; done
