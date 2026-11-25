#!/bin/bash
# Stage 4b (one GPU job): the third ablation of our fixer, +HF +PatchGAN (w_gan 0.05 from step 1000), seeds 0 and 1.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
TRAIN=scene_20,scene_21,scene_22,scene_23,scene_24; VAL=scene_19
$GJ bash -c "for seed in 0 1; do grep -q DONE logs/unet_hfgan_s\$seed.log 2>/dev/null || $PY src/fixers/unet/train.py --pairs $DATA/pairs \
  --scenes $TRAIN --val_scenes $VAL --out $DATA/fixers/hfgan_s\$seed --total_steps 1800 --minutes 22 --w_hf 0.5 --w_gan 0.05 --seed \$seed >> logs/unet_hfgan_s\$seed.log 2>&1; done"
for d in $DATA/fixers/hfgan_s*; do cp $d/train_log.json results/training/unet_$(basename $d).json; done
