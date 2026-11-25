#!/bin/bash
# Stage 5 (one GPU job, ~45 min): full fine-tune of the RELEASED Difix U-Net on our training pairs
# (fp32 master weights + 8-bit AdamW, bf16 autocast, 4000 steps, val selection on scene_19; peak ~18 GB).
# Our from-scratch reimplementation of Difix training (from SD-Turbo) is in src/fixers/difix_reimpl/.
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
$GJ $PY src/fixers/difix_ft/finetune.py --pairs $DATA/pairs --scenes scene_20,scene_21,scene_22,scene_23,scene_24 \
    --val_scenes scene_19 --out $DATA/difix_ft --total_steps 4000 --minutes 42 --val_every 1000 > logs/difix_ft.log 2>&1
mkdir -p results/training; cp $DATA/difix_ft/ft_log.json results/training/difix_ft.json
