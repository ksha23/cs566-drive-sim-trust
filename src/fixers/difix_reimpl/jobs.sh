#!/bin/bash
# Our trainer vs the official trainer, same data (driving_pairs.json), same settings, 1801 steps each from SD-Turbo,
# each one GPU job (<= 60 min) through the shared lock.
#   ours        our driving model at 576x1024 (the official data loader's size)
#   official    official trainer; with the reference view (--mv_unet, V=2) it OOMs on 24 GB at 576x1024 (step 0)
#               and at 448x768 (step 2): it decodes both views without activation checkpointing and keeps bf16
#               AdamW states. The matched pair therefore runs WITHOUT the reference view (V=1) at 448x768
#               (>= 400 px high for the official 400x400 Gram crop).
#   ours448     our trainer, same setting (--no_ref, 448x768)
H=$(cd $(dirname $0); pwd); DX=${CS566D_DATA:-$HOME/cs566/data/D}/difix_reimpl; GJ=${CS566D_GPUJOB:-$HOME/cs566/gpujob.sh}; [ -n "${CS566D_GPUJOB:-}" ] || [ -x "$GJ" ] || GJ=env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
case $1 in
  ours)     $GJ bash $H/run_ours.sh $DX/ours_matched 1801 --minutes 57 --val_every 300 >> $DX/ours_matched.log 2>&1 ;;
  official) $GJ timeout 3500 bash $H/run_official.sh $DX/official_matched 1801 --checkpointing_steps 600 --data_height 448 --data_width 768 --no_mv --dataset_path=$DX/driving_pairs_noref.json >> $DX/official_matched.log 2>&1 ;;
  ours448)  $GJ bash $H/run_ours.sh $DX/ours_matched448 1801 --minutes 57 --val_every 300 --size 448x768 --no_ref >> $DX/ours_matched448.log 2>&1 ;;
esac
