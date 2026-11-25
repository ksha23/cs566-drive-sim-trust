#!/bin/bash
# Official Difix trainer (nv-tlabs/Difix3D src/train_difix.py, patched copy in $DX/official_src made with
# official_trainer.patch in this folder) with the SAME data and settings as our trainer's matched run.
# usage: run_official.sh <out_dir> <steps> [extra args]
DX=${CS566D_DATA:-$HOME/cs566/data/D}/difix_reimpl; SMOKE=${CS566D_SMOKE:-$HOME/cs566/smoke/D}
export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
OUT=$1; STEPS=$2; shift 2
MV=--mv_unet; if [ "$1" == "--no_mv" ] || [[ " $* " == *" --no_mv "* ]]; then MV=""; set -- $(echo "$@" | sed "s/--no_mv//"); fi
cd $DX/official_src
$DX/venv/bin/python -m accelerate.commands.launch --num_processes 1 --num_machines 1 --mixed_precision=bf16 --dynamo_backend no train_difix.py \
  --output_dir=$OUT --dataset_path=$DX/driving_pairs.json --max_train_steps $STEPS \
  --resolution=512 --learning_rate 2e-5 --lr_scheduler constant_with_warmup --lr_warmup_steps 100 \
  --train_batch_size=1 --dataloader_num_workers 6 --enable_xformers_memory_efficient_attention --gradient_checkpointing \
  --checkpointing_steps=$((STEPS - 1)) --eval_freq 100000 --num_samples_eval 5 --viz_freq 100000 \
  --lambda_lpips 1.0 --lambda_l2 1.0 --lambda_gram 0.5 --gram_loss_warmup_steps 0 --allow_tf32 --seed 0 \
  --report_to tensorboard --tracker_project_name difix --tracker_run_name official --timestep 199 $MV "$@"
