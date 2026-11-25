#!/bin/bash
#SBATCH -p interactive
#SBATCH -J cs566-score
#SBATCH --array=0-6
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler16
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH --requeue
#SBATCH -o /srv/home/kasha2/cs566/D/logs/score_%A_%a.log
# Stage 22 'collect' on a Slurm cluster: the same metric and CLRNet-lane commands as scripts/22_final_runs.sh collect,
# for outputs computed on the cluster. Tasks 0-3: FaithFusion updates on one road (ff_eval); tasks 4-6: heading /
# camera height on one road (the 0-deviation frames and NVIDIA Fixer's outputs copied from the main machine).
# Resumable after requeue: lane detections and heading rows skip finished methods. Results are copied back to the
# main machine before stage 23.
source /srv/home/kasha2/cs566/D/repo/scripts/euler/env.sh
T=(ff:scene_10 ff:scene_3 ff:scene_11 ff:scene_4 hd:scene_3 hd:scene_11 hd:scene_4)
k=${T[$SLURM_ARRAY_TASK_ID]}; s=${k#*:}
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $k"
set -e
if [ ${k%%:*} = ff ]; then
  mkdir -p results/faithfusion
  $PY src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/faithfusion/eval_${s}_ff.json --methods prog_ff_flat,prog_ff_eig
  scripts/lib/lanes_scene.sh $s $DATA/fixed results/faithfusion/ntl_${s}_ff.json prog_ff_flat prog_ff_eig
else
  bash scripts/15_heading_height.sh lanes $s
  bash scripts/15_heading_height.sh eval $s
fi
echo "done $k"
