#!/bin/bash
#SBATCH -p interactive
#SBATCH -J cs566-ff
#SBATCH --array=0-23%6
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler16
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o logs/ff_%A_%a.log
# Stage 22 'ff' on a Slurm cluster: one FaithFusion progressive update (scene x arm x clip) per array task, on the
# lowest-priority partition (never preempts other jobs; a preempted task is requeued and starts its clip again).
# Metrics run afterwards on the main machine (scripts/22_final_runs.sh), from the renders this writes to fixed/.
source "${CS566D_ROOT:-$HOME/cs566/D}/repo/scripts/euler/env.sh"
SC=(scene_10 scene_3 scene_11 scene_4); ARM=(flat eig); CL=(clip_0 clip_1 clip_2)
i=$SLURM_ARRAY_TASK_ID; s=${SC[i / 6]}; a=${ARM[(i / 3) % 2]}; c=${CL[i % 3]}
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $s $c $a"
[ -f $DATA/splats/$s/$c/prog_ff_$a/log.json ] && { echo "already done"; exit 0; }
scripts/lib/ff_py.sh -m progressive.faithfusion.prog_ff $s $c $a
