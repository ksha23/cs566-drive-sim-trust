#!/bin/bash
#SBATCH -p interactive
#SBATCH -J cs566-hd
#SBATCH --array=0-14%6
#SBATCH --gres=gpu:1
#SBATCH --exclude=euler16
#SBATCH -c 6
#SBATCH --mem=48G
#SBATCH -t 02:00:00
#SBATCH --requeue
#SBATCH -o /srv/home/kasha2/cs566/D/logs/hd_%A_%a.log
# Stage 22 'heading' on a Slurm cluster: one fixer on one road's heading / height renders per array task (the renders
# come from the main machine). The 0-deviation frames are left out: on the main machine they are links to the lateral
# outputs (scripts/15_heading_height.sh, link_zero). NVIDIA Fixer, lanes and metrics run on the main machine.
source /srv/home/kasha2/cs566/D/repo/scripts/euler/env.sh
SC=(scene_3 scene_11 scene_4); ME=(difix difix_ref difix_ft difix_reimpl_ref ours_hfgan_s0)
i=$SLURM_ARRAY_TASK_ID; s=${SC[i / 5]}; m=${ME[i % 5]}
echo "$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) $s $m"
HR=$DATA/heading; M=$HR/$s/meta_nozero.json
$PY -c "import json; json.dump([m for m in json.load(open('$HR/$s/meta.json')) if '/y+0_z+0.0/' not in m['path']], open('$M', 'w'))"
A="$HR $s $HR/fixed --meta $M --skip_existing"
case $m in
  difix|difix_ref) $PY src/fixers/apply.py $A --methods $m ;;
  difix_ft) $PY src/fixers/apply.py $A --methods difix_ft --ckpt $DATA/difix_ft/best_unet.pt ;;
  ours_hfgan_s0) $PY src/fixers/apply.py $A --methods ours --ckpt $DATA/fixers/hfgan_s0/best.pt --tag ours_hfgan_s0 ;;
  difix_reimpl_ref) PYTHONPATH=$REPO/src $DATA/difix_reimpl/venv/bin/python src/fixers/apply.py $A --methods difix_reimpl \
      --ckpt $DATA/difix_reimpl/ours_matched/best.pt --tag difix_reimpl_ref ;;
esac
