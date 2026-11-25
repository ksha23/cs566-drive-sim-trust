#!/bin/bash
# Stage 12 (GPU job, <= 60 min): HUGSIM closed loop with the released LTF (NAVSIM) policy on a list of scenarios with
# one fixer between simulator and policy, for LTF seeds $SEEDS. Skips finished runs (re-queue to resume) and stops
# starting new runs after $BUDGET_S seconds. Summarize with src/hugsim/summarize_closed_loop.py.
# usage: SEEDS="0 1 2" $GJ scripts/12_hugsim_closed_loop.sh <none|difix|difix_ft|<unet name>> "<scenario yamls>" [--unet name=ckpt]
source "$(dirname "$0")/lib/env.sh"
source $SMOKE/hugsim_env.sh
fixer=$1; scen=$2; shift 2
OUT=$DATA/hugsim/closed_loop; t0=$(date +%s)
cd $SMOKE/HUGSIM
for seed in ${SEEDS:-0}; do
for s in $scen; do
  [ $(( $(date +%s) - t0 )) -gt ${BUDGET_S:-3000} ] && { echo "budget reached"; exit 0; }
  n=$(basename $s .yaml); o=$OUT/$fixer/seed$seed/$n
  [ -f $o/eval.json ] && continue
  rm -rf $o; mkdir -p $o
  timeout 900 pixi run python $REPO/src/hugsim/closed_loop_fix.py --scenario $s --fixer $fixer --seed $seed \
      --fixer_args "$*" --out $o > $o/sim.log 2>&1
  echo "$fixer seed$seed $n: $(grep -oE "Collision with background|Collision with foreground|Far from preset trajectory|Complete" $o/sim.log | tail -1) $(grep EVAL $o/sim.log | cut -c1-200)"
  pkill -u $USER -f "ltf_e2e_seed.py output=$o" ; pkill -u $USER -f "fixer_server.py --dir $o"; true
done
done
