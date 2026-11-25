#!/bin/bash
# Driver for stage 17 after the fits: every fixer (one GPU job per direction x method), CLRNet lanes (GPU job per
# direction), metrics (GPU job per direction), summary + figures (CPU). Resumable (apply.py --skip_existing).
cd $(dirname $0)/../..; GJ=${CS566D_GPUJOB:-$HOME/cs566/gpujob.sh}; [ -n "${CS566D_GPUJOB:-}" ] || [ -x "$GJ" ] || GJ=env; export SEQS="seq1 seq21 seq17 seq12 seq13"
while pgrep -f "17_mv2.sh fitall" > /dev/null; do sleep 60; done
bash scripts/17_mv2.sh fixall
for d in car2scooty scooty2car; do
  $GJ bash -c "bash scripts/17_mv2.sh lanes $d && bash scripts/17_mv2.sh eval $d" > logs/mv2_lanes_eval_$d.log 2>&1
done
bash scripts/17_mv2.sh summary > logs/mv2_summary.log 2>&1
echo MV2 CHAIN DONE
