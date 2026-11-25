#!/bin/bash
# DL3DV evaluation of the Difix models on the held-out TEST scenes (both sparsity levels) and the VAL scene.
#   usage: eval_job.sh released|trained|metrics|all     (released / trained = GPU inference, metrics = GPU metrics)
# Models: released Difix (nvidia/difix, no ref) and Difix-ref (nvidia/difix_ref); official trainer matched run
# (model_1801.pkl, no ref, 448x768); our trainer matched run (last.pt, no ref, 448x768); our trainer with reference
# view (best.pt by VAL LPIPS, 576x1024). Outputs $DATA/dl3dv/out/<model>/<key>.png; metrics -> results/dl3dv/.
H=$(cd $(dirname $0); pwd); REPO=$(cd $H/../..; pwd); DATA=${CS566D_DATA:-$HOME/cs566/data/D}; DL=$DATA/dl3dv
SMOKE=${CS566D_SMOKE:-$HOME/cs566/smoke/D}; export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
PY=$DATA/difix_reimpl/venv/bin/python; X=$REPO/src/fixers/difix_reimpl; O=$DL/out; T=$DL/dl3dv_test.json
export PYTHONPATH=$REPO/src
mkdir -p $REPO/results/dl3dv
run() { [ -f $O/$1/.done ] && return; $PY $X/infer.py "${@:2}" --val --json $T --out $O/$1 && touch $O/$1/.done; }
case $1 in
  released|all)
    run released --model released --no_ref
    run released_ref --model released ;;&
  trained|all)
    run official448 --model official --ckpt $DL/official448/checkpoints/model_1801.pkl --size 448x768 --no_ref
    run ours448 --model ours --ckpt $DL/ours448/last.pt --size 448x768 --no_ref
    run oursref --model ours --ckpt $DL/oursref/best.pt ;;&
  metrics|all)
    for L in 30 60; do
      PAIRS_JSON=$DL/dl3dv_test_s$L.json TARGETS_DIR=$DL/test_targets_s$L $PY $X/val_metrics.py $REPO/results/dl3dv/test_s$L.json \
        raw=- released=$O/released released_ref=$O/released_ref official448=$O/official448 ours448=$O/ours448 oursref=$O/oursref
    done ;;
esac
