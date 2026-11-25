#!/bin/bash
# Outputs of ours / official-trainer / released Difix on the VAL triplets (scene_19) and on the TEST scenes
# (scene_16, scene_17), then metrics. Each mode is one GPU job (run through ~/cs566/gpujob.sh).
#   usage: eval_job.sh base     # ours + released on val, ours on test (+ image metrics and CLRNet lanes)
#          eval_job.sh official # matched pair (official trainer / ours at 448x768, last checkpoints) on val + test, val table
H=$(cd $(dirname $0); pwd); REPO=$(cd $H/../../..; pwd); DATA=${CS566D_DATA:-$HOME/cs566/data/D}; DX=$DATA/difix_reimpl
SMOKE=${CS566D_SMOKE:-$HOME/cs566/smoke/D}; export HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache
PY=$DX/venv/bin/python; OURS=$DX/ours_matched/best.pt
mkdir -p $REPO/results/difix_reimpl
test_eval() {  # method
  cd $REPO
  for s in scene_16 scene_17; do
    scripts/lib/py.sh src/eval/evaluate.py $DATA/pairs $s $DATA/fixed results/difix_reimpl/eval_${s}_$1.json --methods $1
    scripts/lib/lanes_scene.sh $s $DATA/fixed results/difix_reimpl/ntl_${s}_$1.json $1
  done
}
case $1 in
  base)
    $PY $H/infer.py --model ours --ckpt $OURS --val --out $DX/val_out/ours_ref
    $PY $H/infer.py --model released --val --out $DX/val_out/released_ref
    $PY $H/infer.py --model released --no_ref --val --out $DX/val_out/released
    $PY $H/infer.py --model ours --ckpt $OURS --scenes scene_16,scene_17 --method difix_reimpl_ref
    test_eval difix_reimpl_ref ;;
  official)
    OFF=$(ls -v $DX/official_matched/checkpoints/model_*.pkl | tail -1); echo "official ckpt $OFF"
    $PY $H/infer.py --model official --ckpt $OFF --val --out $DX/val_out/official448 --size 448x768 --no_ref
    $PY $H/infer.py --model ours --ckpt $DX/ours_matched448/last.pt --val --out $DX/val_out/ours448 --size 448x768 --no_ref
    $PY $H/val_metrics.py $REPO/results/difix_reimpl/val_scene19.json raw=- ours_ref=$DX/val_out/ours_ref \
        ours448=$DX/val_out/ours448 official448=$DX/val_out/official448 \
        released_ref=$DX/val_out/released_ref released=$DX/val_out/released
    $PY $H/infer.py --model official --ckpt $OFF --scenes scene_16,scene_17 --method difix_official448 --size 448x768 --no_ref
    $PY $H/infer.py --model ours --ckpt $DX/ours_matched448/last.pt --scenes scene_16,scene_17 --method difix_reimpl448 --size 448x768 --no_ref
    test_eval difix_official448; test_eval difix_reimpl448 ;;
esac
