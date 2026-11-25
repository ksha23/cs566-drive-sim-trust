#!/bin/bash
# Stage 17: MV2 [9] car <-> scooter (lateral offset + camera height, REAL ground truth; transfer to India / GoPro).
# docs/mv2.md.
#   download                  car front camera + scooter of every sequence (HF sanjay810/mv2-dataset, academic use)
#   masks <seq...>            YOLO11m-seg movable-object masks (GPU job)
#   geom <seq...>             triangulation with the given poses, road plane, camera heights, metric scale (CPU)
#   fit <seq> <dir>           splat on one vehicle, render the other + own held-out frames (GPU job; dir car2scooty|scooty2car)
#   fitall                    every sequence x direction (GPU jobs)
#   fix <dir> <method>        one fixer on all renders of a direction (GPU job, resumable);  fixall
#   lanes <dir>               CLRNet on real + every method (GPU job; horizon = median per vehicle)
#   eval <dir>; summary       metrics (GPU job) and tables
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
O=$DATA/mv2/out; F=$DATA/mv2/fixed
SEQS=${SEQS:-"seq1 seq2 seq3 seq4 seq5"}; DIRS="car2scooty scooty2car"
METHODS="difix difix_ref nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"
merge_meta() { $PY -c "import json,glob; json.dump(sum([json.load(open(p)) for p in sorted(glob.glob('$O/$1/*/meta.json'))],[]), open('$O/$1/meta_all.json','w'))"; }
fix() {
  d=$1; m=$2; merge_meta $d; A="$O x $F --meta $O/$d/meta_all.json --skip_existing"
  case $m in
    difix|difix_ref|nv_fixer) $PY src/fixers/apply.py $A --methods $m ;;
    difix_ft) $PY src/fixers/apply.py $A --methods difix_ft --ckpt $DATA/difix_ft/best_unet.pt ;;
    ours_hfgan_s0) $PY src/fixers/apply.py $A --methods ours --ckpt $DATA/fixers/hfgan_s0/best.pt --tag ours_hfgan_s0 ;;
    difix_reimpl_ref)
      HF_HOME=$SMOKE/hf TORCH_HOME=$SMOKE/cache/torch XDG_CACHE_HOME=$SMOKE/cache PYTHONPATH=$REPO/src \
        $DATA/difix_reimpl/venv/bin/python src/fixers/apply.py $A --methods difix_reimpl \
        --ckpt $DATA/difix_reimpl/ours_matched/best.pt --tag difix_reimpl_ref ;;
  esac
}
lanes() {  # one CLRNet pass per vehicle (car / scooter cameras have different pitch -> own horizon row)
  d=$1; D=$O/$d; merge_meta $d
  export XDG_CACHE_HOME=$SMOKE/cache TORCH_HOME=$SMOKE/cache/torch
  LP=$DATA/lanes/env/bin/python
  for v in car scooty; do
    HZ=$($PY -c "import json,numpy as np; M=[m for m in json.load(open('$D/meta_all.json')) if m['vehicle']=='$v']; print(np.median([m['horizon'] for m in M]) if M else '')")
    [ -z "$HZ" ] && continue
    $PY -c "import json; M=[m for m in json.load(open('$D/meta_all.json')) if m['vehicle']=='$v']; open('$D/lp_$v.txt','w').write('\n'.join(m['path'] for m in M)); open('$D/lr_$v.txt','w').write('\n'.join(m['real'] for m in M))"
    echo "$v horizon row $HZ"
    sed "s|^|$O/|" $D/lr_$v.txt > $D/list_real_$v.txt
    $LP src/eval/lanes/clrnet_infer.py --list $D/list_real_$v.txt --root $O --out $D/lanes_real_$v.json --horizon $HZ
    for m in raw $METHODS; do
      R=$F/$m; [ $m = raw ] && R=$O
      sed "s|^|$R/|" $D/lp_$v.txt > $D/list_${m}_$v.txt
      $LP src/eval/lanes/clrnet_infer.py --list $D/list_${m}_$v.txt --root $R --out $D/lanes_${m}_$v.json --horizon $HZ
    done
  done
  $PY -c "
import json, glob, os
for m in ['real', 'raw'] + '$METHODS'.split():
    J = {}
    for v in ('car', 'scooty'):
        p = '$D/lanes_%s_%s.json' % (m, v)
        if os.path.exists(p): J.update(json.load(open(p)))
    json.dump(J, open('$D/lanes_%s.json' % m, 'w'))"
}
case $1 in
  download) $PY src/mv2/download.py ;;
  masks) shift; $PY src/mv2/data.py masks "$@" ;;
  geom) shift; $PY src/mv2/data.py geom "$@" ;;
  fit) $PY src/mv2/fit.py $2 $3 ;;
  fitall) for s in $SEQS; do for d in $DIRS; do
            [ -f $O/$d/$s/meta.json ] || $GJ bash scripts/17_mv2.sh fit $s $d > logs/mv2_fit_${d}_$s.log 2>&1; done; done ;;
  fix) fix $2 $3 ;;
  fixall) for d in $DIRS; do for m in $METHODS; do
            $GJ bash scripts/17_mv2.sh fix $d $m > logs/mv2_fix_${d}_$m.log 2>&1; done; done ;;
  lanes) lanes $2 ;;
  eval) $PY src/mv2/evaluate.py eval $2 --methods raw,${METHODS// /,} ;;
  summary) CUDA_VISIBLE_DEVICES= $PY src/mv2/evaluate.py summary > /dev/null ;;
esac
