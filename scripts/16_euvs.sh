#!/bin/bash
# Stage 16: EUVS [6] heading axis with REAL ground truth (MARS scenes of EUVS Setting3; docs/euvs.md).
#   download                 3 Setting3 scenes (gated HF dataset: needs a valid login; uses the standard token location)
#   fit <scene> <proto>      splat on the training views + render the test views (GPU job; proto L3 | rot60)
#   fitall                   every scene x proto as its own GPU job
#   fix <proto> <method>     one fixer on all test renders of a protocol (GPU job, resumable)
#   fixall                   every proto x fixer
#   lanes <proto>            CLRNet on the front-camera test views (real + every method; GPU job)
#   eval <proto>; summary    metrics (GPU job) and tables
source "$(dirname "$0")/lib/env.sh"; cd "$REPO"
E=$DATA/euvs; SCENES="MARS_location_41_scene_02 MARS_location_43_scene_01 MARS_location_24_scene_01"
METHODS="difix difix_ref nv_fixer difix_ft difix_reimpl_ref ours_hfgan_s0"
merge_meta() { $PY -c "import json,glob; json.dump(sum([json.load(open(p)) for p in sorted(glob.glob('$E/out/$1/*/meta.json'))],[]), open('$E/out/$1/meta_all.json','w'))"; }
fix() {
  p=$1; m=$2; merge_meta $p; A="$E/out x $E/fixed --meta $E/out/$p/meta_all.json --skip_existing"
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
lanes() {  # front camera only; horizon row = median over those views
  p=$1; O=$E/out/$p; merge_meta $p
  export XDG_CACHE_HOME=$SMOKE/cache TORCH_HOME=$SMOKE/cache/torch
  HZ=$($PY -c "import json,numpy as np; M=[m for m in json.load(open('$O/meta_all.json')) if m['channel']==2]; print(np.median([m['horizon'] for m in M]) if M else 0)")
  $PY -c "import json; M=[m for m in json.load(open('$O/meta_all.json')) if m['channel']==2]; open('$O/lane_paths.txt','w').write('\n'.join(m['path'] for m in M)); open('$O/lane_real.txt','w').write('\n'.join(m['real'] for m in M))"
  [ -s $O/lane_paths.txt ] || { echo "no front-camera views in $p"; return; }
  echo "horizon row $HZ"
  sed "s|^|$E/out/|" $O/lane_real.txt > $O/list_real.txt
  $DATA/lanes/env/bin/python src/eval/lanes/clrnet_infer.py --list $O/list_real.txt --root $E/out --out $O/lanes_real.json --horizon $HZ
  for m in raw $METHODS; do
    R=$E/fixed/$m; [ $m = raw ] && R=$E/out
    sed "s|^|$R/|" $O/lane_paths.txt > $O/list_$m.txt
    $DATA/lanes/env/bin/python src/eval/lanes/clrnet_infer.py --list $O/list_$m.txt --root $R --out $O/lanes_$m.json --horizon $HZ
  done
}
case $1 in
  download) env -u HF_HOME $SMOKE/envs/difix/bin/python src/euvs/download.py --level trans_rot ;;
  fit) $PY src/euvs/fit.py $E/scenes/trans_rot/$2 $3 ;;
  fitall) for p in L3 rot60; do for s in $SCENES; do
            $GJ bash scripts/16_euvs.sh fit $s $p > logs/euvs_fit_${p}_$s.log 2>&1; done; done ;;
  fix) fix $2 $3 ;;
  fixall) for p in L3 rot60; do for m in $METHODS; do
            $GJ bash scripts/16_euvs.sh fix $p $m > logs/euvs_fix_${p}_$m.log 2>&1; done; done ;;
  lanes) lanes $2 ;;
  eval) $PY src/euvs/evaluate.py eval $2 --methods raw,${METHODS// /,} ;;
  summary) CUDA_VISIBLE_DEVICES= $PY src/euvs/evaluate.py summary > /dev/null ;;
esac
