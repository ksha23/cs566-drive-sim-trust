#!/bin/bash
# Driver: after both fit stages finished, run the rest of stage 19 as separate GPU jobs through the shared lock.
cd $(dirname $0)/../..; GJ=${CS566D_GPUJOB:-$HOME/cs566/gpujob.sh}; [ -n "${CS566D_GPUJOB:-}" ] || [ -x "$GJ" ] || GJ=env; W=${CS566D_DATA:-$HOME/cs566/data/D}/dl3dv/work
until [ $(ls $W/*/s60/meta.json 2>/dev/null | wc -l) -ge 11 ] && [ $(ls $W/*/s30/meta.json 2>/dev/null | wc -l) -ge 11 ]; do sleep 60; done
scripts/19_dl3dv.sh json > logs/dl3dv_json.log 2>&1
$GJ scripts/19_dl3dv.sh ours448 && echo ours448 done
$GJ scripts/19_dl3dv.sh official448 && echo official448 done
$GJ scripts/19_dl3dv.sh oursref && echo oursref done
$GJ timeout 3550 scripts/19_dl3dv.sh eval released && echo released done
$GJ timeout 3550 scripts/19_dl3dv.sh eval trained && echo trained done
$GJ timeout 3550 scripts/19_dl3dv.sh eval metrics && echo metrics done
