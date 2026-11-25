# Sourced by every entry script: machine-specific paths (override via environment) and helpers.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
export CS566D_DATA=${CS566D_DATA:-$HOME/cs566/data/D}          # data + outputs (not in git)
export CS566D_SMOKE=${CS566D_SMOKE:-$HOME/cs566/smoke/D}       # external envs, repos, weights
DATA=$CS566D_DATA; SMOKE=$CS566D_SMOKE
GJ=${CS566D_GPUJOB:-$HOME/cs566/gpujob.sh}                     # shared-GPU lock wrapper (waits for a free GPU)
[ -n "${CS566D_GPUJOB:-}" ] || [ -x "$GJ" ] || GJ=env           # no wrapper on this machine: run GPU stages directly
PY=$REPO/scripts/lib/py.sh                                      # python in the project (difix) env
mkdir -p "$REPO/logs"
