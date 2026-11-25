# Sourced by the Euler job scripts: the same roots the pipeline reads (scripts/lib/env.sh), on the cluster.
R=${CS566D_ROOT:-/srv/home/kasha2/cs566/D}
export CS566D_DATA=$R/data CS566D_SMOKE=$R/smoke CS566D_GPUJOB=env   # Slurm schedules the GPU: no lock wrapper
export HF_HOME=$R/smoke/hf HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
cd $R/repo
source scripts/lib/env.sh
