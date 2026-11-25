"""Download N scene archives of DL3DV/DL3DV-ALL-960P (first N zips of a subset folder, sorted by name, deterministic)
and unzip them to $CS566D_DATA/dl3dv/960P/<hash>/. Used to enlarge the 11-scene DL3DV-10K-Sample for the Difix
reimplementation check.  usage: python src/dl3dv/download_960p.py [N=14] [folder=1K]
"""
import os, sys, zipfile
from huggingface_hub import HfApi, hf_hub_download
if 'HF_TOKEN' not in os.environ and os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
    os.environ['HF_TOKEN'] = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import DATA

N = int(sys.argv[1]) if len(sys.argv) > 1 else 14
folder = sys.argv[2] if len(sys.argv) > 2 else "1K"
R = "DL3DV/DL3DV-ALL-960P"; out = f"{DATA}/dl3dv/960P"; os.makedirs(out, exist_ok=True)
zips = sorted(x.path for x in HfApi().list_repo_tree(R, path_in_repo=folder, repo_type="dataset") if x.path.endswith(".zip"))[:N]
for z in zips:
    h = os.path.basename(z)[:-4]
    if os.path.isdir(f"{out}/{h}"):
        continue
    p = hf_hub_download(R, z, repo_type="dataset", local_dir=f"{out}/_zips")
    zipfile.ZipFile(p).extractall(f"{out}/{h}.tmp"); os.rename(f"{out}/{h}.tmp", f"{out}/{h}"); os.remove(p)
    print("got", h, flush=True)
