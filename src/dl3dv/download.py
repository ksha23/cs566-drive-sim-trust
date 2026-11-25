"""Download a small DL3DV subset for validating our Difix reimplementation (gated HF dataset; access granted to the
student's account). Default: all 11 scenes of DL3DV/DL3DV-10K-Sample, only what we use: COLMAP sparse model,
transforms.json and the 4x-downsampled images (960x540).
usage: python src/dl3dv/download.py [repo] [scene_hash ...]   -> $CS566D_DATA/dl3dv/<repo name>
"""
import os, sys
from huggingface_hub import snapshot_download
# the project env sets HF_HOME to the shared cache; the gated DL3DV repo needs the user's login token
if 'HF_TOKEN' not in os.environ and os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
    os.environ['HF_TOKEN'] = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import DATA

repo = sys.argv[1] if len(sys.argv) > 1 else "DL3DV/DL3DV-10K-Sample"
scenes = sys.argv[2:] or ["*"]
pats = ["meta.csv", "README.md"]
for s in scenes:
    pats += [f"{s}/colmap/colmap/sparse/0/cameras.bin", f"{s}/colmap/colmap/sparse/0/points3D.bin",
             f"{s}/colmap/colmap/sparse/0/images.bin", f"{s}/colmap/transforms.json", f"{s}/colmap/images_4/*"]
p = snapshot_download(repo, repo_type="dataset", local_dir=f"{DATA}/dl3dv/{repo.split('/')[-1]}", allow_patterns=pats,
                      max_workers=8)
print("done", p)
