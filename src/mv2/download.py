"""Download MV2 [9] (HF sanjay810/mv2-dataset, academic use): the car's front camera and the scooter ("scooty") of every
sequence (images, poses, intrinsics) plus per-sequence scale_factor / transform. Drone and car side views are skipped.
Retries (the Hub rate-limits many small files; resumes where it stopped).
usage: python src/mv2/download.py [seq ...]     (default: all sequences) -> $CS566D_DATA/mv2/raw
"""
import os, sys, time
from huggingface_hub import snapshot_download
if 'HF_TOKEN' not in os.environ and os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
    os.environ['HF_TOKEN'] = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.config import DATA

seqs = sys.argv[1:] or ["seq*"]
pats = ["README.md"]
for s in seqs:
    pats += [f"{s}/car/images/car_front_*", f"{s}/car/poses/car_front_0*", f"{s}/car/intrinsics/car_front_0*",
             f"{s}/scooty/*", f"{s}/*.npy"]
for attempt in range(30):
    try:
        p = snapshot_download("sanjay810/mv2-dataset", repo_type="dataset", local_dir=f"{DATA}/mv2/raw",
                              allow_patterns=pats, ignore_patterns=["*/._*", "*/drone/*"], max_workers=4)
        print("done", p, flush=True); break
    except Exception as e:
        print("attempt", attempt, "failed:", str(e)[-200:], flush=True); time.sleep(120)
