"""Record which DL3DV Hugging Face repos the logged-in account can download from (one small file each) ->
results/dl3dv/access_check.json. usage: scripts/lib/py.sh src/dl3dv/access_check.py"""
import datetime, json, os, sys, tempfile
from huggingface_hub import HfApi, hf_hub_download
if 'HF_TOKEN' not in os.environ and os.path.exists(os.path.expanduser('~/.cache/huggingface/token')):
    os.environ['HF_TOKEN'] = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
from common.config import REPO

PROBES = {"DL3DV/DL3DV-10K-Sample": "5c3af581028068a3c402c7cbe16ecf9471ddf2897c34ab634b7b1b6cf81aba00/colmap/transforms.json",
          "DL3DV/DL3DV-Benchmark": "032dee9fb0a8bc1b90871dc5fe950080d0bcd3caf166447f44e60ca50ac04ec7/nerfstudio/transforms.json",
          "DL3DV/DL3DV-ALL-960P": "1K/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f.zip"}
api = HfApi(); out = dict(date=datetime.datetime.now().isoformat(timespec="seconds"), user=api.whoami()["name"], repos={})
for r, f in PROBES.items():
    try:
        if f.endswith(".zip"):  # do not pull a 300 MB archive: a ranged HEAD via get_hf_file_metadata is enough
            from huggingface_hub import get_hf_file_metadata, hf_hub_url
            import requests
            u = hf_hub_url(r, f, repo_type="dataset")
            resp = requests.get(u, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}", "Range": "bytes=0-99"}, timeout=60)
            ok = resp.status_code in (200, 206); msg = f"HTTP {resp.status_code}" + ("" if ok else f": {resp.text[:200]}")
        else:
            with tempfile.TemporaryDirectory() as t:
                hf_hub_download(r, f, repo_type="dataset", local_dir=t); ok, msg = True, "downloaded"
    except Exception as e:
        ok, msg = False, str(e).strip().splitlines()[-1][:300]
    out["repos"][r] = dict(probe_file=f, access=ok, message=msg); print(r, ok, msg, flush=True)
os.makedirs(f"{REPO}/results/dl3dv", exist_ok=True)
json.dump(out, open(f"{REPO}/results/dl3dv/access_check.json", "w"), indent=1)
