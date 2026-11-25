"""Download and unpack EUVS [6] scenes (HF dataset ai4ce/EUVS-Benchmark, gated: accept the terms and log in first;
the token is read from the standard login location, so run WITHOUT the project's HF_HOME override).

EUVS levels (paper, Sec. 3): Setting1 = translation only (nuPlan, other lanes), Setting2 = ROTATION ONLY (nuPlan: train
on the 3 front + 3 rear cameras, test on the 2 side-facing cameras of the same traversals), Setting3 = TRANSLATION +
ROTATION (MARS / Argoverse2, other traversal routes through the same place, e.g. intersections).

Default selection (the smallest archives, to fit the time budget):
  rotation      Setting2/nuplan_vegas_location_53, nuplan_sg_location_6, nuplan_vegas_location_50   (92-104 MB)
  trans_rot     Setting3/MARS_location_41_scene_02, MARS_location_43_scene_01, MARS_location_24_scene_01 (0.35-0.65 GB)

usage: python src/euvs/download.py [--level rotation|trans_rot|all] [--scenes a,b] [--out $CS566D_DATA/euvs/scenes]
"""
import argparse, os, shutil, tarfile, zipfile

LEVELS = {
    "rotation": ["Setting2/nuplan_vegas_location_53.tar.gz", "Setting2/nuplan_sg_location_6.tar.gz",
                 "Setting2/nuplan_vegas_location_50.tar.gz"],
    "trans_rot": ["Setting3/MARS_location_41_scene_02.zip", "Setting3/MARS_location_43_scene_01.zip",
                  "Setting3/MARS_location_24_scene_01.zip"],
}
REPO = "ai4ce/EUVS-Benchmark"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="all"); ap.add_argument("--scenes", default=None)
    ap.add_argument("--out", default=os.path.expanduser(os.environ.get("CS566D_DATA", "~/cs566/data/D")) + "/euvs/scenes")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download, whoami
    try:
        print("HF user:", whoami()["name"], flush=True)
    except Exception as e:  # gated dataset: nothing can be downloaded without a valid login
        raise SystemExit(f"EUVS is gated and the HF login is not valid ({type(e).__name__}: {e}). "
                         "Accept the terms on huggingface.co/datasets/ai4ce/EUVS-Benchmark and run `huggingface-cli login`.")
    files = sum(LEVELS.values(), []) if a.level == "all" else LEVELS[a.level]
    if a.scenes:
        files = [f for f in files if any(s in f for s in a.scenes.split(","))]
    for f in files:
        level = "rotation" if f.startswith("Setting2") else "trans_rot" if f.startswith("Setting3") else "translation"
        name = os.path.basename(f).split(".")[0]
        dst = f"{a.out}/{level}/{name}"
        if os.path.exists(f"{dst}/.done"):
            print("have", dst, flush=True); continue
        p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=f"{a.out}/_archives")
        os.makedirs(dst, exist_ok=True)
        print("unpacking", p, "->", dst, flush=True)
        if p.endswith(".zip"):
            zipfile.ZipFile(p).extractall(dst)
        else:
            tarfile.open(p).extractall(dst)
        open(f"{dst}/.done", "w").write(f)
        os.remove(p)  # keep only the unpacked scene
    shutil.rmtree(f"{a.out}/_archives/.cache", ignore_errors=True)


if __name__ == "__main__":
    main()
