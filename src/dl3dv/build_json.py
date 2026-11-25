"""Pairs JSONs for the DL3DV validation of our Difix reimplementation, in the layout both trainers read
({"train": {id: {image, target_image, ref_image, prompt, fg}}, "test": {...}}).
Scene split of the 11 DL3DV-10K-Sample scenes (the only DL3DV repo the account is authorised for): sorted by hash,
every 3rd scene (index % 3 == 0) except the VAL scene is a held-out TEST scene (3 scenes); VAL = dafa9c7c
(training-time monitoring / our ref model's checkpoint choice only); the other 6 are TRAIN.
Two sparsity levels (src/dl3dv/fit_pairs.py): S = 30 and S = 60 (splat trained on every S-th frame).
  train: sparse-reconstruction pairs (held-out frames) + underfitting pairs (all frames), both levels, TRAIN scenes
  test split of dl3dv_pairs.json: every 2nd sparse pair of the VAL scene (both levels)
  dl3dv_test_s30.json / dl3dv_test_s60.json / dl3dv_test.json (both): every 3rd sparse pair of the TEST scenes
Writes $CS566D_DATA/dl3dv/{dl3dv_pairs.json, dl3dv_pairs_noref.json, dl3dv_test*.json, split.json}.
"""
import json, os
from common.config import DATA

W = f"{DATA}/dl3dv/work"
P = "remove degradation"
LEVELS = (30, 60)
VAL = "dafa9c7cbda9d1ddaa8a2b51fc8c54f4eb44161f5e5c53685dc744580ca77751"


def item(m, kind, ref=True):
    d = dict(image=f"{W}/{m[kind]}", target_image=f"{W}/{m['real']}", prompt=P, fg=f"{W}/{m['scene']}/fg_empty.png",
             variant=kind, scene=m["scene"], stride=m["stride"])
    if ref:
        d["ref_image"] = f"{W}/{m['ref']}"
    return d


def key(m, kind):
    return f"{m['scene'][:8]}_s{m['stride']}_{kind}_{m['name']}"


def main():
    scenes = sorted(os.listdir(W))
    scenes = [s for s in scenes if all(os.path.exists(f"{W}/{s}/s{L}/meta.json") for L in LEVELS)]
    assert len(scenes) == 11, scenes
    test = [s for k, s in enumerate(scenes) if k % 3 == 0 and s != VAL]
    train = [s for s in scenes if s not in test + [VAL]]
    M = {(s, L): json.load(open(f"{W}/{s}/s{L}/meta.json")) for s in scenes for L in LEVELS}
    for ref in (True, False):
        J = {"train": {}, "test": {}}
        for s in train:
            for L in LEVELS:
                for m in M[s, L]:
                    if m["sparse"]:
                        J["train"][key(m, "sparse")] = item(m, "sparse", ref)
                    J["train"][key(m, "underfit")] = item(m, "underfit", ref)
        for L in LEVELS:
            for k, m in enumerate([m for m in M[VAL, L] if m["sparse"]]):
                if k % 2 == 0:
                    J["test"][key(m, "sparse")] = item(m, "sparse", ref)
        json.dump(J, open(f"{DATA}/dl3dv/dl3dv_pairs{'' if ref else '_noref'}.json", "w"))
    T = {}
    for L in LEVELS:
        T[L] = {"train": {}, "test": {}}
        for s in test:
            for k, m in enumerate([m for m in M[s, L] if m["sparse"]]):
                if k % 3 == 0:
                    T[L]["test"][key(m, "sparse")] = item(m, "sparse", True)
        json.dump(T[L], open(f"{DATA}/dl3dv/dl3dv_test_s{L}.json", "w"))
    json.dump({"train": {}, "test": {**T[30]["test"], **T[60]["test"]}}, open(f"{DATA}/dl3dv/dl3dv_test.json", "w"))
    logs = {f"{s[:8]}_s{L}": json.load(open(f"{W}/{s}/s{L}/log.json"))["heldout_psnr_mean"] for s in scenes for L in LEVELS}
    split = dict(train=train, val=[VAL], test=test, n_train=len(J["train"]), n_val=len(J["test"]),
                 n_test={L: len(T[L]["test"]) for L in LEVELS},
                 n_train_by_kind={k: sum(v["variant"] == k for v in J["train"].values()) for k in ("sparse", "underfit")},
                 heldout_psnr_by_scene_level=logs)
    json.dump(split, open(f"{DATA}/dl3dv/split.json", "w"), indent=1)
    print(json.dumps(split, indent=1))


if __name__ == "__main__":
    main()
