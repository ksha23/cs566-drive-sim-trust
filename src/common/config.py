"""All machine-specific paths in one place. Override with environment variables (see scripts/lib/env.sh):
  CS566D_DATA   data/output root (Para-Lane extracts, splats, pairs, fixer checkpoints, fixed renders)
  CS566D_SMOKE  root of the external envs/repos/weights (Difix3D repo, HF cache, YOLO weights, HUGSIM)"""
import os

DATA = os.path.expanduser(os.environ.get("CS566D_DATA", "~/cs566/data/D"))
SMOKE = os.path.expanduser(os.environ.get("CS566D_SMOKE", "~/cs566/smoke/D"))
PARALANE_ROOTS = [os.path.join(DATA, "paralane/extracted/0315"),               # scenes we downloaded
                  os.path.join(SMOKE, "data/paralane/extracted/0315")]         # scene_16 (smoke test)
DIFIX3D_SRC = os.path.join(SMOKE, "Difix3D/src")                               # released Difix code (model def., pipeline)
YOLO_WEIGHTS = os.path.join(SMOKE, "weights/yolo11m.pt")
HUGSIM_ROOT = os.path.join(SMOKE, "HUGSIM")
HUGSIM_DATA = os.path.join(DATA, "hugsim")
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
