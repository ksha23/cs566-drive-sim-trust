# Example frame

One evaluation frame of the held-out Para-Lane road scene_17, used by `scripts/quick_check.sh` and shown in the
report. The splat was fitted on lane 2 only and rendered at a real pose of lane 1, 4.1 m to the side.

| File | What it is |
|---|---|
| `real.png` | the real image recorded at that pose (lane 1) |
| `moving_mask.png` | Para-Lane's moving-object mask of `real.png` |
| `raw_render.png` | the splat rendered at that pose |
| `difix_ft.png` | the render repaired by Difix-FT (released Difix fine-tuned on our training pairs) |
| `our_reimplementation.png` | the render repaired by our re-implementation of the Difix training procedure (reference view) |

Our re-implementation paints a wide white marking that is not on the real road, so its lane-paint precision is low
while its lane geometry is right.

Data: Para-Lane (Z. Ni et al., 3DV 2025), CC BY-NC 4.0, https://www.modelscope.cn/datasets/nizqleo/Paralane.
The renders and repaired images are derived from it and are shared under the same non-commercial terms.
