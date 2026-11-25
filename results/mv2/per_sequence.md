# MV2 per sequence (cross-vehicle frames) — `src/mv2/per_sequence.py`

| dir | seq | median lateral (m) | camera-height diff (m) | raw LPIPS | best fixer LPIPS (method) | raw PSNR | raw paint F1 |
|---|---|---|---|---|---|---|---|
| car2scooty | seq1 | 7.6 | -0.48 | 0.513 | 0.441 (difix_ref) | 13.78 | 0.00 |
| car2scooty | seq12 | 8.8 | -0.38 | 0.549 | 0.465 (difix_ref) | 13.31 | 0.01 |
| car2scooty | seq13 | 8.4 | -0.43 | 0.541 | 0.443 (difix_ref) | 15.47 | 0.03 |
| car2scooty | seq17 | 5.3 | -0.15 | 0.476 | 0.377 (difix) | 16.58 | 0.43 |
| car2scooty | seq21 | 4.0 | -0.39 | 0.416 | 0.282 (difix_ref) | 16.94 | 0.57 |
| scooty2car | seq1 | 7.6 | +0.48 | 0.628 | 0.468 (difix_ref) | 13.81 | 0.02 |
| scooty2car | seq12 | 8.8 | +0.38 | 0.619 | 0.474 (difix_ref) | 15.29 | 0.06 |
| scooty2car | seq13 | 8.4 | +0.43 | 0.620 | 0.464 (difix_ref) | 15.15 | 0.06 |
| scooty2car | seq17 | 5.3 | +0.15 | 0.621 | 0.443 (difix_ref) | 16.20 | 0.16 |
| scooty2car | seq21 | 4.0 | +0.39 | 0.473 | 0.293 (difix_ref) | 20.67 | 0.09 |

Correlation over the 5 sequences of the raw splat's mean LPIPS / PSNR with the lateral separation: car2scooty: r = +0.98 / -0.84; scooty2car: r = +0.75 / -0.82 (5 points each; indicative only).
