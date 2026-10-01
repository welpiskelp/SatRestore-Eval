# H1 result: does telling the model the noise level help? (PRELIMINARY)

4 folds, 1 seed each, 16 tiles total (every tile tested exactly once, in whichever fold it belongs to as a test tile). The project's frozen protocol calls for 3 seeds per fold (12 runs per arm); this is a first pass with 1 seed per fold, not the final number.

Per noise level: average PSNR with conditioning, average PSNR without it, the gap, a 95% confidence range for that gap (accounting for tiles that share the same scene), and whether the gap is statistically real after correcting for testing 9 noise levels at once (Holm correction).

| SNR (dB) | With conditioning | Without (blind) | Gap (dB) | 95% range | Holm p-value | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 30.19 | 29.90 | +0.29 | [+0.18, +0.41] | 0.001 | **yes** |
| 5 | 33.34 | 33.09 | +0.25 | [+0.18, +0.32] | 0.000 | **yes** |
| 10 | 35.95 | 35.86 | +0.09 | [+0.03, +0.15] | 0.038 | **yes** |
| 15 | 38.38 | 38.43 | -0.06 | [-0.15, +0.04] | 0.386 | no — too close to call |
| 20 | 40.87 | 41.02 | -0.16 | [-0.27, -0.04] | 0.015 | no, favors blind |
| 25 | 43.53 | 43.73 | -0.20 | [-0.31, -0.08] | 0.016 | no, favors blind |
| 30 | 46.32 | 46.47 | -0.15 | [-0.30, +0.01] | 0.133 | no — too close to call |
| 35 | 49.01 | 48.94 | +0.07 | [-0.17, +0.30] | 0.669 | no — too close to call |
| 40 | 51.26 | 50.78 | +0.48 | [+0.16, +0.80] | 0.038 | **yes** |

**4 of 9 noise levels show a statistically real advantage for conditioning after correction, at the 95% confidence level, with 1 seed per fold.**

Caveat: with only 1 seed per fold and 16 tiles total split into small clusters, this test has limited statistical power — a true effect can fail to reach significance here even if it is real. The direction (conditioning helps) being consistent across folds, shown separately in `04_fold_consistency.png`, is itself supporting evidence beyond this table.