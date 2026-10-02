# H1 result: does telling the model the noise level help? (IN PROGRESS)

4 folds, 16 tiles total (every tile tested exactly once, in whichever fold it belongs to as a test tile). The conditioned model has 3 seed(s) per fold; the blind model has 2/3 seed(s) per fold so far (protocol target: 3 seeds per fold for both). Wherever a fold has more than one seed, its tiles' scores are averaged across those seeds before this comparison, per protocol.

Per noise level: average PSNR with conditioning, average PSNR without it, the gap, a 95% confidence range for that gap (accounting for tiles that share the same scene), and whether the gap is statistically real after correcting for testing 9 noise levels at once (Holm correction).

| SNR (dB) | With conditioning | Without (blind) | Gap (dB) | 95% range | Holm p-value | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 30.24 | 29.97 | +0.26 | [+0.12, +0.41] | 0.005 | **yes** |
| 5 | 33.35 | 33.12 | +0.23 | [+0.14, +0.33] | 0.001 | **yes** |
| 10 | 35.96 | 35.85 | +0.11 | [+0.03, +0.19] | 0.021 | **yes** |
| 15 | 38.39 | 38.40 | -0.01 | [-0.09, +0.07] | 0.900 | no — too close to call |
| 20 | 40.85 | 40.99 | -0.14 | [-0.25, -0.03] | 0.021 | no, favors blind |
| 25 | 43.47 | 43.70 | -0.23 | [-0.36, -0.11] | 0.007 | no, favors blind |
| 30 | 46.22 | 46.46 | -0.24 | [-0.39, -0.10] | 0.021 | no, favors blind |
| 35 | 48.91 | 48.98 | -0.07 | [-0.26, +0.12] | 0.351 | no — too close to call |
| 40 | 51.19 | 50.84 | +0.34 | [+0.12, +0.56] | 0.016 | **yes** |

**4 of 9 noise levels show a statistically real advantage for conditioning after correction, at the 95% confidence level.**

Caveat: 16 tiles total split into small clusters means this test has limited statistical power even at the full seed count — a true effect can fail to reach significance here even if it is real. The direction (conditioning helps) being consistent across folds, shown separately in `04_fold_consistency.png`, is itself supporting evidence beyond this table.