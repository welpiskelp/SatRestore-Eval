# H2 result: does cross-band attention improve spectral fidelity more than PSNR?

Comparing `full_medium` (with cross-band attention) vs `e2_no_crossband` (parameter-matched, attention block removed). SAM and ERGAS are lower-is-better; their 'gap' and CI below are sign-flipped so a positive number always means cross-band attention is better, matching PSNR's convention. SAM (degrees), ERGAS (dimensionless) and PSNR (dB) are reported as three separate tests, not combined into one 'relative improvement' statistic -- their units aren't comparable.

**Power caveat while partial:** with only 8 of 16 tiles currently available for the no-crossband arm, the exact Wilcoxon test's best possible raw p-value is about 0.0078 (all 8 tiles agreeing), which Holm-inflates to about 0.07 across 9 SNR levels -- structurally just above the 0.05 line even for a perfectly consistent effect. Several rows below show large, consistently-signed gaps that are not yet 'significant' for this reason, not because the effect is weak. Re-run once all 16 tiles are in.

## sam / gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 5.922 | 7.070 | +1.148 | [+0.855, +1.440] | 0.000 | **yes** |
| 5 | 4.026 | 4.521 | +0.495 | [+0.363, +0.627] | 0.000 | **yes** |
| 10 | 3.055 | 3.290 | +0.235 | [+0.152, +0.318] | 0.000 | **yes** |
| 15 | 2.422 | 2.531 | +0.109 | [+0.048, +0.171] | 0.001 | **yes** |
| 20 | 1.940 | 1.996 | +0.056 | [+0.004, +0.108] | 0.022 | **yes** |
| 25 | 1.535 | 1.582 | +0.047 | [+0.002, +0.092] | 0.022 | **yes** |
| 30 | 1.183 | 1.247 | +0.063 | [+0.026, +0.101] | 0.002 | **yes** |
| 35 | 0.888 | 0.975 | +0.087 | [+0.052, +0.122] | 0.000 | **yes** |
| 40 | 0.660 | 0.782 | +0.122 | [+0.065, +0.180] | 0.000 | **yes** |

**9 of 9 SNR levels significant for sam/gaussian.**

## sam / correlated_gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 22.293 | 21.679 | -0.614 | [-1.122, -0.106] | 0.046 | no, favors no-crossband |
| 5 | 13.581 | 13.354 | -0.228 | [-0.654, +0.199] | 0.701 | no |
| 10 | 8.200 | 8.182 | -0.018 | [-0.283, +0.246] | 1.000 | no |
| 15 | 5.171 | 5.206 | +0.036 | [-0.098, +0.169] | 1.000 | no |
| 20 | 3.488 | 3.527 | +0.038 | [-0.030, +0.107] | 0.701 | no |
| 25 | 2.508 | 2.555 | +0.047 | [-0.005, +0.099] | 0.145 | no |
| 30 | 1.861 | 1.926 | +0.065 | [+0.015, +0.114] | 0.003 | **yes** |
| 35 | 1.368 | 1.442 | +0.074 | [+0.030, +0.118] | 0.000 | **yes** |
| 40 | 0.982 | 1.063 | +0.081 | [+0.039, +0.122] | 0.000 | **yes** |

**3 of 9 SNR levels significant for sam/correlated_gaussian.**

## ergas / gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 11.741 | 13.104 | +1.363 | [+1.151, +1.574] | 0.000 | **yes** |
| 5 | 8.359 | 8.904 | +0.546 | [+0.469, +0.622] | 0.000 | **yes** |
| 10 | 6.246 | 6.497 | +0.251 | [+0.194, +0.309] | 0.000 | **yes** |
| 15 | 4.726 | 4.876 | +0.150 | [+0.081, +0.219] | 0.000 | **yes** |
| 20 | 3.542 | 3.689 | +0.148 | [+0.063, +0.233] | 0.000 | **yes** |
| 25 | 2.598 | 2.779 | +0.181 | [+0.100, +0.263] | 0.000 | **yes** |
| 30 | 1.874 | 2.088 | +0.214 | [+0.146, +0.281] | 0.000 | **yes** |
| 35 | 1.362 | 1.597 | +0.234 | [+0.174, +0.295] | 0.000 | **yes** |
| 40 | 1.042 | 1.298 | +0.255 | [+0.190, +0.320] | 0.000 | **yes** |

**9 of 9 SNR levels significant for ergas/gaussian.**

## ergas / correlated_gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 35.510 | 34.725 | -0.785 | [-1.340, -0.230] | 0.011 | no, favors no-crossband |
| 5 | 20.648 | 20.244 | -0.405 | [-0.733, -0.076] | 0.044 | no, favors no-crossband |
| 10 | 12.697 | 12.585 | -0.113 | [-0.281, +0.056] | 0.187 | no |
| 15 | 8.270 | 8.287 | +0.017 | [-0.072, +0.106] | 0.669 | no |
| 20 | 5.616 | 5.675 | +0.058 | [+0.005, +0.112] | 0.044 | **yes** |
| 25 | 3.871 | 3.959 | +0.088 | [+0.040, +0.137] | 0.000 | **yes** |
| 30 | 2.653 | 2.777 | +0.124 | [+0.075, +0.174] | 0.000 | **yes** |
| 35 | 1.816 | 1.972 | +0.156 | [+0.103, +0.209] | 0.000 | **yes** |
| 40 | 1.283 | 1.476 | +0.193 | [+0.134, +0.253] | 0.000 | **yes** |

**5 of 9 SNR levels significant for ergas/correlated_gaussian.**

## psnr / gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 30.238 | 29.189 | +1.049 | [+0.924, +1.174] | 0.000 | **yes** |
| 5 | 33.349 | 32.699 | +0.649 | [+0.545, +0.754] | 0.000 | **yes** |
| 10 | 35.960 | 35.547 | +0.414 | [+0.317, +0.510] | 0.000 | **yes** |
| 15 | 38.387 | 38.091 | +0.296 | [+0.192, +0.400] | 0.000 | **yes** |
| 20 | 40.848 | 40.515 | +0.333 | [+0.197, +0.468] | 0.000 | **yes** |
| 25 | 43.468 | 42.938 | +0.530 | [+0.373, +0.687] | 0.000 | **yes** |
| 30 | 46.220 | 45.358 | +0.862 | [+0.709, +1.015] | 0.000 | **yes** |
| 35 | 48.907 | 47.628 | +1.279 | [+1.089, +1.468] | 0.000 | **yes** |
| 40 | 51.187 | 49.418 | +1.769 | [+1.464, +2.075] | 0.000 | **yes** |

**9 of 9 SNR levels significant for psnr/gaussian.**

## psnr / correlated_gaussian (FINAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=16/16 seeds/tile=[3])

| SNR (dB) | With cross-band | Without | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 21.135 | 21.160 | -0.025 | [-0.142, +0.092] | 1.000 | no |
| 5 | 25.603 | 25.652 | -0.049 | [-0.158, +0.061] | 0.968 | no |
| 10 | 29.622 | 29.618 | +0.004 | [-0.083, +0.091] | 1.000 | no |
| 15 | 33.179 | 33.120 | +0.059 | [-0.012, +0.130] | 0.073 | no |
| 20 | 36.375 | 36.269 | +0.106 | [+0.037, +0.174] | 0.004 | **yes** |
| 25 | 39.447 | 39.249 | +0.197 | [+0.107, +0.287] | 0.000 | **yes** |
| 30 | 42.616 | 42.239 | +0.377 | [+0.269, +0.485] | 0.000 | **yes** |
| 35 | 45.874 | 45.219 | +0.655 | [+0.518, +0.792] | 0.000 | **yes** |
| 40 | 48.958 | 47.865 | +1.093 | [+0.882, +1.304] | 0.000 | **yes** |

**5 of 9 SNR levels significant for psnr/correlated_gaussian.**

## H2 verdict

All 4 folds x 3 seeds present for both arms -- this is the final H2 result. H2 is supported if SAM and/or ERGAS show more/stronger significant SNR levels than PSNR does; it is not supported if PSNR shows an equal or stronger pattern.