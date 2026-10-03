# H2 result: does cross-band attention improve spectral fidelity more than PSNR?

Comparing `full_medium` (with cross-band attention) vs `e2_no_crossband` (parameter-matched, attention block removed). SAM and ERGAS are lower-is-better; their 'gap' and CI below are sign-flipped so a positive number always means cross-band attention is better, matching PSNR's convention. SAM (degrees), ERGAS (dimensionless) and PSNR (dB) are reported as three separate tests, not combined into one 'relative improvement' statistic -- their units aren't comparable.

**Power caveat while partial:** with only 8 of 16 tiles currently available for the no-crossband arm, the exact Wilcoxon test's best possible raw p-value is about 0.0078 (all 8 tiles agreeing), which Holm-inflates to about 0.07 across 9 SNR levels -- structurally just above the 0.05 line even for a perfectly consistent effect. Several rows below show large, consistently-signed gaps that are not yet 'significant' for this reason, not because the effect is weak. Re-run once all 16 tiles are in.

## sam / gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 6.606 | 8.007 | +1.401 | [+0.965, +1.837] | 0.070 | no |
| 5 | 4.405 | 4.967 | +0.562 | [+0.377, +0.748] | 0.070 | no |
| 10 | 3.296 | 3.566 | +0.270 | [+0.126, +0.413] | 0.070 | no |
| 15 | 2.593 | 2.712 | +0.120 | [+0.002, +0.238] | 0.156 | no |
| 20 | 2.071 | 2.121 | +0.050 | [-0.053, +0.152] | 0.500 | no |
| 25 | 1.645 | 1.678 | +0.033 | [-0.051, +0.117] | 0.500 | no |
| 30 | 1.277 | 1.326 | +0.048 | [-0.016, +0.112] | 0.164 | no |
| 35 | 0.963 | 1.045 | +0.082 | [+0.014, +0.149] | 0.070 | no |
| 40 | 0.716 | 0.858 | +0.142 | [+0.011, +0.274] | 0.070 | no |

**0 of 9 SNR levels significant for sam/gaussian.**

## sam / correlated_gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 23.922 | 23.548 | -0.374 | [-1.609, +0.861] | 1.000 | no |
| 5 | 14.826 | 14.575 | -0.252 | [-1.270, +0.767] | 1.000 | no |
| 10 | 9.006 | 8.947 | -0.058 | [-0.706, +0.590] | 1.000 | no |
| 15 | 5.626 | 5.641 | +0.015 | [-0.329, +0.359] | 1.000 | no |
| 20 | 3.756 | 3.792 | +0.035 | [-0.143, +0.214] | 1.000 | no |
| 25 | 2.685 | 2.742 | +0.057 | [-0.069, +0.182] | 1.000 | no |
| 30 | 2.007 | 2.083 | +0.076 | [-0.037, +0.188] | 0.164 | no |
| 35 | 1.498 | 1.580 | +0.082 | [-0.014, +0.177] | 0.070 | no |
| 40 | 1.089 | 1.180 | +0.091 | [-0.000, +0.182] | 0.070 | no |

**0 of 9 SNR levels significant for sam/correlated_gaussian.**

## ergas / gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 12.021 | 13.601 | +1.580 | [+1.139, +2.022] | 0.070 | no |
| 5 | 8.492 | 9.092 | +0.600 | [+0.478, +0.721] | 0.070 | no |
| 10 | 6.312 | 6.577 | +0.265 | [+0.160, +0.370] | 0.070 | no |
| 15 | 4.767 | 4.905 | +0.138 | [+0.014, +0.263] | 0.070 | no |
| 20 | 3.569 | 3.692 | +0.123 | [-0.018, +0.265] | 0.070 | no |
| 25 | 2.607 | 2.770 | +0.163 | [+0.039, +0.287] | 0.070 | no |
| 30 | 1.864 | 2.076 | +0.212 | [+0.118, +0.305] | 0.070 | no |
| 35 | 1.338 | 1.591 | +0.253 | [+0.169, +0.336] | 0.070 | no |
| 40 | 1.015 | 1.307 | +0.292 | [+0.197, +0.387] | 0.070 | no |

**0 of 9 SNR levels significant for ergas/gaussian.**

## ergas / correlated_gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (higher=better, sign-flipped) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 36.174 | 35.542 | -0.632 | [-1.900, +0.636] | 0.781 | no |
| 5 | 20.929 | 20.499 | -0.430 | [-1.188, +0.329] | 0.547 | no |
| 10 | 12.819 | 12.664 | -0.155 | [-0.553, +0.242] | 0.781 | no |
| 15 | 8.335 | 8.315 | -0.020 | [-0.233, +0.194] | 0.945 | no |
| 20 | 5.645 | 5.678 | +0.033 | [-0.080, +0.147] | 0.922 | no |
| 25 | 3.877 | 3.951 | +0.074 | [-0.008, +0.156] | 0.141 | no |
| 30 | 2.651 | 2.773 | +0.121 | [+0.052, +0.191] | 0.070 | no |
| 35 | 1.812 | 1.978 | +0.167 | [+0.099, +0.234] | 0.070 | no |
| 40 | 1.277 | 1.495 | +0.218 | [+0.136, +0.299] | 0.070 | no |

**0 of 9 SNR levels significant for ergas/correlated_gaussian.**

## psnr / gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 31.674 | 30.526 | +1.148 | [+0.938, +1.357] | 0.070 | no |
| 5 | 34.833 | 34.138 | +0.695 | [+0.478, +0.912] | 0.070 | no |
| 10 | 37.487 | 37.044 | +0.443 | [+0.243, +0.642] | 0.070 | no |
| 15 | 39.930 | 39.631 | +0.300 | [+0.088, +0.512] | 0.070 | no |
| 20 | 42.411 | 42.097 | +0.314 | [+0.048, +0.580] | 0.070 | no |
| 25 | 45.077 | 44.553 | +0.524 | [+0.230, +0.817] | 0.070 | no |
| 30 | 47.899 | 46.987 | +0.912 | [+0.662, +1.163] | 0.070 | no |
| 35 | 50.670 | 49.223 | +1.448 | [+1.201, +1.694] | 0.070 | no |
| 40 | 52.984 | 50.892 | +2.092 | [+1.702, +2.481] | 0.070 | no |

**0 of 9 SNR levels significant for psnr/gaussian.**

## psnr / correlated_gaussian (PARTIAL: full-model tiles=16/16 seeds/tile=[3], no-crossband tiles=8/16 seeds/tile=[2, 3])

| SNR (dB) | With cross-band | Without | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 22.771 | 22.666 | +0.105 | [-0.057, +0.267] | 0.391 | no |
| 5 | 27.265 | 27.258 | +0.007 | [-0.198, +0.212] | 1.000 | no |
| 10 | 31.298 | 31.284 | +0.014 | [-0.184, +0.212] | 1.000 | no |
| 15 | 34.853 | 34.810 | +0.043 | [-0.133, +0.219] | 1.000 | no |
| 20 | 38.058 | 37.976 | +0.082 | [-0.069, +0.232] | 0.781 | no |
| 25 | 41.151 | 40.974 | +0.177 | [+0.004, +0.350] | 0.094 | no |
| 30 | 44.328 | 43.946 | +0.382 | [+0.192, +0.572] | 0.070 | no |
| 35 | 47.576 | 46.852 | +0.725 | [+0.535, +0.915] | 0.070 | no |
| 40 | 50.635 | 49.369 | +1.266 | [+0.998, +1.533] | 0.070 | no |

**0 of 9 SNR levels significant for psnr/correlated_gaussian.**

## H2 verdict

**PARTIAL -- not all folds/seeds are in yet.** The table above is a progress signal from whatever data currently exists, not the final H2 result. Re-run this script as more of the 12 full-model and 12 no-crossband runs land.