# H1 result across all three noise types (not just gaussian)

`scripts/compare_arms.py` only ever looked at gaussian noise, even though every checkpoint was already evaluated against gaussian, poisson_gaussian, and correlated_gaussian by `scripts/evaluate_run.py` (its defaults cover all three). No new training or evaluation runs were needed for this -- the data was already in `db/results_pilot_full.sqlite`, it just had never been reported this way.

## gaussian (FINAL: cond seeds/fold [3], blind seeds/fold [3])

| SNR (dB) | Conditioned | Blind | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 30.24 | 29.95 | +0.29 | [+0.16, +0.42] | 0.001 | **yes** |
| 5 | 33.35 | 33.10 | +0.25 | [+0.17, +0.33] | 0.001 | **yes** |
| 10 | 35.96 | 35.84 | +0.12 | [+0.06, +0.18] | 0.006 | **yes** |
| 15 | 38.39 | 38.39 | -0.00 | [-0.08, +0.07] | 1.000 | no |
| 20 | 40.85 | 40.97 | -0.12 | [-0.23, -0.01] | 0.086 | no |
| 25 | 43.47 | 43.66 | -0.20 | [-0.34, -0.05] | 0.065 | no |
| 30 | 46.22 | 46.38 | -0.16 | [-0.35, +0.03] | 0.173 | no |
| 35 | 48.91 | 48.80 | +0.11 | [-0.15, +0.37] | 1.000 | no |
| 40 | 51.19 | 50.54 | +0.65 | [+0.36, +0.95] | 0.001 | **yes** |

**4 of 9 noise levels significant for gaussian.**

## poisson_gaussian (FINAL: cond seeds/fold [3], blind seeds/fold [3])

| SNR (dB) | Conditioned | Blind | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 29.67 | 29.57 | +0.10 | [-0.02, +0.23] | 0.173 | no |
| 5 | 32.88 | 32.72 | +0.17 | [+0.09, +0.24] | 0.002 | **yes** |
| 10 | 35.56 | 35.47 | +0.09 | [+0.02, +0.15] | 0.150 | no |
| 15 | 38.02 | 38.03 | -0.01 | [-0.08, +0.07] | 0.991 | no |
| 20 | 40.47 | 40.60 | -0.13 | [-0.24, -0.02] | 0.150 | no |
| 25 | 43.08 | 43.31 | -0.23 | [-0.38, -0.08] | 0.036 | no, favors blind |
| 30 | 45.87 | 46.10 | -0.23 | [-0.42, -0.03] | 0.150 | no |
| 35 | 48.66 | 48.64 | +0.02 | [-0.24, +0.28] | 0.991 | no |
| 40 | 51.05 | 50.46 | +0.59 | [+0.29, +0.88] | 0.001 | **yes** |

**2 of 9 noise levels significant for poisson_gaussian.**

## correlated_gaussian (FINAL: cond seeds/fold [3], blind seeds/fold [3])

| SNR (dB) | Conditioned | Blind | Gap (dB) | 95% range | Holm p | Real effect? |
|---|---|---|---|---|---|---|
| 0 | 21.13 | 21.50 | -0.37 | [-0.49, -0.24] | 0.000 | no, favors blind |
| 5 | 25.60 | 25.80 | -0.20 | [-0.35, -0.04] | 0.018 | no, favors blind |
| 10 | 29.62 | 29.52 | +0.10 | [-0.06, +0.25] | 0.130 | no |
| 15 | 33.18 | 32.78 | +0.40 | [+0.25, +0.56] | 0.001 | **yes** |
| 20 | 36.37 | 35.82 | +0.56 | [+0.41, +0.71] | 0.000 | **yes** |
| 25 | 39.45 | 38.95 | +0.50 | [+0.37, +0.63] | 0.000 | **yes** |
| 30 | 42.62 | 42.28 | +0.34 | [+0.24, +0.44] | 0.000 | **yes** |
| 35 | 45.87 | 45.64 | +0.23 | [+0.07, +0.39] | 0.010 | **yes** |
| 40 | 48.96 | 48.55 | +0.40 | [+0.15, +0.66] | 0.005 | **yes** |

**6 of 9 noise levels significant for correlated_gaussian.**

## Cross-noise-type summary

| Noise type | Status | Significant SNR levels |
|---|---|---|
| gaussian | FINAL | 4/9 |
| poisson_gaussian | FINAL | 2/9 |
| correlated_gaussian | FINAL | 6/9 |

If the same extremes-significant / middle-null pattern holds across all three noise types, that is evidence the H1 effect is about degradation severity generally, not an artifact of the gaussian noise model specifically. If it does not hold, that is itself worth reporting rather than hiding.