# Seed-count variance analysis for H1 (methodology-section material)

Seeds present: [0, 1, 2]. Gaussian noise, PSNR, region=all -- same slice as `scripts/compare_arms.py`. This does not launch any new runs; it only re-reads the 3 seeds/fold already collected for the full H1 protocol.

## 1-2. Within-tile seed variance vs between-tile variance

Within-tile seed variance: for a fixed tile and SNR, how much does PSNR move across the 3 seeds (same model, same data, only the random init/data order differs). Between-tile variance: for a fixed SNR, how much does the seed-averaged PSNR move across the 16 different tiles (different scenes). If within-tile variance is small relative to between-tile variance, the choice of 16 tiles -- not the seed count -- is what actually drives uncertainty in this experiment.

| SNR (dB) | arm | within-tile seed var | between-tile var | ratio (within/between) |
|---|---|---|---|---|
| 0 | full_medium | 0.0095 | 6.1719 | 0.002 |
| 0 | e1_blind | 0.0125 | 5.7047 | 0.002 |
| 5 | full_medium | 0.0027 | 5.6854 | 0.000 |
| 5 | e1_blind | 0.0058 | 5.5984 | 0.001 |
| 10 | full_medium | 0.0017 | 5.6409 | 0.000 |
| 10 | e1_blind | 0.0069 | 5.5635 | 0.001 |
| 15 | full_medium | 0.0023 | 5.6482 | 0.000 |
| 15 | e1_blind | 0.0075 | 5.6362 | 0.001 |
| 20 | full_medium | 0.0044 | 5.7535 | 0.001 |
| 20 | e1_blind | 0.0078 | 5.8225 | 0.001 |
| 25 | full_medium | 0.0119 | 5.8543 | 0.002 |
| 25 | e1_blind | 0.0164 | 5.8626 | 0.003 |
| 30 | full_medium | 0.0206 | 5.7435 | 0.004 |
| 30 | e1_blind | 0.0563 | 5.5821 | 0.010 |
| 35 | full_medium | 0.0224 | 5.3778 | 0.004 |
| 35 | e1_blind | 0.2030 | 5.1308 | 0.040 |
| 40 | full_medium | 0.0375 | 4.6707 | 0.008 |
| 40 | e1_blind | 0.5198 | 4.7859 | 0.109 |

**full_medium: mean within-tile seed variance = 0.0126 dB^2, mean between-tile variance = 5.6162 dB^2 -- between-tile variance is 447.5x larger.**  
**e1_blind: mean within-tile seed variance = 0.0929 dB^2, mean between-tile variance = 5.5208 dB^2 -- between-tile variance is 59.4x larger.**  

## 3. Does the H1 gap estimate change from 1 seed to 2 seeds to 3 seeds?

For each SNR level: the conditioned-minus-blind gap (mean over the 16 tiles) computed three ways -- using only seed 0, averaging seeds 0-1, and averaging all 3 seeds (the final, reported H1 number). If the 1-seed and 3-seed columns are close, a single seed would have given essentially the same answer here; if they differ a lot, the extra seeds were doing real work.

| SNR (dB) | gap (seed 0 only) | gap (seeds 0-1 avg) | gap (seeds 0-2 avg, final) | move 1-seed vs final | move 2-seed vs final |
|---|---|---|---|---|---|
| 0 | +0.293 | +0.258 | +0.288 | +0.005 | -0.029 |
| 5 | +0.249 | +0.234 | +0.248 | +0.001 | -0.014 |
| 10 | +0.090 | +0.118 | +0.122 | -0.032 | -0.003 |
| 15 | -0.057 | -0.004 | -0.003 | -0.054 | -0.002 |
| 20 | -0.158 | -0.125 | -0.120 | -0.038 | -0.005 |
| 25 | -0.195 | -0.205 | -0.196 | +0.001 | -0.009 |
| 30 | -0.145 | -0.204 | -0.160 | +0.014 | -0.044 |
| 35 | +0.069 | -0.037 | +0.111 | -0.041 | -0.148 |
| 40 | +0.481 | +0.340 | +0.652 | -0.171 | -0.312 |

**Mean |1-seed minus final| = 0.040 dB. Mean |2-seed minus final| = 0.063 dB.**

## Verdict

The 2-seed average still moved 0.063 dB on average when the third seed was added -- seed variance is not negligible at the scale of this experiment's effects, so collecting 3 seeds (rather than 1-2) was the right call and should not be reduced in any follow-up without re-checking this.

This is reported as a methodology-section robustness check, not as grounds to redo H1 with a different seed count -- the 3-seed, 4-fold protocol is already complete and frozen.