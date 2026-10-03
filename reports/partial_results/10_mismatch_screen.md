# SNR-mismatch screen (E4): does the model actually use the conditioning signal?

For each fold's first test tile, the already-trained `full_medium` checkpoint is evaluated at the TRUE SNR (offset 0, from the existing H1 run) and again after being told an SNR that is off by +/-5 and +/-10 dB (same noisy input each time -- only what the model is TOLD changes). If conditioning is actually used, PSNR should drop as the lie gets bigger, and should drop in both directions (telling it 'cleaner than it is' and 'noisier than it is' should both hurt, just not necessarily symmetrically).

| Fold | Tile | SNR (dB) | -10dB | -5dB | 0 (true) | +5dB | +10dB | Monotonic around 0? |
|---|---|---|---|---|---|---|---|---|
| 0 | brest1 | 0 | 26.68 | 30.04 | **31.36** | 29.35 | 26.25 | yes |
| 0 | brest1 | 20 | 39.77 | 41.75 | **42.61** | 41.97 | 40.66 | yes |
| 0 | brest1 | 40 | 51.27 | 52.40 | **52.57** | 52.05 | 51.17 | yes |
| 1 | portsmouth | 0 | 27.90 | 32.39 | **34.73** | 33.71 | 30.61 | yes |
| 1 | portsmouth | 20 | 42.68 | 44.54 | **45.75** | 45.30 | 43.39 | yes |
| 1 | portsmouth | 40 | 53.96 | 55.60 | **55.59** | 53.95 | 51.80 | yes |
| 2 | suez1 | 0 | 24.20 | 26.41 | **27.30** | 24.74 | 21.88 | yes |
| 2 | suez1 | 20 | 34.70 | 36.26 | **37.57** | 37.48 | 35.94 | yes |
| 2 | suez1 | 40 | 45.35 | 47.64 | **48.57** | 48.50 | 47.84 | yes |
| 3 | panama | 0 | 29.73 | 31.63 | **32.47** | 31.07 | 28.36 | yes |
| 3 | panama | 20 | 38.59 | 40.12 | **41.73** | 42.45 | 41.68 | no |
| 3 | panama | 40 | 47.48 | 49.27 | **50.22** | 50.46 | 50.34 | no |

## Verdict

**Not every row is monotonic.** Some mismatch directions/magnitudes do not degrade PSNR as expected. This does not necessarily mean conditioning is broken -- single-seed, single-tile-per-fold screening has real noise -- but it means the 'conditioning clearly works' claim should be qualified, not stated flatly, until checked against more tiles/seeds.

**Caveat:** this is a screening pass (1 tile per fold, 1 seed, 3 SNR levels, gaussian noise only), not the full protocol grid -- scoped this way so it would finish in hours rather than days. Treat it as supporting evidence that the conditioning signal is used, not as a precise quantification of the mismatch penalty.