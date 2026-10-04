# BM3D vs conditioned vs blind, matched protocol (apples-to-apples)

Tiles with all 9 SNR levels in the matched BM3D run so far: ['rotterdam1', 'rotterdam2', 'rotterdam3', 'toulon'] (4/16). Partial tiles excluded. Gaussian noise, PSNR, region=all.

| SNR (dB) | BM3D | Conditioned (full_medium) | Blind (e1_blind) | BM3D vs Conditioned | Winner |
|---|---|---|---|---|---|
| 0 | 28.43 | 30.57 | 30.12 | -2.14 | conditioned |
| 5 | 31.54 | 33.59 | 33.26 | -2.05 | conditioned |
| 10 | 34.75 | 36.09 | 35.95 | -1.34 | conditioned |
| 15 | 38.03 | 38.44 | 38.51 | -0.41 | conditioned |
| 20 | 41.36 | 40.89 | 41.14 | +0.47 | BM3D |
| 25 | 44.72 | 43.61 | 43.97 | +1.11 | BM3D |
| 30 | 48.14 | 46.57 | 46.88 | +1.57 | BM3D |
| 35 | 51.64 | 49.50 | 49.48 | +2.15 | BM3D |
| 40 | 55.30 | 52.01 | 51.36 | +3.29 | BM3D |

**Crossover point(s): 20 dB** -- BM3D and the conditioned model swap which one wins around here.

Caveat: only 4/16 tiles so far (more BM3D batches still running); this table will be regenerated as more tiles land. These 4 tiles are not a random sample -- rotterdam1/2/3 are the overlap-group trio and toulon has the most ships, so don't treat this as the final 16-tile answer yet.