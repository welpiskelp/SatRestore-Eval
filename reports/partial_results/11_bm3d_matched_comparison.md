# BM3D vs conditioned vs blind, matched protocol (apples-to-apples)

Tiles with all 9 SNR levels in the matched BM3D run so far: ['brest1', 'marseille', 'panama', 'portsmouth', 'rotterdam1', 'rotterdam2', 'rotterdam3', 'toulon'] (8/16). Partial tiles excluded. Gaussian noise, PSNR, region=all.

| SNR (dB) | BM3D | Conditioned (full_medium) | Blind (e1_blind) | BM3D vs Conditioned | Winner |
|---|---|---|---|---|---|
| 0 | 29.06 | 31.37 | 31.04 | -2.31 | conditioned |
| 5 | 32.19 | 34.37 | 34.12 | -2.17 | conditioned |
| 10 | 35.40 | 36.91 | 36.80 | -1.51 | conditioned |
| 15 | 38.67 | 39.31 | 39.34 | -0.65 | conditioned |
| 20 | 41.94 | 41.79 | 41.96 | +0.15 | BM3D |
| 25 | 45.24 | 44.48 | 44.73 | +0.76 | BM3D |
| 30 | 48.60 | 47.30 | 47.49 | +1.30 | BM3D |
| 35 | 52.10 | 50.02 | 49.86 | +2.08 | BM3D |
| 40 | 55.79 | 52.23 | 51.48 | +3.56 | BM3D |

**Crossover point(s): 20 dB** -- BM3D and the conditioned model swap which one wins around here.

Caveat: only 8/16 tiles so far (more BM3D batches still running); this table will be regenerated as more tiles land. These tiles are not a random sample of the 16 -- rotterdam1/2/3 are the overlap-group trio and toulon has the most ships -- so don't treat this as the final 16-tile answer yet.