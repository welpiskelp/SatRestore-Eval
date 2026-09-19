# Progress Log

**Project:** SNR-Conditioned, Spectrally Aware, Ship-Preserving Reconstruction of Sentinel-2 Maritime Imagery under Realistic Noise
**Repo:** https://github.com/welpiskelp/SatRestore-Eval (branch `Main`)
**Last updated:** 2026-09-19

---

## Status at a glance

| Phase | Status |
|---|---|
| 1. Local setup, EOTDL download, repo scaffold, audit | DONE (committed locally; GitHub push blocked on authentication) |
| 2a. Tile preparation (pickle-free canonical arrays, aligned masks) | DONE (committed locally) |
| 2b. Overlap-aware tile-level fold splits (4 folds, 10/2/4) | DONE (not yet committed) |
| 2c. Patch index (128×128, stride 64) with per-patch ship/water stats | DONE (not yet committed); edge-aligned patches added, see step 11 |
| 3. Degradation engine (Gaussian / Poisson-Gaussian, SNR sampling) | DONE (not yet committed); verified on real tiles, see step 12 |
| 4. SQLite reference DB | NOT STARTED |
| 5. Models (NAFNet+FiLM+cross-band, baselines) | NOT STARTED |
| 6. Training / experiments / ablations | NOT STARTED |
| 7. Kaggle GPU setup + 3-person split | NOT STARTED (deferred until training code exists) |

---

## Finished steps

### 1. Environment
- Local Python 3.13.9, project virtual environment created at `C:\SAR\.venv`.
- Installed: `eotdl==2026.6.29`, `rasterio==1.5.1`, `numpy==2.5.3` (pinned in `requirements.txt`).

### 2. EOTDL auth + download
- Authenticated via `eotdl auth login` (already valid session — logged in as `abhipotharaju10@gmail.com`).
- **Important finding:** `eotdl datasets get` (the CLI's built-in downloader) buffers the *entire* file into memory with no streaming and no progress output before writing to disk — for this dataset's 6.2GB zip asset, that looked like a hang. Wrote a custom `scripts/download_dataset.py` instead, which:
  - Resolves the presigned URL the same way the CLI does internally (reuses `eotdl`'s own auth/repo code).
  - Streams the download in 8MB chunks with a progress bar, supports resuming a partial download.
  - Verifies the SHA1 checksum from the STAC catalog after download (**verified match**: `53d83f86e8fcfe7b3d363228d343e3cd117647ef`).
  - Extracts the zip automatically.
- Downloaded asset: `S2-SHIPS.zip`, 6.22GB, from EOTDL's STAC-based `S2-SHIPS` dataset (1 STAC item, 1 bundled asset).

### 3. Data cleanup
The zip contained both an already-extracted `S2SHIPS/` folder tree *and* a redundant `S2SHIPS.tar` re-packaging of nearly the same content. After inspecting both (byte-for-byte size comparison on overlapping folders) and merging in the two subfolders that existed **only** in the tar:

- Kept: `dataset_tif/` (740MB — 16 tiles × 12 raw bands + NDWI/true-color/previews), `dataset_npy/` (2.6GB — per-tile ship instance masks), `water_mask/` (26MB), `s2ships_labels_mask/` (204KB), `coco-*.json` (ship annotations).
- Deleted: `S2-SHIPS.zip` (6.2GB, fully extracted + checksum-verified, no longer needed), `S2SHIPS.tar` (4.8GB, duplicate of kept content), `pretrained_backbones/` (1.9GB — SSL-pretrained weights from the *original* S2-SHIPS paper's own backbones; not used by our NAFNet-based model), nested `S2SHIPS.tar.xz` (5.6MB, redundant re-compression).
- **Result: 3.4GB kept locally**, down from ~16GB of raw/duplicate archives. `data/` is git-ignored — none of this is or will be pushed to GitHub.

### 4. Repo scaffold
- Connected `C:\SAR` to the existing (near-empty) GitHub repo `welpiskelp/SatRestore-Eval` as `origin`, branch renamed to `Main` to match.
- Created: `.gitignore` (excludes `.venv/`, `__pycache__/`, `*.pyc`, `/data/`), `README.md`, `requirements.txt`, `reports/` (audit output), and empty `src/{data,degrade,models,losses,train,eval}` package stubs (`__init__.py` only — no logic yet) matching the eventual layout from the project doc.
- `configs/` and `db/` intentionally **not** created yet — nothing to put in them until the preprocessing/DB phases.

### 5. Data audit (`scripts/audit.py`)
Verified real on-disk structure first (each tile = one directory under `dataset_tif/<tile>/` with 12 **single-band** GeoTIFFs, not one 12-band file; no SCL band or nodata metadata shipped, so cloud/nodata is only a rough pixel-saturation heuristic). Script checks per tile: band count/shape/dtype, per-band pixel stats, water-mask presence, ship-mask `.npy` presence, and COCO ship-annotation counts.

**Result: 16/16 tiles pass all checks** — all have 12/12 bands (uint16, shape 938×1783 as expected), a water mask, and a ship-mask `.npy`.

Tiles: `brest1, marseille, panama, portsmouth, rome, rotterdam1, rotterdam2, rotterdam3, southampton, suez1–6, toulon`.

**Notable finding:** the COCO file has 143 "annotation records," but each record bundles multiple ship polygons in its `segmentation` list. Counting individual polygons gives **1,053** — which matches the project doc's "~1,053 ship instances" exactly. So "ship instance" = segmentation polygon count, not annotation-record count. Full per-tile breakdown is in `reports/audit_report.json`.

### 6. Git
- Repo initialized, `origin` set to the GitHub repo, commits made locally on `Main` (code/config/report only — `data/` is never included).
- A push was attempted and failed: `Authentication failed` (no stored GitHub credentials; the `gh` CLI is not installed). **Nothing has been pushed yet.** GitHub account: `abhip-10`. Needs the user to set up credentials (e.g. Git Credential Manager browser login, or a personal access token entered by the user, not pasted into chat).

### 7. Tile preparation (`scripts/prepare_tiles.py`, `src/data/`)
Converts the raw download into compact, pickle-free arrays under `data/processed/` (664MB total, git-ignored; was 2.6GB of raw npy):
- `tiles/<tile>_image.npy` — uint16, shape (12, 938, 1783), channels in canonical band order.
- `tiles/<tile>_ship.npy` — uint8 ship-pixel mask. `tiles/<tile>_water.npy` — uint8 water mask, aligned to the imagery grid.
- `tiles_meta.json` — per tile: date, shape, CRS, bounds, ship pixel count, water fraction, fraction of ship pixels on water.

Shared code added: `src/data/bands.py` (single source of truth for band order, native-10m band list) and `src/data/tiles.py` (readers, including a guarded pickle loader). `scripts/audit.py` now imports these instead of duplicating them; re-run confirmed identical audit results (16/16 tiles, 1,053 polygons).

### 8. Findings that affect later design
- **`dataset_npy/*.npy` are pickled dicts**, not plain masks: `data` = (938, 1783, 12) float64 raw digital numbers (imagery), `label` = (938, 1783, 1) binary ship mask (0/1). Unpickling can execute code, so `load_pickled_npy` disassembles the pickle first and refuses anything except numpy array internals (all 16 files passed). The pickle is touched only once, by `prepare_tiles.py`; everything downstream reads plain arrays.
- **Band order mismatch:** the npy channel order is `B01,B02,B03,B04,B05,B06,B07,B08,B09,B11,B12,B8A` (B8A last). The project's canonical order is Sentinel-2 wavelength order (`B8A` after `B08`). `prepare_tiles.py` reorders and verifies every channel of every tile bit-for-bit against the per-band GeoTIFFs. Always index bands by name via `src/data/bands.py`.
- **Water masks are on a different grid:** 938×1784 at 10.000 m versus imagery 938×1783 at 10.0048 m (same origin and extent). Resampled with nearest-neighbour onto the imagery grid rather than cropped, because a crop would drift up to about 0.9 pixel at the right edge.
- **Tiles overlap spatially (threat to tile-level splits):** `suez1`/`suez2` overlap by 93.6% (near-duplicate scenes from the same day); `rotterdam1`/`rotterdam2` by 6.9%; `rotterdam2`/`rotterdam3` by 13.9%. The other tiles are independent. Overlapping tiles must never sit on opposite sides of a train/val/test split, or the test set leaks.
- **Ship distribution is very uneven:** `toulon` has 263 of the 1,053 ship polygons; `suez1`–`suez4` have 5–8 each. `rotterdam1` has the most ship pixels (10,957). Folds should be balanced on ship counts, not just tile counts.
- **Ship pixels mostly lie on the water mask** (94–100% for most tiles) but less for `rome` (68%) and `marseille` (81%), likely ships in port. Relevant when interpreting ship-vs-water metrics (E5).
- Water fraction varies widely (2% for `suez1`/`suez2` up to 84% for `portsmouth`).

### 9. Fold splits (`scripts/make_splits.py`, `src/data/splits.py`, `configs/splits.json`)
Tile-level 4-fold cross-validation: 10 train / 2 val / 4 test tiles per fold, every tile is a test tile exactly once.
- **Overlap groups are computed from the tile bounds, not hard-coded:** `{rotterdam1, rotterdam2, rotterdam3}` and `{suez1, suez2}`. A group is never split across train/val/test within a fold. Since the rotterdam trio needs 3 of the 4 test slots, it necessarily forms one test set together with one other tile.
- **Test sets:** all packings of whole groups into 4 equal test sets are searched exhaustively (17,325 candidates); the chosen one minimises the coefficient of variation of ship polygons plus that of ship pixels across the 4 test sets. Val pairs are then picked per fold to have about half a test set of ship polygons, spreading val use across tiles. The seed only orders the folds and breaks ties; the test partition itself is determined by the optimisation.
- **Result (seed 0):** test ship polygons per fold are 266 / 247 / 278 / 262 (well balanced). Test ship pixels are not balanced (6,319 / 17,876 / 3,934 / 6,211) because `rotterdam1` alone has 10,957 ship pixels and the rotterdam trio is forced into one test set (fold 1). Expect fold 1's test metrics to be dominated by rotterdam.
- **Verification:** `python scripts/make_splits.py --check` re-validates the saved file (each tile tested once, splits disjoint and complete, no overlap group split). A deliberately leaky split (suez1 in val, suez2 in train) is rejected by the validator.
- Scene-level similarity beyond spatial overlap is not controlled for. Non-overlapping neighbours: `portsmouth`/`southampton` are 4.4 km apart (same date, in different folds' test sets); `suez3`–`suez6` are 14–18 km apart (same date). No shared pixels, but shared scene statistics; state this as a limitation.

### 10. Risk validation (checked against the data on 2026-09-19)
Each risk raised earlier was tested. Verdict is whether it harms the paper's claims as things stand.

| Risk | Evidence | Harm now? | Required action |
|---|---|---|---|
| Saturated pixels (65535) | Only `panama` (10.8% of pixels any-band), `rotterdam2` (11.8%), `rotterdam3` (6.2%); saturation appears in every band B01–B8A, not only B09. 198 of 5,408 patches (3.7%) have >5% B09 saturation. **0 ship pixels are saturated; 0 of 1,228 ship-containing patches are affected.** But saturated pixels inflate per-tile signal power `P_b` by up to 3.2 dB (`panama` B09), about 1 dB in B04 (3 tiles), 0 dB in `toulon`. | No, not on ship claims. Yes if `P_b` is computed naively: nominal SNR would be off by 1–3 dB in three tiles. | Compute `P_b` excluding saturated pixels (record both). Store per-patch saturation fraction in the patch index. Decide a filter/flag rule for training patches (likely clouds; not visually confirmed). |
| Ship mask vs COCO polygons | Rasterised polygons vs pixel mask: overall IoU 0.97 (per tile 0.92–0.99). Mask has 34,340 px, polygons rasterise to 34,355. 1,044 mask connected components versus 1,053 polygons (touching ships merge). 4 polygons have no mask pixel. | No. | Use polygon count (1,053) for instance counts and bootstrap; use the pixel mask for loss weighting. |
| Water mask (`rome`, `marseille`) | Visual check. `rome`: the mask misses the river almost completely; the small boats moored along it are labelled as ships, so 210 of 222 off-water ship pixels are more than 5 px from any water pixel. `marseille`: mask is fine; ships moored at the quay sit beside the mask. `toulon`, `panama`, `brest1`, `rotterdam1` off-water pixels are all within 5 px of water. | No for main claims (RQ1-RQ3 do not need the water mask). Yes for E5 water-based metrics (water-mask IoU, NDWI error, ship-to-water contrast) on `rome` if the mask is used as is. | Do not use the delivered water mask for `rome` in E5; use a local-background definition, or report `rome` separately or exclude it from water-based metrics. |
| Overlap / leakage | Overlap groups are kept together in one split, so no shared pixels cross train/val/test. Scene-similarity across neighbours remains. | No, if stated as a limitation. | Disclose in threats to validity. |
| Fold imbalance | Test ship polygons balanced (247–278). Test ship pixels are not. In fold 1, `rotterdam1` alone is 122 of 247 test polygons (49%) and 61% of test ship pixels. | No, provided results are reported per tile and instance-level bootstrap is used. | Report per-tile results, not only per-fold means. |

Overall: nothing blocks the paper. Two items must be handled in later steps: `P_b` excluding saturated pixels (degradation engine) and not using the `rome` water mask for E5.

### 11. Patch index (`scripts/make_patch_index.py`, `src/data/patches.py`)
`data/processed/patch_index.csv` (git-ignored, regenerate with the script): one row per patch, 128×128 at stride 64 plus one edge-aligned row and column (`y = 810`, `x = 1655`), 14 × 27 = 378 patches per tile, **6,048 patches**, every pixel of every tile covered. Columns: `patch_id, tile, y, x, ship_pixels, ship_instances, water_frac, sat_any_frac, sat_b09_frac, zero_any_frac`. Imagery is not copied; patches are sliced from `data/processed/tiles/` at load time. `ship_instances` counts ship polygons whose centroid lies inside the patch, so a ship near a border can appear in several overlapping patches. `tiles.py` gained `load_ship_centroids`.
- **Verified:** patches compared against independent windowed GeoTIFF reads, all 12 bands identical (the (y = row, x = column) convention is right), including edge-aligned patches and the bottom-right corner patch. 6,048 unique patch ids, 378 per tile. Toulon coverage check: 1,672,454 of 1,672,454 pixels covered. 1,323 patches contain ship pixels; 243 have more than 5% saturated pixels; 99 contain zero-valued pixels.
- **Edge coverage decision (resolved: edge-aligned patches added).** With the plain stride-64 grid, 103 of 1,053 ship instances (9.8%) and 1,032 of 34,340 ship pixels fell outside every patch, concentrated in `toulon` (77 of its 263), `brest1` (10), `portsmouth` (8) and `rotterdam1` (6). That would have cut fold 2's test set from 278 to 201 ship instances and broken the fold balance. Now 0 of 1,053 instances and 0 of 34,340 ship pixels are outside every patch. Cost: edge patches overlap their neighbours more (106 of 128 rows or columns instead of 64), a minor training weighting effect to mention as a limitation.
- **Off-image ship:** one `portsmouth` polygon has its centroid at y = 938.3 on a 938-row tile (a ship hanging off the bottom edge). Centroids are clipped into the image (`clip_centroids`) so the ship belongs to the edge patches.
- **Overlap multiplicity:** the per-patch `ship_instances` column sums to 3,948 for 1,053 ships, so each ship appears in about 3.75 patches on average. Any per-instance statistic (E5, bootstrap CIs over ship instances) must count each ship once, for example by assigning it to the one patch that contains its centroid most centrally, and must not sum over patches.
- **Zero-valued pixels:** only in `rotterdam3` (99 patches with any zero pixel, worst 15.7% of a patch; 0.41% of the tile), scattered, only in bands B01, B06, B07 and B8A, never zero in all 12 bands, so they are invalid pixels rather than a nodata border. 50 ship pixels fall on such pixels. Treat zeros as invalid: exclude from `P_b`, and consider masking them in SAM and the loss.

### 12. Degradation engine (`src/degrade/`, `scripts/compute_signal_stats.py`, `scripts/check_degradation.py`)
- **SNR definition:** `SNR_b = 10·log10(P_b / N_b)`. `P_b` = mean square of band `b` over the tile's **valid pixels** (not 0, not saturated at 65535), computed once per tile and band (`data/processed/signal_stats.json`, git-ignored, regenerate with the script). `N_b` = average noise power added to the band. Mean square (not variance) is used, so the mean intensity dominates `P_b`.
- **Effect of excluding invalid pixels:** naive `P_b` would be inflated by up to 3.24 dB (`panama` B09), 2.85 dB (`rotterdam2` B09), 2.13 dB (`rotterdam3` B01); under 0.03 dB for the 13 clean tiles.
- **Noise types**, both scaled to the same `N_b` so a nominal SNR is comparable across them: `gaussian` (`y = x + n`, `n ~ N(0, N_b)`) and `poisson_gaussian` (`y = a·Poisson(x/a) + n`, gain `a = ρ·N_b / mean_b`, read noise variance `(1−ρ)·N_b`, so pixel variance `a·x + (1−ρ)·N_b` averages to `N_b`). `ρ` (`shot_fraction`) is **assumed 0.5, not calibrated to the sensor**; it is logged with every degradation and is the knob for E2/A9.
- **Design choices:** noise is added everywhere, including saturated and zero pixels (those pixels must be masked in metrics and loss later). Output is float32 in raw DN units and **not clipped**, so the realised SNR is not distorted at low signal (clipping would be a further assumption). SNR can be a scalar or per band (needed for the A5 conditioning ablation). `degrade()` returns the noisy patch plus a params dict (noise type, seed, per-band SNR, σ, gain, read σ, ρ) that maps directly onto the planned `degradations` table.
- **Seeding:** `degradation_seed(base_seed, patch_id, key)` uses SHA-256 (stable across processes and platforms, unlike Python's `hash`). Test levels (`TEST_SNR_LEVELS` = 0, 5, …, 30 dB) use `key = round(snr·100)` so every model sees the identical noisy patch (paired comparison); training draws SNR uniformly in dB from `TRAIN_SNR_RANGE` = 5–30 dB (`sample_train_snr`).
- **Verified on real tiles** (`toulon`, heavily saturated `panama`, zero-pixel `rotterdam3`; both noise types at 0, 10, 30 dB): realised SNR is within 0.015 dB of target in every band (tolerance 0.1 dB). Poisson-Gaussian noise is signal dependent (noise power in the brighter half of B08 is 1.60× the darker half), Gaussian is not (1.00×). Same seed gives identical noise, different seeds differ, per-band SNR arrays are honoured.
- **Not covered:** the "clean" reference is itself a real, already noisy image; realised SNR is relative to that reference (a threat to validity in the doc). Noise is spatially and spectrally independent, which real sensor noise may not be.

---

## Next steps

1. **SQLite reference DB** (`db/`) — tables for tiles, degradations, runs, per-tile/per-region metrics. `data/processed/tiles_meta.json`, `data/processed/signal_stats.json` and `reports/audit_report.json` seed the `tiles` table; the `degrade()` params dict maps onto `degradations`.
2. **Dataset / loader** — combine the patch index, tile arrays, splits, degradation and masks (ship, water, valid pixels) into a training and evaluation dataset. Needs PyTorch (not installed locally yet; a CPU build is enough for smoke tests). Must apply the per-instance counting rule for ships (see open questions).
3. **Models** — NAFNet backbone + FiLM SNR conditioning + cross-band attention + ship-weighted loss; baselines (FFDNet-style, SwinIR, DnCNN, BM3D).
4. **Training + experiments/ablations** (E1–E8, A1–A9 from the project doc) — the phase that actually needs GPU.
5. **Kaggle GPU setup + 3-person split** — deferred until there's real training code to run; see earlier discussion in this conversation for the memory-fit and quota-splitting analysis.

## Open questions / things to verify later
- GitHub push: blocked on authentication (see step 6). User to decide how to authenticate; nothing has been pushed.
- Tile overlap policy: overlapping tiles are kept in the same split (implemented). Whether to additionally mask out the overlapping strips is open; the `rotterdam` overlaps are small (7% and 14%), the `suez1`/`suez2` overlap is nearly total. Note that within a split, duplicated content between `suez1` and `suez2` is still counted twice in that split's statistics.
- Fold 1's test set is dominated by `rotterdam1` (49% of its test polygons, 61% of its ship pixels). Report per-tile results and not only per-fold averages (see step 10).
- Per-instance counting rule for evaluation: each ship appears in about 3.75 overlapping patches, so the assignment of a ship to a single patch (for E5 and bootstrap CIs) must be defined before evaluation code is written (see step 11).
- Invalid-pixel policy: `P_b` now excludes zeros and saturated pixels (done, step 12). Still open: whether to mask invalid pixels in SAM, ERGAS and the loss (probably yes), decided with the dataset/loss.
- Saturated-patch policy: how to filter or flag patches with heavy saturation for training and evaluation (243 patches with >5% saturated pixels in any band on the 6,048-patch grid: `panama` 93, `rotterdam2` 106, `rotterdam3` 44; none contain ship pixels). Options: keep and mask saturated pixels, or drop patches above a threshold. To be decided with the dataset.
- Poisson-Gaussian shot fraction `ρ` = 0.5 is an assumption, not calibrated to Sentinel-2; E2/A9 should vary it or justify it.
- Water mask for `rome` is defective (misses the river); decide the E5 treatment (local background, separate reporting, or exclusion).
- Ship instances: resolved as polygon count (1,053), which matches the project doc; the pixel mask agrees with the polygons (IoU 0.97).
