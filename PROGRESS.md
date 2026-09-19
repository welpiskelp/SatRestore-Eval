# Progress Log

**Project:** SNR-Conditioned, Spectrally Aware, Ship-Preserving Reconstruction of Sentinel-2 Maritime Imagery under Realistic Noise
**Repo:** https://github.com/welpiskelp/SatRestore-Eval (branch `Main`)
**Last updated:** 2026-09-19

---

## Status at a glance

| Phase | Status |
|---|---|
| 1. Local setup, EOTDL download, repo scaffold, audit | DONE (committed locally; GitHub push blocked on authentication) |
| 2a. Tile preparation (pickle-free canonical arrays, aligned masks) | DONE (not yet committed) |
| 2b. Overlap-aware tile-level fold splits (4 folds, 10/2/4) | NOT STARTED |
| 2c. Patch index (128×128, stride 64) with per-patch ship/water stats | NOT STARTED |
| 3. Degradation engine (Gaussian / Poisson-Gaussian, SNR sampling) | NOT STARTED |
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

---

## Next steps

1. **Fold splits (2b)** — build a seeded, overlap-aware 4-fold assignment (10 train / 2 val / 4 test tiles per fold, each tile tested exactly once), keeping `{suez1, suez2}` and `{rotterdam1, rotterdam2, rotterdam3}` together within a fold, and balancing ship counts across test sets. Save to `configs/splits.json` so it is versioned and reviewable.
2. **Patch index (2c)** — 128×128 patches at stride 64 over each tile (13 × 26 = 338 patches per tile, 5,408 total; the last 42 rows and 55 columns are not covered unless edge-aligned patches are added — decision needed). Store as an index (tile, row, col) plus per-patch ship pixel count, water fraction and nodata fraction, and slice patches from the compact tile arrays on the fly rather than duplicating imagery on disk (about 2.1GB materialized versus 0.64GB of tiles).
3. **Degradation engine** — Gaussian + Poisson-Gaussian noise, per-band SNR = 10·log10(P_b/σ_b²), continuous 5–30dB sampling for training / fixed 0–30dB (7 levels) for test.
4. **SQLite reference DB** (`db/`) — tables for tiles, degradations, runs, per-tile/per-region metrics. `data/processed/tiles_meta.json` and `reports/audit_report.json` are natural seeds for the `tiles` table.
5. **Models** — NAFNet backbone + FiLM SNR conditioning + cross-band attention + ship-weighted loss; baselines (FFDNet-style, SwinIR, DnCNN, BM3D).
6. **Training + experiments/ablations** (E1–E8, A1–A9 from the project doc) — the phase that actually needs GPU.
7. **Kaggle GPU setup + 3-person split** — deferred until there's real training code to run; see earlier discussion in this conversation for the memory-fit and quota-splitting analysis.

## Open questions / things to verify later
- GitHub push: blocked on authentication (see step 6). User to decide how to authenticate; nothing has been pushed.
- Edge coverage for patches: with 128×128 at stride 64, the last 42 rows and 55 columns of every tile are not covered. Either accept that, or add edge-aligned patches (more coverage, but overlap unevenly). Not yet decided.
- Tile overlap policy: keep overlapping tiles in the same split (planned), versus additionally masking out the overlapping strips. The `rotterdam` overlaps are small (7% and 14%), the `suez1`/`suez2` overlap is nearly total.
- Ship-mask definition for the ship-aware loss: pixel mask (`label`, 0/1) from the npy versus rasterised COCO polygons. The pixel mask is what is prepared now; the two should be reconciled when the loss is built.
- Ship-on-water fractions for `rome` (68%) and `marseille` (81%) are low; worth a visual check before relying on the water mask for E5 metrics.
- Exact definition to use for "ship instance" in later ship-aware loss/eval work — segmentation-polygon count (1,053) vs. annotation-record count (143) — resolved here as polygon count, matching the doc, but worth double-checking against how the ship-aware loss actually consumes masks once that's built.
