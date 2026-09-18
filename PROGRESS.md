# Progress Log

**Project:** SNR-Conditioned, Spectrally Aware, Ship-Preserving Reconstruction of Sentinel-2 Maritime Imagery under Realistic Noise
**Repo:** https://github.com/welpiskelp/SatRestore-Eval (branch `Main`)
**Last updated:** 2026-09-18

---

## Status at a glance

| Phase | Status |
|---|---|
| 1. Local setup, EOTDL download, repo scaffold, audit | DONE (committed locally, not yet pushed) |
| 2. Patch preprocessing (128×128, stride 64) | NOT STARTED |
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
- Repo initialized, `origin` set to the GitHub repo, one commit made locally on `Main` containing all of the above (code/config/report only — **not pushed yet**, and never includes `data/`).

---

## Next steps

1. **Patch preprocessing** — cut all 16 tiles into 128×128 patches at stride 64 (per-tile, so the 4-fold tile-level CV split stays clean). Needs its own scoping pass (file layout for patches, how splits are recorded, where patches live on disk) before writing code.
2. **Degradation engine** — Gaussian + Poisson-Gaussian noise, per-band SNR = 10·log10(P_b/σ_b²), continuous 5–30dB sampling for training / fixed 0–30dB (7 levels) for test.
3. **SQLite reference DB** (`db/`) — tables for tiles, degradations, runs, per-tile/per-region metrics. `reports/audit_report.json` is a natural seed for the `tiles` table.
4. **Models** — NAFNet backbone + FiLM SNR conditioning + cross-band attention + ship-weighted loss; baselines (FFDNet-style, SwinIR, DnCNN, BM3D).
5. **Training + experiments/ablations** (E1–E8, A1–A9 from the project doc) — the phase that actually needs GPU.
6. **Kaggle GPU setup + 3-person split** — deferred until there's real training code to run; see earlier discussion in this conversation for the memory-fit and quota-splitting analysis.

## Open questions / things to verify later
- Whether to push the current commit to GitHub now or keep accumulating locally first (user's call).
- Exact definition to use for "ship instance" in later ship-aware loss/eval work — segmentation-polygon count (1,053) vs. annotation-record count (143) — resolved here as polygon count, matching the doc, but worth double-checking against how the ship-aware loss actually consumes masks once that's built.
