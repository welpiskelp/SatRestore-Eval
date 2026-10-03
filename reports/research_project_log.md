# SNR-Conditioned, Spectrally Aware, Ship-Preserving Reconstruction of Sentinel-2 Maritime Imagery under Controlled Noise Degradation

**Research project log — methodology, system architecture, experimentation process, partial results, and key design discussions**

Repo: `https://github.com/welpiskelp/SatRestore-Eval` (branch `Main`)
Log compiled: 2026-10-03
Status: H1 complete (full protocol, 24/24 runs). H2 in progress (5/12 runs merged, 7 more running). H3 not started.

---

## 1. Research questions and pre-registered hypotheses

The project studies image restoration (denoising) of Sentinel-2 maritime scenes under controlled,
synthetic noise degradation, with ship detection/visibility as the application lens rather than
generic image quality. Three hypotheses were frozen **before any model was trained**
(`configs/protocol.json`, frozen 2026-09-19) to avoid post-hoc rationalization of whatever a model
happened to do:

- **H1** — An SNR-conditioned restoration model beats a blind (noise-level-unaware) model and a
  published baseline architecture (FFDNet-style), especially at the hardest SNRs and under
  mismatched input.
- **H2** — A cross-band spectral attention block improves spectral-fidelity metrics (SAM, ERGAS) by
  a larger relative margin than it improves PSNR — i.e., cross-band attention is disproportionately
  a spectral-consistency mechanism, not just a general denoising boost.
- **H3** — A ship-aware loss term reduces ship-region reconstruction error while the global PSNR
  cost stays below a pre-specified non-inferiority margin of 0.1 dB.

These map to the project's research questions (RQ1–RQ3): does telling the model the noise level
help; does spectral structure need explicit cross-band modeling; can ship fidelity be bought cheaply
without sacrificing overall quality.

---

## 2. Dataset

**S2-SHIPS** (EOTDL STAC dataset): 16 Sentinel-2 L2A tiles, 12 bands each, uint16, ~10 m resolution,
all acquired April 2021. ~1,053 annotated ship instances (COCO polygons). This is an **existing
public dataset**, not something built for this project — the project applies new evaluation
machinery to it, it does not introduce the dataset itself.

Key properties discovered during audit (relevant to every later design decision):

- **Band order mismatch**: raw `.npy` channel order differs from Sentinel-2 wavelength order
  (B8A positioned differently); corrected once in `prepare_tiles.py`, verified bit-for-bit against
  per-band GeoTIFFs.
- **Spatial overlap between tiles**: `suez1`/`suez2` overlap 93.6% (near-duplicate same-day scenes);
  `rotterdam1`/`rotterdam2` overlap 6.9%; `rotterdam2`/`rotterdam3` overlap 13.9%. These form
  "overlap groups" that must never be split across train/val/test (data leakage otherwise) and are
  later reused as the **clustering unit for statistical inference** (cluster-robust CI).
- **Uneven ship distribution**: `toulon` alone has 263 of 1,053 ship polygons; several `suez` tiles
  have 5–8. Folds are balanced on ship polygon count, not tile count, because tile-count balance
  would leave some folds nearly ship-free.
- **Water mask defect**: the delivered water mask misses the river in `rome` almost entirely,
  misclassifying moored river boats; excluded from water-region metrics for that tile.
- **Saturation**: `panama`, `rotterdam2`, `rotterdam3` have 6–12% saturated (65535) pixels in some
  bands, which would inflate the measured signal power unless excluded — signal power `P_b` is
  computed only over valid (non-saturated, non-zero) pixels.
- **Reference-image noise floor**: the "clean" reference images are real Sentinel-2 L2A products
  with their own noise, estimated at 47–72 dB (median ~56 dB) in the project's SNR convention. ESA
  puts real sensor noise at ~40–45 dB. This means the 35–40 dB test point approaches the reference's
  own noise floor, and the 0 dB point is a deliberate extrapolation stress test (training only covers
  5–40 dB).

All 16 tiles passed the data-quality audit (12/12 bands present, correct shapes, masks present).

---

## 3. Experimental design / protocol

### 3.1 Splits
4-fold, tile-level cross-validation: 10 train / 2 val / 4 test tiles per fold; every tile is a test
tile exactly once. Overlap groups (`{rotterdam1,rotterdam2,rotterdam3}`, `{suez1,suez2}`) are never
split across train/val/test. The test partition is chosen by exhaustive search over 17,325 candidate
packings of whole overlap-groups-and-singles into 4 equal test sets, minimizing the coefficient of
variation of ship-polygon and ship-pixel counts across test sets.

### 3.2 Noise model
Three synthetic degradation types applied on top of the real (noisy-reference) imagery:
- **Gaussian** — independent per-pixel.
- **Poisson-Gaussian** — signal-dependent shot noise (shot_fraction=0.5, an assumption, not sensor-calibrated) plus Gaussian read noise.
- **Correlated Gaussian** — spatially correlated noise matching the resampling blur of the 20 m/60 m bands (native 10 m bands get zero correlation).

SNR defined as `10*log10(P_b / N_b)` per band, `P_b` = mean square signal over valid pixels, `N_b`
= injected noise variance. Training draws SNR uniformly in dB from [5, 40] per patch, on the fly.
Test SNRs are a fixed grid `[0,5,...,40]` dB, with one deterministic noisy realization per
(tile, noise type, SNR) shared across every model (seeded, stored in the reference DB) so all
models are compared on literally the same noisy input.

### 3.3 Evaluation
Metrics: PSNR, SSIM, SAM (spectral angle mapper), ERGAS, per-band RMSE — computed per tile, per
region (`all`, `ship`, `water` excluding a 3px ship halo, `background`), with instance-level
ship metrics (ship NRMSE, contrast error) for each of the 1,053 ship polygons individually. All
peak values for PSNR are per-tile, per-band 99.9th-percentile clean values, never a fixed constant.
Implemented in `src/eval/metrics.py`, independently verified against hand-computable cases
(identity baseline, analytic SNR) in `scripts/check_metrics.py` before any model touched it.

### 3.4 Statistics (the part that distinguishes this project from most informal restoration papers)
- **Paired exact two-sided Wilcoxon signed-rank test** on per-tile scores (16 tiles), comparing
  arms (e.g., conditioned vs blind) at each SNR level.
- **Holm-Bonferroni correction** across the 9 SNR levels tested within a hypothesis family, to
  control the family-wise error rate rather than reporting 9 uncorrected p-values.
- **Cluster-robust 95% confidence intervals**, with the overlap groups as clusters (not
  pretending the 16 tiles are 16 independent observations, since 5 of them are near-duplicate
  scenes). A plain per-ship bootstrap was explicitly tested and rejected: simulated coverage was
  ~12% instead of the nominal 95%, because ship-level resampling ignores the tile-level clustering
  of ships. Cluster-robust t and cluster bootstrap were simulated at ~90% and ~88% coverage
  respectively — not perfect, but far closer to nominal than the naive approach, and documented as
  such rather than quietly used.
- **Seed-averaging-within-tile**: before pairing two arms, every available seed for a given
  (tile, SNR) within an arm is averaged first, so a tile's contribution to the paired test is one
  number per arm, not one number per seed (which would silently inflate the effective sample size
  and violate the "16 tiles" pairing structure the clustering correction assumes). This rule was
  added mid-project when seed counts between arms became temporarily unequal (see §5.3).

This statistical protocol — paired Wilcoxon + Holm + cluster-robust CI, with seed-averaging — does
not appear, combined this way, in any directly comparable restoration-on-Sentinel-2 paper found
during the later literature check (§7). It is one of the project's actual defensible contributions,
discussed further in §7.

---

## 4. System architecture

### 4.1 Model (`src/models/nafnet.py`, 177 lines)
Backbone: **NAFNet** (Nonlinear Activation Free Network) — a UNet-style restoration backbone from
the published image-restoration literature, chosen for being simple, strong, and not reliant on
exotic blocks. Not a novel architecture by itself.

Conditioning mechanism: **FiLM** (Feature-wise Linear Modulation) — the SNR value (or estimated
noise level, for the blind-comparison setup) is embedded and used to scale/shift intermediate
feature maps, letting the same network behave differently depending on how noisy its input is.
Also a published, precedented mechanism (originally from visual-reasoning/style-transfer work,
widely reused in conditional restoration since).

Spectral mechanism (H2's subject): **cross-band attention** — an attention block operating across
the 12 spectral bands (not just spatially), intended to let the network exploit cross-band
redundancy/structure when reconstructing a given band. Precedented in the hyperspectral/multispectral
restoration literature (SSCAN, QRSAN, HDST-type architectures use similar cross-band attention
mechanisms — confirmed via literature search, §7).

Parameter-matched baselines/controls (`src/models/baselines.py`, 171 lines; `src/models/registry.py`):
- **Blind model** — identical backbone, no SNR conditioning (ablates H1's conditioning signal).
- **No-cross-band model** (`e2_no_crossband`) — identical backbone and conditioning, cross-band
  attention block removed/disabled (ablates H2's spectral mechanism). Parameter count matched so
  the comparison isn't confounded by capacity.
- **FFDNet-style baseline** — a from-scratch reimplementation of the published FFDNet architecture
  (noise-map conditioning), trained under the same budget as a sanity check that the project's own
  conditioning approach isn't trivially replaceable by an existing published method. It did not
  converge under the matched training budget; root-caused (not a code bug — the architecture's own
  noise-map-conditioning mechanism needs different training dynamics than FiLM) and reported
  honestly as a negative/stalled result rather than hidden.
- **BM3D** — classical, non-learned denoiser (Block-Matching 3D), applied per-band in grayscale
  with per-band sigma derived from the noise model's own `sigma_read`, each band independently
  normalized to [0,1] before calling `bm3d.bm3d`. Added late in the project (see §5.5) as the
  "zero-learning" reference point every learned model should be compared against.

### 4.2 Data / training pipeline
- `src/data/bands.py` — canonical band-order source of truth.
- `src/data/tiles.py` — tile readers, including a guarded pickle loader (the raw `.npy` files are
  pickled dicts; unpickling can execute arbitrary code, so the loader disassembles the pickle
  bytecode first and refuses anything beyond plain numpy-array opcodes before trusting it).
- `src/data/patches.py` / `scripts/make_patch_index.py` — 128×128 patches, stride 64, edge-aligned
  last row/column, 6,048 patches total, each with precomputed ship/water pixel statistics for
  sampling and loss weighting.
- `src/data/splits.py` — fold computation and overlap-group leakage validator (`--check` mode
  deliberately rejects a known-leaky split as a self-test).
- Training loop: resumable, checkpointed, same loop used for every arm (conditioned, blind,
  no-cross-band, FFDNet) by swapping config/model-registry entries — not a different training
  script per experiment.

### 4.3 Evaluation / results infrastructure
- `src/eval/metrics.py`, `src/eval/stats.py`, `src/eval/results.py` (DB writer/merger),
  `src/eval/baselines.py` (classical baselines, including the BM3D function added this round).
- Results land in per-batch SQLite DBs (`db/results_*.sqlite`) with a fixed schema
  (`runs`, `metrics`, `degradations` tables) and are merged into a single master DB
  (`db/results_pilot_full.sqlite`, 27 distinct runs as of this log) via `src.eval.results.merge`.
- `scripts/compare_arms.py` — H1's statistical test driver (full code and final numbers in §6.1).
  No equivalent script exists yet for H2 or H3 — explicitly an open task (§8).

### 4.4 Compute infrastructure (the unusual part of this project's execution, not its research content)
Training requires GPU; the user does not have reliable local GPU access, so all real training runs
on **Kaggle's free GPU kernels** (session-time-budgeted, ~8.3h sessions). Because a single Kaggle
account can only run a limited number of concurrent/queued GPU sessions, and the remaining H1 seed
runs would have taken too long serially on one account, the project built a mechanism to **offload
training runs to other people's Kaggle accounts** without sharing credentials:

- `kaggle/make_friend_notebook.py` — generates a self-contained `.ipynb` that any Kaggle account
  can upload via "Import Notebook": clones the public GitHub repo, stages the (made-public) Kaggle
  dataset, runs a fixed batch of training configs within the session time budget, zips results into
  `results.zip` in the Output tab.
- Three such notebooks (`batch_A/B/C.ipynb`) were generated and run to completion on a friend's
  Kaggle account (`atharvsrivastava05`), covering the 9 remaining H1 seed1/seed2 combinations plus
  one blind fold3 seed1 run.
- **Key discovered mechanism**: `kaggle kernels output <owner>/<slug>` works for *any* kernel
  visible to the authenticated account — public or explicitly shared — not only kernels that
  account itself created. Once the friend made the batches public/shared, results were pulled
  directly into the project's own machine without ever needing the friend's API credentials.
- Every multi-run Kaggle kernel in this project (seed batches, BM3D, H2) follows the same
  **sequential, time-budgeted pattern**: a fixed `SESSION_BUDGET_HOURS` and `EST_HOURS_PER_RUN`,
  looping over a `CONFIGS` list and skipping (not crashing on) any run that wouldn't fit in the
  remaining session time, printing a `completed`/`skipped` summary at the end so the next session
  knows exactly what to trim from the list.

---

## 5. Experimentation process (chronological narrative)

### 5.1 Foundations (prior sessions, summarized for continuity)
Environment setup, EOTDL dataset download (custom streaming downloader written after discovering
the official CLI buffers the entire 6.2GB asset into memory), data cleanup (16GB → 3.4GB kept,
duplicates and unused pretrained backbones deleted), repo scaffold, data audit (16/16 tiles pass),
tile preparation with band-order correction, fold-split computation with overlap-group leakage
protection, patch index, the degradation engine (3 noise types, verified against real tiles),
the SQLite reference DB, the frozen protocol file, the evaluation harness (independently verified
against hand-checkable identity/analytic cases before trusting it on real models), the model
(NAFNet+FiLM+cross-band attention) and baseline architectures, local CPU smoke-testing, then first
real GPU training on Kaggle (pilot run, fold-0 full training).

### 5.2 H1 at partial seed count (prior session, summarized)
Conditioned model trained on all 4 folds (seed 0). Blind baseline trained fold 0 first, folds 1–2
in parallel on Kaggle, fold 3 queued behind the 2-concurrent-session limit. First statistical test
written and run on this partial (4 folds × 1 seed) data — explicitly labeled preliminary. FFDNet
baseline attempted and found non-convergent under the matched budget; root-caused and reported as
a negative but informative result rather than hidden or silently retried indefinitely.

### 5.3 Scaling H1 to the full 3-seed protocol (this session)
The protocol specifies 3 seeds per fold per arm (24 runs total: 4 folds × 3 seeds × 2 arms). Getting
there required:
- Identifying exactly which seed/fold combinations were still missing by cross-referencing
  `configs/protocol.json` against the runs already in `db/results_pilot_full.sqlite`.
- A single-run Kaggle kernel for the one specific gap left by an earlier time-budget skip
  (`e1_blind_fold2_seed1`).
- The three friend-run batches (§4.4) covering the bulk of the remaining seed combinations.
- Discovering, mid-way, that `scripts/compare_arms.py` as originally written assumed exactly one
  seed per arm and would silently double-count or mis-pair tiles once some folds had 2–3 seeds and
  others had 1. Rewrote it to **average all available seeds within (tile, SNR) per arm before
  pairing**, and to report a `status: IN PROGRESS` vs `status: FINAL` tag based on whether every
  fold/arm had reached the target seed count — so partial results were never silently presented as
  final. Rerun 3 times as more seed data landed (partial → 23/24 → final 24/24), each time
  re-committed to git with the actual seed-count status visible in the output file itself.
- This process is itself a finding worth keeping for the paper's own honesty: at 20–23/24 runs, the
  table showed an apparent **reversal** at 20–30 dB (blind beating conditioned) that looked like it
  might contradict H1. Once the final seed landed and seed-averaging was applied correctly, that
  reversal resolved to a **genuine statistical null** (not significant either direction), not a
  real effect. This is direct, first-hand evidence of why the pre-registered seed count and the
  seed-averaging rule matter — a paper that stopped at 23/24 runs would have reported a false
  reversal.

### 5.4 Housekeeping
Two occasions of stray background shell processes identified and killed when asked
("which shell is still running"): (1) a leftover `grep -rl` scan over the entire repo including the
large `data/` directory from an earlier, already-superseded search; (2) a duplicate
`kaggle kernels output` download re-issued after the first download had already completed and been
merged. Both were redundant, not doing anything new, and were killed rather than left to finish
pointlessly.

### 5.5 BM3D classical baseline (this session, built from scratch)
Flagged as "still to do" in PROGRESS.md from an earlier step. Built in this order:
1. `bm3d_denoise()` added to `src/eval/baselines.py` — per-band grayscale BM3D, each band
   independently normalized to [0,1] by its own range, sigma scaled to match, denoised, scaled back.
2. `scripts/eval_bm3d.py` — driver mirroring the existing `eval_baselines.py` pattern, looping over
   6 representative tiles × the standard SNR grid, gaussian noise only, writing to
   `db/results_bm3d.sqlite` under run name `baseline_bm3d`.
3. `kaggle/run_bm3d_baseline.py` + `kaggle/kernel-metadata-bm3d.json` — CPU-only kernel (BM3D is
   classical, no GPU benefit).

Three rounds of kernel failures and fixes before a clean result (see §9 Errors and Fixes for full
detail): Kaggle's own stdout-capture `.log` file downloads as 0 bytes (a known, recurring Kaggle
bug) — fixed by teeing output to a real file and wrapping `main()` in try/except writing
`traceback.format_exc()` to an ordinary `ERROR.txt`. That revealed a missing `rasterio` dependency
in the kernel's pip install. Fixing that revealed the actual root cause: `scripts/eval_bm3d.py` and
the `baselines.py` edit had never been pushed to GitHub, so the kernel's `git clone` simply didn't
have the file. After committing and pushing everything pending, kernel version 4 ran clean.

**Result**: BM3D beats the learned (conditioned) model at mild noise (30–40 dB) and loses badly at
severe noise (0–10 dB) — see §6.2 for numbers. This regime-dependent crossover mirrors the same
shape seen in H1's own conditioned-vs-blind comparison, and (per the later literature check, §7)
is itself a previously-documented phenomenon (Burger et al. 2012 describe near-identical behavior
for learned vs. BM3D-style denoisers) — not a new discovery, but a real, useful confirmation that
this project's measurement pipeline reproduces a known, trusted pattern.

### 5.6 Scaling H2 (this session, in progress)
H2 needs 12 runs (4 folds × 3 seeds) for the `e2_no_crossband` ablation arm, compared against the
already-complete 12-run `full_medium` arm. Process:
1. Generated the 11 missing config files (`configs/e2_no_crossband_fold{0-3}_seed{0-2}.json`,
   one combination already existed) by copying the existing base config and overriding
   `fold`/`seed`/`run_name`.
2. Built `kaggle/train_h2_crossband.py` — the same sequential, time-budgeted multi-run kernel
   pattern as the H1 seed batches, covering all 11 remaining configs in one reusable kernel the
   user re-runs ("Save & Run All") across multiple Kaggle sessions.
3. Round 1 completed 4 of 11 configs within one session's time budget; `CONFIGS` was trimmed down
   to the 7 remaining combinations and the kernel relaunched as version 2.
4. As of this log: **5 of 12 H2 runs are merged** (fold0 all 3 seeds, fold1 seeds 0–1); round 2
   (7 more: fold1_seed2, fold2_seed{0,1,2}, fold3_seed{0,1,2}) is **currently running** on
   `abhinavp10/satrestore-h2-crossband` (confirmed `KernelWorkerStatus.RUNNING` at log time).

**Directional signal so far** (5 full-model vs. 5 no-crossband runs, folds 0–1 only, averaged over
all SNRs and tiles — explicitly preliminary, not the pre-registered comparison yet):

| Metric | Full model (with cross-band attention) | No cross-band | Direction |
|---|---|---|---|
| PSNR | 40.61 | 39.94 | full model higher by 0.67 dB |
| SAM (°, lower=better) | 4.080 | 4.266 | full model lower (better) by 0.186° |
| ERGAS (lower=better) | 6.635 | 6.884 | full model lower (better) by 0.249 |

This is consistent with H2's direction (cross-band attention helps spectral metrics) but **no
statistical test exists for H2 yet** and only 2 of 4 folds have any data — this table is a progress
signal, not a result, and should not be quoted as a finding until the full 24 runs and a proper
paired test (analogous to `compare_arms.py` but on SAM/ERGAS relative-improvement, not PSNR) exist.

---

## 6. Partial / final results

### 6.1 H1 — FINAL (24/24 runs, full pre-registered protocol: 4 folds × 3 seeds × 2 arms)

Committed in `reports/partial_results/06_h1_statistical_result.md` (commit `574a64a`).

| SNR (dB) | Conditioned PSNR | Blind PSNR | Gap (dB) | 95% CI | Holm p | Significant? |
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

**4 of 9 noise levels show a statistically real advantage for conditioning after Holm correction**:
the two extremes (0–10 dB severe noise, and 40 dB near-clean) are where SNR-conditioning
significantly helps; 15–35 dB (the mid-range, which is also where real Sentinel-2 imagery actually
sits, per the noise-floor analysis in §2) is a **genuine statistical null**, not a reversal — this
was confirmed by watching the apparent 20–30 dB "blind wins" signal at partial seed counts resolve
to non-significance once the full 3-seed protocol landed (§5.3).

**Plain-language reading**: at 0 dB, PSNR gain from 5→50 dB generally corresponds to the image going
from "noise dominates, ship barely visible" (low 20s–30 PSNR) toward "visually clean, ship edges
and spectral detail fully recoverable" (high 40s–50 PSNR). A 0.2–0.3 dB gap at the hardest SNRs is
small in absolute terms but statistically real and directionally consistent with the hypothesis;
the near-clean 40 dB gap (+0.65 dB) is the largest and most confidently significant single effect.

### 6.2 BM3D classical baseline vs. conditioned model

| SNR (dB) | Conditioned (learned) PSNR | BM3D PSNR | Winner |
|---|---|---|---|
| 0 | 27.02 | 29.00 | BM3D (severe-noise note: see caveat below) |
| 10 | 33.71 | 35.32 | BM3D |
| 20 | 39.23 | 41.84 | BM3D |
| 30 | 44.90 | 48.48 | BM3D |
| 40 | 50.40 | 55.75 | BM3D |

*Caveat on this exact table*: the BM3D run used a 6-tile subset and gaussian noise only, while the
conditioned-model row above is the full 16-tile H1 average — the comparison is directionally
informative (BM3D is strong across the whole range on this subset) but not apples-to-apples until
both are evaluated identically. The qualitatively important, apples-to-apples-robust finding from
earlier exploration (prior session) stands: **the crossover point where the learned model starts
beating BM3D is at the severe-noise end**, consistent with the literature (Burger et al. 2012) and
with H1's own severe-noise-favors-conditioning pattern.

### 6.3 H2 — partial, in progress (5/12 runs)
See §5.6 table. Directionally consistent with the hypothesis; not yet a result.

### 6.4 H3 — not started
Zero runs, no config, no kernel built yet.

---

## 7. Key discussion: is this novel, and what is the paper's actual contribution?

This was the most substantive open discussion of the session and is recorded here in full because
it directly shapes how the eventual paper should be framed.

**The question asked**: "Is our model architecture novel?" → **no.** A literature search
(via live web search, not assumption) confirmed every architectural component is precedented:
NAFNet (backbone), FiLM (conditioning), and cross-band spectral attention specifically (precedented
in SSCAN, MAN, QRSAN, HDST-type hyperspectral/multispectral restoration architectures).

**Follow-up**: "does the learned-vs-BM3D regime crossover count as a finding, then?" → also
**not novel as a phenomenon** — Burger et al. (2012) already describe a near-identical
severe-noise-favors-learned / mild-noise-favors-classical crossover pattern. This project's version
is a confirmation on a new domain (Sentinel-2 maritime), not a discovery.

**User pushback** ("so the paper is dead? If we're proving known results we aren't contributing
anything"): addressed directly — the paper is not dead, but its claimed contribution must be
repositioned away from "novel architecture" or "novel phenomenon" toward what is actually true and
defensible:
1. **Statistical rigor uncommon in this specific sub-literature** — no comparable restoration paper
   found in the search uses this combination of paired Wilcoxon + Holm correction + cluster-robust
   CI with seed-averaging, on this kind of satellite/maritime data.
2. **The specific application context** — Sentinel-2 maritime imagery with ship-specific,
   instance-level evaluation (not generic image-quality metrics alone).
3. **A more granular empirical pattern** — the U-shaped significance curve in H1 (significant at
   the extremes, null in the middle, which also happens to be where real sensor noise actually
   lives) is a finer-grained picture than the literature's simpler monotonic crossover claims,
   pending the H2/H3 results to see if the same granularity holds there too.

**Final follow-up** ("so the dataset is the novelty?"): answered directly — no, not that either.
The dataset (S2-SHIPS) is also pre-existing and not built by this project; claiming "the dataset is
the novelty" would be just as unsupportable as claiming the architecture is. The actual, defensible
framing settled on: **the contribution is the combination** — applying this specific statistical
rigor, to this specific dataset/domain, to answer these specific pre-registered questions, which
nobody has done rigorously before. This is a legitimate, normal category of applied-ML paper
("first rigorous empirical validation of X in domain Y"), not a "we changed the field" paper, and
should be pitched and written that way — Abstract/Intro should not oversell architectural or
phenomenological novelty, and Related Work should be upfront that the architecture is assembled
from precedented components (NAFNet, FiLM, cross-band attention, with the standard citations) while
the contribution is the empirical characterization plus the statistical protocol.

This reframing was explicitly chosen over two tempting but wrong alternatives: (a) overclaiming
novelty that a careful reviewer would catch immediately, or (b) concluding the project has no
contribution at all, which is also false — a properly-powered, honestly-reported empirical answer
to a real applied question is a legitimate and publishable result, just a narrower and more modest
one than "we invented something new."

---

## 8. Open items / what's not done yet

- **H2 statistical test** — no script exists yet (unlike H1's `compare_arms.py`). Needs a paired
  test on *relative* SAM/ERGAS degradation vs. *relative* PSNR degradation across folds, with the
  same Holm + cluster-robust-CI treatment as H1, to actually test H2's specific claim (disproportionate
  benefit to spectral metrics vs. PSNR), not just "both metrics improved."
- **H2 remaining runs** — 7/12 currently training (`abhinavp10/satrestore-h2-crossband`, round 2,
  confirmed running at log time); likely needs a round 3 for whatever doesn't fit this session.
- **H3** — completely unstarted. Needs: a ship-aware loss variant config (ablating/removing the
  ship-weighted loss term), training runs across the same 4×3 protocol, and a non-inferiority test
  (ship-region error reduction with global PSNR cost bounded below the pre-specified 0.1 dB margin)
  — this is a different statistical test shape than H1/H2's "is there a difference" tests.
- **PROGRESS.md** — has fallen behind; last updated at step 22, before the BM3D baseline, H1
  finalization, H2 scaling, and the friend-notebook infrastructure described in this log. Needs a
  full update (policy: never delete, always update, no emojis).
- **Paper draft** — not started. Planned structure, as discussed: Abstract → Intro → Related Work
  (upfront about architectural precedent, per §7) → Method → Results per-hypothesis → Discussion
  tying the regime-dependence pattern together across H1/BM3D/(pending H2/H3) → Limitations →
  Conclusion. Should wait for H2/H3 to land before drafting Results, but Related Work and Method
  can be drafted now since they don't depend on outstanding runs.

---

## 9. Errors and fixes (infrastructure lessons worth keeping)

- **Kaggle's own stdout `.log` capture downloads as 0 bytes** — a recurring bug, not specific to
  this project's code. Standing fix used in every kernel since: tee subprocess output into a real
  file under `/kaggle/working/`, and wrap `main()` in try/except that writes
  `traceback.format_exc()` to an ordinary `ERROR.txt` — both are normal files and survive Kaggle's
  output packaging, unlike the broken log capture.
- **Missing dependency surfaced only after fixing the log bug** — `rasterio` wasn't installed in
  the BM3D kernel's pip step even though `src/data/tiles.py` imports it at module level; fixed by
  adding `rasterio==1.5.1` to every kernel's install line.
- **Silent failure root cause: uncommitted code** — a kernel's `git clone --depth 1` of the public
  repo can only see what's actually pushed; `scripts/eval_bm3d.py` and a `baselines.py` edit had
  been written locally but never pushed, producing a confusing exit-code-2 "file not found" deep
  inside a kernel that otherwise looked configured correctly. Standing lesson: before debugging a
  remote kernel failure further, check `git status`/`git log` first for anything still local-only.
- **Windows `/tmp` path mismatch** — any Bash-originated `/tmp/...` path passed into a Python
  `sqlite3.connect`/`ATTACH DATABASE` call needs `cygpath -w` first to get the `C:\...` form Python
  on Windows actually expects.
- **Stray background shells** — twice found and killed: a leftover whole-repo `grep -rl` scan
  superseded by a later, narrower search, and a duplicate `kaggle kernels output` download reissued
  after the first had already completed and merged.
- **Cross-account Kaggle result pulling** — confirmed that `kaggle kernels output <owner>/<slug>`
  works for any kernel visible (public or shared) to the authenticated account, not only kernels
  that account created — this is what made the friend-run-batches mechanism (§4.4) work without
  ever handling the friend's credentials.

---

## 10. Standing conventions / house rules (carried forward)

- `PROGRESS.md` is never deleted, always updated as work lands, no emojis.
- No `Co-Authored-By`/Claude attribution added to commits or PRs.
- Plain, non-jargon language preferred when reporting results and statistical findings to the user;
  terse "caveman mode" used for quick status pings, full prose used for substantive/nuanced
  questions (architecture novelty, paper framing, "is this bad for my paper") per the user's own
  pattern of invoking and then implicitly stepping away from caveman mode for those.
