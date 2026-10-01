"""Generate the Monday progress-packet visuals into reports/partial_results/.

Uses the fold-0 conditioned checkpoint on brest1 (a fold-0 TEST tile, unseen during training)
for the qualitative figures, and the merged results DB for the quantitative figures/tables.
Run from the repo root: python scripts/make_partial_results.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.data.tiles import load_ship_centroids
from src.degrade.noise import test_noisy_tile
from src.degrade.stats import load_signal_stats
from src.eval.evaluate import load_checkpoint
from src.eval.predict import predict_tile, tile_condition

OUT = Path("reports/partial_results")
OUT.mkdir(parents=True, exist_ok=True)
DB = "db/results_pilot_full.sqlite"
TILE = "brest1"  # fold-0 TEST tile, unseen by the fold-0 conditioned model
RGB = [3, 2, 1]  # B04, B03, B02 -> R, G, B

COND_COLOR = "#1b6ca8"
BLIND_COLOR = "#e07b39"
FFDNET_COLOR = "#9e2b25"


def to_rgb(arr_chw, lo=None, hi=None):
    """arr_chw: (12,H,W) DN array -> uint8 (H,W,3) true color, 2-98 percentile stretch."""
    rgb = np.stack([arr_chw[i] for i in RGB], axis=-1).astype(np.float32)
    if lo is None:
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    out = np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1)
    return out, (lo, hi)


# ---------------------------------------------------------------------------
# 1 + 2: full-tile reconstructions and a ship close-up gallery, fold-0 conditioned model
# ---------------------------------------------------------------------------
print("loading fold-0 conditioned checkpoint + brest1 tile...")
model, cfg, scale = load_checkpoint("runs/full_medium_fold0_seed0/best.pt")
stats = load_signal_stats(Path("data/processed/signal_stats.json"))
clean = np.load(f"data/processed/tiles/{TILE}_image.npy")
centroids = load_ship_centroids(Path("data/S2-SHIPS/S2SHIPS"))[TILE]
s_ = stats[TILE]

SNRS = [5.0, 30.0]
preds, noisies = {}, {}
for snr in SNRS:
    noisy, params = test_noisy_tile(clean, s_["power"], s_["mean"], TILE, snr, "gaussian")
    cond = tile_condition(params, scale, cfg["model"]["cond_mode"])
    pred = predict_tile(model, noisy, cond, scale)
    noisies[snr], preds[snr] = noisy, pred
    print(f"  snr={snr}dB done")

# --- full-scale tile figure (one row per SNR: clean | noisy | restored) ---
fig, axes = plt.subplots(len(SNRS), 3, figsize=(15, 5.2 * len(SNRS)))
clean_rgb, stretch = to_rgb(clean)
for row, snr in enumerate(SNRS):
    noisy_rgb, _ = to_rgb(noisies[snr], *stretch)
    pred_rgb, _ = to_rgb(preds[snr], *stretch)
    for col, (img, title) in enumerate([
        (clean_rgb, "Clean reference"),
        (noisy_rgb, f"Noisy input ({snr:g} dB SNR)"),
        (pred_rgb, f"Model output ({snr:g} dB SNR)"),
    ]):
        ax = axes[row, col]
        ax.imshow(img)
        ax.set_title(title, fontsize=12)
        ax.axis("off")
    axes[row, 0].set_ylabel(f"SNR {snr:g} dB", fontsize=12)
fig.suptitle(f"Full-tile reconstruction — {TILE} (fold-0 test tile, not used in training)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "01_full_tile_reconstruction.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 01_full_tile_reconstruction.png")

# --- ship close-up gallery: 4 ships x (clean, noisy@5dB, restored@5dB, noisy@30dB, restored@30dB) ---
size = 56
h, w = clean.shape[1:]
picks = centroids[:: max(1, len(centroids) // 4)][:4]
fig, axes = plt.subplots(len(picks), 5, figsize=(15, 3.1 * len(picks)))
col_titles = ["Clean", "Noisy 5 dB", "Restored 5 dB", "Noisy 30 dB", "Restored 30 dB"]
for row, (cx, cy) in enumerate(picks):
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    crops = [
        clean[:, y0 : y0 + size, x0 : x0 + size],
        noisies[5.0][:, y0 : y0 + size, x0 : x0 + size],
        preds[5.0][:, y0 : y0 + size, x0 : x0 + size],
        noisies[30.0][:, y0 : y0 + size, x0 : x0 + size],
        preds[30.0][:, y0 : y0 + size, x0 : x0 + size],
    ]
    lo, hi = np.percentile(crops[0], 2), np.percentile(crops[0], 98)
    for col, crop in enumerate(crops):
        rgb, _ = to_rgb(crop, lo, hi)
        ax = axes[row, col]
        ax.imshow(rgb)
        ax.axis("off")
        if row == 0:
            ax.set_title(col_titles[col], fontsize=11)
fig.suptitle(f"Ship close-ups — {TILE} (fold-0 test tile)", fontsize=14)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT / "02_ship_closeups.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 02_ship_closeups.png")

# ---------------------------------------------------------------------------
# 3: PSNR vs SNR, conditioned vs blind vs ffdnet, averaged over whatever folds exist per arm
# ---------------------------------------------------------------------------
con = sqlite3.connect(DB)
cur = con.cursor()


def mean_psnr_by_snr(name_like):
    q = """
    select m.snr_db, avg(m.value)
    from metrics m join runs r on m.run_id = r.run_id
    where m.metric='psnr' and m.region='all' and m.noise_type='gaussian' and m.band is null
    and r.name like ?
    group by m.snr_db order by m.snr_db
    """
    rows = cur.execute(q, (name_like,)).fetchall()
    return [r[0] for r in rows], [r[1] for r in rows]


snr_c, psnr_c = mean_psnr_by_snr("full_medium_fold%_seed0")
snr_b, psnr_b = mean_psnr_by_snr("e1_blind_fold%_seed0")
snr_f, psnr_f = mean_psnr_by_snr("e1_ffdnet_matched_fold0_seed0")

n_cond = cur.execute("select count(distinct name) from runs where name like 'full_medium_fold%_seed0'").fetchone()[0]
n_blind = cur.execute("select count(distinct name) from runs where name like 'e1_blind_fold%_seed0'").fetchone()[0]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(snr_c, psnr_c, "-o", color=COND_COLOR, label=f"Conditioned ({n_cond} folds)")
ax1.plot(snr_b, psnr_b, "--s", color=BLIND_COLOR, label=f"Blind, no conditioning ({n_blind} folds)")
ax1.plot(snr_f, psnr_f, ":^", color=FFDNET_COLOR, label="FFDNet baseline (did not converge)")
ax1.set_xlabel("Input noise level (SNR, dB)")
ax1.set_ylabel("Reconstruction quality (PSNR, dB) — higher is better")
ax1.set_title("Overall reconstruction quality")
ax1.legend(fontsize=9)
ax1.grid(alpha=0.3)

diff = [c - b for c, b in zip(psnr_c, psnr_b)] if snr_c == snr_b else None
if diff:
    colors = [COND_COLOR if d >= 0 else BLIND_COLOR for d in diff]
    ax2.bar([str(int(s)) for s in snr_c], diff, color=colors)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Input noise level (SNR, dB)")
    ax2.set_ylabel("Conditioned − Blind (dB)")
    ax2.set_title("Where conditioning helps (blue) or doesn't (orange)")
    ax2.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(OUT / "03_psnr_vs_snr_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 03_psnr_vs_snr_comparison.png")

# ---------------------------------------------------------------------------
# 4: cross-fold consistency at the SNR extremes
# ---------------------------------------------------------------------------
folds_q = """
select r.fold, m.snr_db, r.name, avg(m.value)
from metrics m join runs r on m.run_id = r.run_id
where m.metric='psnr' and m.region='all' and m.noise_type='gaussian' and m.band is null
and m.snr_db in (0, 40) and (r.name like 'full_medium_fold%_seed0' or r.name like 'e1_blind_fold%_seed0')
group by r.fold, m.snr_db, r.name
"""
by_fold = {}
for fold, snr, name, val in cur.execute(folds_q):
    kind = "cond" if name.startswith("full_medium") else "blind"
    by_fold.setdefault((fold, snr), {})[kind] = val

folds_present = sorted({f for f, _ in by_fold if "cond" in by_fold[(f, _)] and "blind" in by_fold.get((f, _), {})} | {f for f, s in by_fold})
rows_0 = [(f, by_fold.get((f, 0.0), {})) for f in sorted({f for f, s in by_fold if s == 0.0})]
rows_40 = [(f, by_fold.get((f, 40.0), {})) for f in sorted({f for f, s in by_fold if s == 40.0})]

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
for ax, rows, title in [(axes[0], rows_0, "At 0 dB SNR (hardest noise)"), (axes[1], rows_40, "At 40 dB SNR (easiest noise)")]:
    gains = [(f, d["cond"] - d["blind"]) for f, d in rows if "cond" in d and "blind" in d]
    if not gains:
        continue
    xs = [f"fold {f}" for f, _ in gains]
    ys = [g for _, g in gains]
    colors = [COND_COLOR if g >= 0 else BLIND_COLOR for g in ys]
    ax.bar(xs, ys, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Conditioned − Blind PSNR (dB)")
    ax.grid(alpha=0.3, axis="y")
fig.suptitle("Does conditioning's advantage hold up across folds?", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(OUT / "04_fold_consistency.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 04_fold_consistency.png")

# ---------------------------------------------------------------------------
# 5: tables
# ---------------------------------------------------------------------------
summary_q = """
select m.snr_db, r.name, avg(m.value)
from metrics m join runs r on m.run_id = r.run_id
where m.metric='psnr' and m.region='all' and m.noise_type='gaussian' and m.band is null
and (r.name like 'full_medium_fold%_seed0' or r.name like 'e1_blind_fold%_seed0' or r.name like 'e1_ffdnet%')
group by m.snr_db, r.name order by m.snr_db
"""
rows = cur.execute(summary_q).fetchall()
runs_seen = sorted({r[1] for r in rows})
table = {}
for snr, name, val in rows:
    table.setdefault(snr, {})[name] = val

lines = ["# Partial results summary (PSNR, dB; region=all, gaussian noise, averaged over available folds per arm)", ""]
lines.append("| SNR (dB) | " + " | ".join(runs_seen) + " |")
lines.append("|---|" + "---|" * len(runs_seen))
for snr in sorted(table):
    row = [f"{table[snr].get(r, float('nan')):.2f}" if r in table[snr] else "-" for r in runs_seen]
    lines.append(f"| {snr:g} | " + " | ".join(row) + " |")
(OUT / "05_results_summary.md").write_text("\n".join(lines))
print("saved 05_results_summary.md")

import csv
full_q = """
select r.name, r.fold, m.noise_type, m.snr_db, m.region, m.metric, m.value
from metrics m join runs r on m.run_id = r.run_id
where m.band is null
"""
with open(OUT / "05_results_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["run_name", "fold", "noise_type", "snr_db", "region", "metric", "value"])
    w.writerows(cur.execute(full_q).fetchall())
print("saved 05_results_table.csv")

print("\nDone. Files in reports/partial_results/:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
