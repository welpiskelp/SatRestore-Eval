"""More reconstruction evidence for the Monday packet: one full-tile + ship close-up figure per
fold (using that fold's own conditioned checkpoint on one of its own held-out test tiles), plus a
single combined "one ship per fold" summary image, plus a table of the exact PSNR/SSIM numbers
behind every picture (pulled straight from the results DB, not recomputed).

Run: python scripts/make_more_visuals.py
"""
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
RGB = [3, 2, 1]

# one representative, ship-bearing test tile per fold (its OWN test tile, unseen by that fold's model)
FOLD_TILE = {0: "brest1", 1: "rotterdam1", 2: "toulon", 3: "southampton"}
SNRS = [5.0, 30.0]


def to_rgb(arr_chw, lo=None, hi=None):
    rgb = np.stack([arr_chw[i] for i in RGB], axis=-1).astype(np.float32)
    if lo is None:
        lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    return np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1), (lo, hi)


stats = load_signal_stats(Path("data/processed/signal_stats.json"))
all_centroids = load_ship_centroids(Path("data/S2-SHIPS/S2SHIPS"))

per_fold_data = {}  # fold -> dict(clean, noisies, preds, centroids, tile)

for fold, tile in FOLD_TILE.items():
    print(f"fold {fold}: loading checkpoint + tile {tile}...")
    model, cfg, scale = load_checkpoint(f"runs/full_medium_fold{fold}_seed0/best.pt")
    clean = np.load(f"data/processed/tiles/{tile}_image.npy")
    s_ = stats[tile]
    centroids = all_centroids[tile]
    noisies, preds = {}, {}
    for snr in SNRS:
        noisy, params = test_noisy_tile(clean, s_["power"], s_["mean"], tile, snr, "gaussian")
        cond = tile_condition(params, scale, cfg["model"]["cond_mode"])
        preds[snr] = predict_tile(model, noisy, cond, scale)
        noisies[snr] = noisy
        print(f"  snr={snr}dB done")
    per_fold_data[fold] = dict(clean=clean, noisies=noisies, preds=preds, centroids=centroids, tile=tile)

# --- one full-tile figure per fold: clean | noisy 5dB | restored 5dB | noisy 30dB | restored 30dB ---
for fold, d in per_fold_data.items():
    clean_rgb, stretch = to_rgb(d["clean"])
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6))
    panels = [
        (clean_rgb, "Clean reference"),
        (to_rgb(d["noisies"][5.0], *stretch)[0], "Noisy, 5 dB SNR"),
        (to_rgb(d["preds"][5.0], *stretch)[0], "Restored, 5 dB SNR"),
        (to_rgb(d["noisies"][30.0], *stretch)[0], "Noisy, 30 dB SNR"),
        (to_rgb(d["preds"][30.0], *stretch)[0], "Restored, 30 dB SNR"),
    ]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(f"Fold {fold} — {d['tile']} (held out from this fold's training)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT / f"07_full_tile_fold{fold}_{d['tile']}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved 07_full_tile_fold{fold}_{d['tile']}.png")

# --- one combined "one ship per fold" summary image: proof it works on 4 independent scenes ---
size = 56
fig, axes = plt.subplots(4, 3, figsize=(9, 12))
for row, (fold, d) in enumerate(per_fold_data.items()):
    h, w = d["clean"].shape[1:]
    cx, cy = d["centroids"][len(d["centroids"]) // 2]
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    clean_c = d["clean"][:, y0 : y0 + size, x0 : x0 + size]
    noisy_c = d["noisies"][5.0][:, y0 : y0 + size, x0 : x0 + size]
    pred_c = d["preds"][5.0][:, y0 : y0 + size, x0 : x0 + size]
    lo, hi = np.percentile(clean_c, 2), np.percentile(clean_c, 98)
    for col, (crop, title) in enumerate([(clean_c, "Clean"), (noisy_c, "Noisy 5 dB"), (pred_c, "Restored")]):
        rgb, _ = to_rgb(crop, lo, hi)
        ax = axes[row, col]
        ax.imshow(rgb)
        ax.axis("off")
        if row == 0:
            ax.set_title(title, fontsize=11)
    axes[row, 0].text(-0.15, 0.5, f"fold {fold}\n({d['tile']})", transform=axes[row, 0].transAxes,
                       fontsize=10, va="center", ha="right", rotation=90)
fig.suptitle("One ship per fold — the model generalises across 4 independent scenes", fontsize=13)
fig.tight_layout(rect=(0.03, 0, 1, 0.95))
fig.savefig(OUT / "08_one_ship_per_fold.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("saved 08_one_ship_per_fold.png")

# --- table: exact PSNR/SSIM for each shown tile, at the two SNRs shown, from the real eval results ---
con = sqlite3.connect(DB)
cur = con.cursor()
lines = ["# Reconstruction quality for the tiles shown in 01/07/08 (exact numbers from the evaluation DB)", ""]
lines.append("| Fold | Tile | SNR (dB) | PSNR, all pixels | PSNR, ship pixels | SSIM, all pixels |")
lines.append("|---|---|---|---|---|---|")
q = """
select value from metrics
where run_id=(select run_id from runs where name=?) and tile=? and snr_db=? and metric=? and region=? and noise_type='gaussian' and band is null
"""
for fold, d in per_fold_data.items():
    run_name = f"full_medium_fold{fold}_seed0"
    for snr in SNRS:
        psnr_all = cur.execute(q, (run_name, d["tile"], snr, "psnr", "all")).fetchone()[0]
        psnr_ship = cur.execute(q, (run_name, d["tile"], snr, "psnr", "ship")).fetchone()[0]
        ssim_all = cur.execute(q, (run_name, d["tile"], snr, "ssim", "all")).fetchone()[0]
        lines.append(f"| {fold} | {d['tile']} | {snr:g} | {psnr_all:.2f} | {psnr_ship:.2f} | {ssim_all:.3f} |")
(OUT / "09_shown_tiles_quality_table.md").write_text("\n".join(lines), encoding="utf-8")
print("saved 09_shown_tiles_quality_table.md")

print("\nAll files now in reports/partial_results/:")
for p in sorted(OUT.iterdir()):
    print(" ", p.name)
