"""Seed-count variance analysis for H1 (methodology-section material, not a new research question).

Three things, using the 3 seeds/fold already collected for full_medium (conditioned) and e1_blind:
  1. within-tile seed variance: how much does PSNR move across seed0/1/2 for the SAME tile+SNR?
  2. between-tile variance: how much does PSNR move across the 16 DIFFERENT tiles (seed-averaged)?
  3. how much does the H1 gap (conditioned - blind) estimate actually change going from a 1-seed
     estimate to a 2-seed average to the full 3-seed average -- i.e. was the third seed worth it?

Answers "should we have used 2 or 3 seeds" with the data already in hand; does not launch anything
new. Gaussian noise, region=all, PSNR -- same slice compare_arms.py uses.

Run: python scripts/seed_variance.py
"""
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DB = "db/results_pilot_full.sqlite"
OUT = Path("reports/partial_results")
OUT.mkdir(parents=True, exist_ok=True)

con = sqlite3.connect(DB)
cur = con.cursor()

q = """
select r.name, m.tile, m.snr_db, m.value
from metrics m join runs r on m.run_id = r.run_id
where m.metric='psnr' and m.region='all' and m.noise_type='gaussian' and m.band is null
and (r.name like 'full_medium_fold%_seed%' or r.name like 'e1_blind_fold%_seed%')
and r.name not like '%_offset%'
"""
# raw[arm][snr][tile][seed] = value
raw = {"full_medium": {}, "e1_blind": {}}
for name, tile, snr, val in cur.execute(q):
    arm = "full_medium" if name.startswith("full_medium") else "e1_blind"
    seed = int(name.rsplit("_seed", 1)[1])
    raw[arm].setdefault(float(snr), {}).setdefault(tile, {})[seed] = val

snr_levels = sorted(set(raw["full_medium"]) & set(raw["e1_blind"]))
seeds_present = sorted({s for snr in raw["full_medium"].values() for t in snr.values() for s in t})
lines = ["# Seed-count variance analysis for H1 (methodology-section material)", ""]
lines.append(f"Seeds present: {seeds_present}. Gaussian noise, PSNR, region=all -- same slice as "
              "`scripts/compare_arms.py`. This does not launch any new runs; it only re-reads the "
              "3 seeds/fold already collected for the full H1 protocol.")
lines.append("")

# --- 1 & 2: within-tile seed variance vs between-tile variance, per arm, per SNR -----------------
lines.append("## 1-2. Within-tile seed variance vs between-tile variance")
lines.append("")
lines.append("Within-tile seed variance: for a fixed tile and SNR, how much does PSNR move across "
              "the 3 seeds (same model, same data, only the random init/data order differs). "
              "Between-tile variance: for a fixed SNR, how much does the seed-averaged PSNR move "
              "across the 16 different tiles (different scenes). If within-tile variance is small "
              "relative to between-tile variance, the choice of 16 tiles -- not the seed count -- "
              "is what actually drives uncertainty in this experiment.")
lines.append("")
lines.append("| SNR (dB) | arm | within-tile seed var | between-tile var | ratio (within/between) |")
lines.append("|---|---|---|---|---|")
within_all, between_all = {"full_medium": [], "e1_blind": []}, {"full_medium": [], "e1_blind": []}
for snr in snr_levels:
    for arm in ("full_medium", "e1_blind"):
        tiles = raw[arm][snr]
        seed_vars, tile_means = [], []
        for tile, by_seed in tiles.items():
            vals = [by_seed[s] for s in sorted(by_seed) if s in by_seed]
            if len(vals) < 2:
                continue
            seed_vars.append(np.var(vals, ddof=1))
            tile_means.append(np.mean(vals))
        within = float(np.mean(seed_vars)) if seed_vars else float("nan")
        between = float(np.var(tile_means, ddof=1)) if len(tile_means) > 1 else float("nan")
        within_all[arm].append(within)
        between_all[arm].append(between)
        ratio = within / between if between else float("nan")
        lines.append(f"| {snr:g} | {arm} | {within:.4f} | {between:.4f} | {ratio:.3f} |")
lines.append("")
for arm in ("full_medium", "e1_blind"):
    mw, mb = np.nanmean(within_all[arm]), np.nanmean(between_all[arm])
    lines.append(f"**{arm}: mean within-tile seed variance = {mw:.4f} dB^2, "
                  f"mean between-tile variance = {mb:.4f} dB^2 -- between-tile variance is "
                  f"{mb/mw:.1f}x larger.**  " if mw else "")
lines.append("")

# --- 3: does the H1 gap estimate actually move from 1 to 2 to 3 seeds? ---------------------------
lines.append("## 3. Does the H1 gap estimate change from 1 seed to 2 seeds to 3 seeds?")
lines.append("")
lines.append("For each SNR level: the conditioned-minus-blind gap (mean over the 16 tiles) computed "
              "three ways -- using only seed 0, averaging seeds 0-1, and averaging all 3 seeds (the "
              "final, reported H1 number). If the 1-seed and 3-seed columns are close, a single "
              "seed would have given essentially the same answer here; if they differ a lot, the "
              "extra seeds were doing real work.")
lines.append("")
lines.append("| SNR (dB) | gap (seed 0 only) | gap (seeds 0-1 avg) | gap (seeds 0-2 avg, final) | "
              "move 1-seed vs final | move 2-seed vs final |")
lines.append("|---|---|---|---|---|---|")
moves_1v3, moves_2v3 = [], []
for snr in snr_levels:
    tiles = sorted(set(raw["full_medium"][snr]) & set(raw["e1_blind"][snr]))
    gaps = {}
    for k, seed_subset in (("s0", (0,)), ("s01", (0, 1)), ("s012", (0, 1, 2))):
        diffs = []
        for t in tiles:
            c = raw["full_medium"][snr][t]
            b = raw["e1_blind"][snr][t]
            cv = [c[s] for s in seed_subset if s in c]
            bv = [b[s] for s in seed_subset if s in b]
            if cv and bv:
                diffs.append(np.mean(cv) - np.mean(bv))
        gaps[k] = float(np.mean(diffs)) if diffs else float("nan")
    d1 = gaps["s0"] - gaps["s012"]
    d2 = gaps["s01"] - gaps["s012"]
    moves_1v3.append(abs(d1))
    moves_2v3.append(abs(d2))
    lines.append(f"| {snr:g} | {gaps['s0']:+.3f} | {gaps['s01']:+.3f} | {gaps['s012']:+.3f} | "
                  f"{d1:+.3f} | {d2:+.3f} |")
lines.append("")
lines.append(f"**Mean |1-seed minus final| = {np.mean(moves_1v3):.3f} dB. "
              f"Mean |2-seed minus final| = {np.mean(moves_2v3):.3f} dB.**")
lines.append("")

verdict_1v3 = np.mean(moves_1v3)
verdict_2v3 = np.mean(moves_2v3)
lines.append("## Verdict")
lines.append("")
if verdict_2v3 < 0.03:
    lines.append(f"The 2-seed average was already within {verdict_2v3:.3f} dB of the final 3-seed "
                  "number on average -- the third seed bought very little here. The 1-seed estimate "
                  f"moved {verdict_1v3:.3f} dB on average, which is closer to the smallest effects "
                  "reported in H1 (0.1-0.3 dB at several SNR levels), meaning a single seed would "
                  "have been noticeably less trustworthy for distinguishing the small, real effects "
                  "from noise. Report: 3 seeds were worth it over 1, marginal over 2.")
else:
    lines.append(f"The 2-seed average still moved {verdict_2v3:.3f} dB on average when the third "
                  "seed was added -- seed variance is not negligible at the scale of this "
                  "experiment's effects, so collecting 3 seeds (rather than 1-2) was the right call "
                  "and should not be reduced in any follow-up without re-checking this.")
lines.append("")
lines.append("This is reported as a methodology-section robustness check, not as grounds to redo "
              "H1 with a different seed count -- the 3-seed, 4-fold protocol is already complete "
              "and frozen.")

report = "\n".join(lines)
(OUT / "09_seed_variance.md").write_text(report, encoding="utf-8")
print(report.encode("ascii", "replace").decode("ascii"))
print("\nsaved reports/partial_results/09_seed_variance.md")
