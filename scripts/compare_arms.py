"""H1: does SNR conditioning beat the blind model? Paired per-tile Wilcoxon test + Holm correction
across SNR levels + cluster-robust CI (clusters = overlap groups), per configs/protocol.json.

PRELIMINARY: protocol specifies 4 folds x 3 seeds per arm; we have 4 folds x 1 seed. This is a first
real statistical pass, not the final frozen-protocol result (which needs 2 more seeds per fold).

Run: python scripts/compare_arms.py
"""
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval.stats import cluster_robust_ci, exact_wilcoxon, holm

DB = "db/results_pilot_full.sqlite"
OUT = Path("reports/partial_results")
OUT.mkdir(parents=True, exist_ok=True)

OVERLAP_GROUPS = json.load(open("configs/splits.json"))["overlap_groups"]
CLUSTER_OF = {}
for g in OVERLAP_GROUPS:
    for t in g:
        CLUSTER_OF[t] = g[0]

con = sqlite3.connect(DB)
cur = con.cursor()

# per-tile PSNR, region=all, gaussian noise, every SNR level, for the conditioned and blind arms
q = """
select r.name, m.tile, m.snr_db, m.value
from metrics m join runs r on m.run_id = r.run_id
where m.metric='psnr' and m.region='all' and m.noise_type='gaussian' and m.band is null
and (r.name like 'full_medium_fold%_seed0' or r.name like 'e1_blind_fold%_seed0')
"""
cond, blind = {}, {}
for name, tile, snr, val in cur.execute(q):
    d = cond if name.startswith("full_medium") else blind
    d.setdefault(float(snr), {})[tile] = val

snr_levels = sorted(cond)
n_tiles_cond = len({t for d in cond.values() for t in d})
n_tiles_blind = len({t for d in blind.values() for t in d})

rows = []
for snr in snr_levels:
    tiles = sorted(set(cond[snr]) & set(blind[snr]))
    x = [cond[snr][t] for t in tiles]
    y = [blind[snr][t] for t in tiles]
    diffs = [a - b for a, b in zip(x, y)]
    wt = exact_wilcoxon(x, y)
    diff_by_tile = {t: [d] for t, d in zip(tiles, diffs)}
    cluster_of = {t: CLUSTER_OF.get(t, t) for t in tiles}
    mean_diff, lo, hi = cluster_robust_ci(diff_by_tile, cluster_of)
    rows.append({
        "snr_db": snr, "n_tiles": len(tiles),
        "mean_cond": sum(x) / len(x), "mean_blind": sum(y) / len(y),
        "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
        "p_raw": wt["p"],
    })

p_adj = holm([r["p_raw"] for r in rows])
for r, p in zip(rows, p_adj):
    r["p_holm"] = p
    r["significant"] = p < 0.05 and r["mean_diff"] > 0

# ---------------------------------------------------------------------------
# plain-language report
# ---------------------------------------------------------------------------
lines = []
lines.append("# H1 result: does telling the model the noise level help? (PRELIMINARY)")
lines.append("")
lines.append(f"4 folds, 1 seed each, {n_tiles_cond} tiles total (every tile tested exactly once, "
              "in whichever fold it belongs to as a test tile). The project's frozen protocol calls "
              "for 3 seeds per fold (12 runs per arm); this is a first pass with 1 seed per fold, "
              "not the final number.")
lines.append("")
lines.append("Per noise level: average PSNR with conditioning, average PSNR without it, the gap, "
              "a 95% confidence range for that gap (accounting for tiles that share the same scene), "
              "and whether the gap is statistically real after correcting for testing 9 noise levels "
              "at once (Holm correction).")
lines.append("")
lines.append("| SNR (dB) | With conditioning | Without (blind) | Gap (dB) | 95% range | Holm p-value | Real effect? |")
lines.append("|---|---|---|---|---|---|---|")
for r in rows:
    sig = "**yes**" if r["significant"] else ("no — too close to call" if r["p_holm"] >= 0.05 else "no, favors blind")
    lines.append(
        f"| {r['snr_db']:g} | {r['mean_cond']:.2f} | {r['mean_blind']:.2f} | "
        f"{r['mean_diff']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_holm']:.3f} | {sig} |"
    )
lines.append("")
n_sig = sum(r["significant"] for r in rows)
lines.append(f"**{n_sig} of {len(rows)} noise levels show a statistically real advantage for conditioning "
             "after correction, at the 95% confidence level, with 1 seed per fold.**")
lines.append("")
lines.append("Caveat: with only 1 seed per fold and 16 tiles total split into small clusters, this test "
             "has limited statistical power — a true effect can fail to reach significance here even if "
             "it is real. The direction (conditioning helps) being consistent across folds, shown "
             "separately in `04_fold_consistency.png`, is itself supporting evidence beyond this table.")

report = "\n".join(lines)
(OUT / "06_h1_statistical_result.md").write_text(report, encoding="utf-8")
print(report)
print("\nsaved reports/partial_results/06_h1_statistical_result.md")
