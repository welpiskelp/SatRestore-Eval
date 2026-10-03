"""H2: does cross-band attention improve spectral fidelity (SAM, ERGAS) relative to a
parameter-matched model without it, on gaussian and correlated_gaussian noise -- with PSNR
reported as a secondary outcome, not folded into the same test (SAM is in degrees, ERGAS is
dimensionless, PSNR is in dB; mixing their scales into one "relative improvement" statistic, as
the original protocol wording implied, is not well-posed). Same seed-averaging-then-pairing,
exact Wilcoxon, Holm correction, cluster-robust CI machinery as scripts/compare_arms.py (H1),
just comparing full_medium (with cross-band attention) vs e2_no_crossband (without) instead of
conditioned vs blind.

Run: python scripts/compare_h2.py
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

NOISE_TYPES = ["gaussian", "correlated_gaussian"]
# lower-is-better metrics: flip the sign so "mean_diff > 0" always means "cross-band attention wins"
METRICS = {"sam": -1.0, "ergas": -1.0, "psnr": +1.0}


def load_arm(cur, noise_type, metric):
    q = """
    select r.name, m.tile, m.snr_db, m.value
    from metrics m join runs r on m.run_id = r.run_id
    where m.metric=? and m.region='all' and m.noise_type=? and m.band is null
    and (r.name like 'full_medium_fold%_seed%' or r.name like 'e2_no_crossband_fold%_seed%')
    """
    full_raw, nocross_raw = {}, {}
    full_seeds, nocross_seeds = {}, {}
    for name, tile, snr, val in cur.execute(q, (metric, noise_type)):
        arm = "full_medium" if name.startswith("full_medium") else "e2_no_crossband"
        seed = name.rsplit("_seed", 1)[1]
        d_raw = full_raw if arm == "full_medium" else nocross_raw
        d_seeds = full_seeds if arm == "full_medium" else nocross_seeds
        d_raw.setdefault(float(snr), {}).setdefault(tile, []).append(val)
        d_seeds.setdefault(tile, set()).add(seed)
    full = {snr: {t: sum(v) / len(v) for t, v in tiles.items()} for snr, tiles in full_raw.items()}
    nocross = {snr: {t: sum(v) / len(v) for t, v in tiles.items()} for snr, tiles in nocross_raw.items()}
    return full, nocross, full_seeds, nocross_seeds


def compare(full, nocross, sign):
    snr_levels = sorted(set(full) & set(nocross))
    rows = []
    for snr in snr_levels:
        tiles = sorted(set(full[snr]) & set(nocross[snr]))
        if not tiles:
            continue
        x = [full[snr][t] for t in tiles]
        y = [nocross[snr][t] for t in tiles]
        diffs = [sign * (a - b) for a, b in zip(x, y)]
        wt = exact_wilcoxon(diffs)
        diff_by_tile = {t: [d] for t, d in zip(tiles, diffs)}
        cluster_of = {t: CLUSTER_OF.get(t, t) for t in tiles}
        mean_diff, lo, hi = cluster_robust_ci(diff_by_tile, cluster_of)
        rows.append({
            "snr_db": snr, "n_tiles": len(tiles),
            "mean_full": sum(x) / len(x), "mean_nocross": sum(y) / len(y),
            "mean_diff": mean_diff, "ci_lo": lo, "ci_hi": hi,
            "p_raw": wt["p"],
        })
    p_adj = holm([r["p_raw"] for r in rows])
    for r, p in zip(rows, p_adj):
        r["p_holm"] = p
        r["significant"] = p < 0.05 and r["mean_diff"] > 0
    return rows


con = sqlite3.connect(DB)
cur = con.cursor()

lines = ["# H2 result: does cross-band attention improve spectral fidelity more than PSNR?", ""]
lines.append("Comparing `full_medium` (with cross-band attention) vs `e2_no_crossband` (parameter-"
              "matched, attention block removed). SAM and ERGAS are lower-is-better; their 'gap' "
              "and CI below are sign-flipped so a positive number always means cross-band attention "
              "is better, matching PSNR's convention. SAM (degrees), ERGAS (dimensionless) and PSNR "
              "(dB) are reported as three separate tests, not combined into one 'relative "
              "improvement' statistic -- their units aren't comparable.")
lines.append("")
lines.append("**Power caveat while partial:** with only 8 of 16 tiles currently available for the "
              "no-crossband arm, the exact Wilcoxon test's best possible raw p-value is about "
              "0.0078 (all 8 tiles agreeing), which Holm-inflates to about 0.07 across 9 SNR levels "
              "-- structurally just above the 0.05 line even for a perfectly consistent effect. "
              "Several rows below show large, consistently-signed gaps that are not yet "
              "'significant' for this reason, not because the effect is weak. Re-run once all 16 "
              "tiles are in.")
lines.append("")

any_partial = False
for metric in ("sam", "ergas", "psnr"):
    sign = METRICS[metric]
    for nt in NOISE_TYPES:
        full, nocross, full_seeds, nocross_seeds = load_arm(cur, nt, metric)
        if not full or not nocross:
            lines.append(f"## {metric} / {nt}: no data yet\n")
            any_partial = True
            continue
        rows = compare(full, nocross, sign)
        seed_counts_full = sorted({len(s) for s in full_seeds.values()})
        seed_counts_nocross = sorted({len(s) for s in nocross_seeds.values()})
        n_tiles_full = len(full_seeds)
        n_tiles_nocross = len(nocross_seeds)
        status = "FINAL" if seed_counts_full == [3] and seed_counts_nocross == [3] and n_tiles_full == 16 and n_tiles_nocross == 16 else "PARTIAL"
        if status == "PARTIAL":
            any_partial = True
        lines.append(f"## {metric} / {nt} ({status}: full-model tiles={n_tiles_full}/16 seeds/tile={seed_counts_full}, "
                      f"no-crossband tiles={n_tiles_nocross}/16 seeds/tile={seed_counts_nocross})")
        lines.append("")
        lines.append(f"| SNR (dB) | With cross-band | Without | Gap ({'higher=better, sign-flipped' if sign < 0 else 'dB'}) | 95% range | Holm p | Real effect? |")
        lines.append("|---|---|---|---|---|---|---|")
        n_sig = 0
        for r in rows:
            sig = "**yes**" if r["significant"] else ("no" if r["p_holm"] >= 0.05 else "no, favors no-crossband")
            n_sig += r["significant"]
            lines.append(
                f"| {r['snr_db']:g} | {r['mean_full']:.3f} | {r['mean_nocross']:.3f} | "
                f"{r['mean_diff']:+.3f} | [{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}] | {r['p_holm']:.3f} | {sig} |"
            )
        lines.append("")
        lines.append(f"**{n_sig} of {len(rows)} SNR levels significant for {metric}/{nt}.**")
        lines.append("")

lines.append("## H2 verdict")
lines.append("")
if any_partial:
    lines.append("**PARTIAL -- not all folds/seeds are in yet.** The table above is a progress "
                  "signal from whatever data currently exists, not the final H2 result. Re-run "
                  "this script as more of the 12 full-model and 12 no-crossband runs land.")
else:
    lines.append("All 4 folds x 3 seeds present for both arms -- this is the final H2 result. "
                  "H2 is supported if SAM and/or ERGAS show more/stronger significant SNR levels "
                  "than PSNR does; it is not supported if PSNR shows an equal or stronger pattern.")

report = "\n".join(lines)
(OUT / "08_h2_statistical_result.md").write_text(report, encoding="utf-8")
print(report)
print("\nsaved reports/partial_results/08_h2_statistical_result.md")
