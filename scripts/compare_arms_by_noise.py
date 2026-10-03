"""H1 across all three noise types: does conditioning beat blind under gaussian,
poisson_gaussian, and correlated_gaussian noise, not just gaussian?

The evaluation harness (scripts/evaluate_run.py) already evaluates every checkpoint on all three
noise types by default -- this was never a missing experiment, just a missing report. This script
reuses the same seed-averaging-then-pairing protocol as scripts/compare_arms.py but repeats the
whole comparison once per noise type, so H1's "gaussian only" scope limitation goes away without
any new training or evaluation runs.

Run: python scripts/compare_arms_by_noise.py
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

NOISE_TYPES = ["gaussian", "poisson_gaussian", "correlated_gaussian"]


def load_arm(cur, noise_type):
    q = """
    select r.name, m.tile, m.snr_db, m.value
    from metrics m join runs r on m.run_id = r.run_id
    where m.metric='psnr' and m.region='all' and m.noise_type=? and m.band is null
    and (r.name like 'full_medium_fold%_seed%' or r.name like 'e1_blind_fold%_seed%')
    """
    cond_raw, blind_raw = {}, {}
    cond_seeds, blind_seeds = {}, {}
    for name, tile, snr, val in cur.execute(q, (noise_type,)):
        arm = "full_medium" if name.startswith("full_medium") else "e1_blind"
        seed = name.rsplit("_seed", 1)[1]
        d_raw = cond_raw if arm == "full_medium" else blind_raw
        d_seeds = cond_seeds if arm == "full_medium" else blind_seeds
        d_raw.setdefault(float(snr), {}).setdefault(tile, []).append(val)
        d_seeds.setdefault(tile, set()).add(seed)
    cond = {snr: {t: sum(v) / len(v) for t, v in tiles.items()} for snr, tiles in cond_raw.items()}
    blind = {snr: {t: sum(v) / len(v) for t, v in tiles.items()} for snr, tiles in blind_raw.items()}
    return cond, blind, cond_seeds, blind_seeds


def compare(cond, blind):
    snr_levels = sorted(set(cond) & set(blind))
    rows = []
    for snr in snr_levels:
        tiles = sorted(set(cond[snr]) & set(blind[snr]))
        if not tiles:
            continue
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
    return rows


con = sqlite3.connect(DB)
cur = con.cursor()

lines = ["# H1 result across all three noise types (not just gaussian)", ""]
lines.append("`scripts/compare_arms.py` only ever looked at gaussian noise, even though every "
              "checkpoint was already evaluated against gaussian, poisson_gaussian, and "
              "correlated_gaussian by `scripts/evaluate_run.py` (its defaults cover all three). "
              "No new training or evaluation runs were needed for this -- the data was already in "
              "`db/results_pilot_full.sqlite`, it just had never been reported this way.")
lines.append("")

summary = []
for nt in NOISE_TYPES:
    cond, blind, cond_seeds, blind_seeds = load_arm(cur, nt)
    rows = compare(cond, blind)
    seed_counts_cond = sorted({len(s) for s in cond_seeds.values()}) if cond_seeds else []
    seed_counts_blind = sorted({len(s) for s in blind_seeds.values()}) if blind_seeds else []
    status = "FINAL" if seed_counts_cond == [3] and seed_counts_blind == [3] else "PARTIAL"
    lines.append(f"## {nt} ({status}: cond seeds/fold {seed_counts_cond}, blind seeds/fold {seed_counts_blind})")
    lines.append("")
    lines.append("| SNR (dB) | Conditioned | Blind | Gap (dB) | 95% range | Holm p | Real effect? |")
    lines.append("|---|---|---|---|---|---|---|")
    n_sig = 0
    for r in rows:
        sig = "**yes**" if r["significant"] else ("no" if r["p_holm"] >= 0.05 else "no, favors blind")
        n_sig += r["significant"]
        lines.append(
            f"| {r['snr_db']:g} | {r['mean_cond']:.2f} | {r['mean_blind']:.2f} | "
            f"{r['mean_diff']:+.2f} | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_holm']:.3f} | {sig} |"
        )
    lines.append("")
    lines.append(f"**{n_sig} of {len(rows)} noise levels significant for {nt}.**")
    lines.append("")
    summary.append((nt, status, n_sig, len(rows)))

lines.append("## Cross-noise-type summary")
lines.append("")
lines.append("| Noise type | Status | Significant SNR levels |")
lines.append("|---|---|---|")
for nt, status, n_sig, n in summary:
    lines.append(f"| {nt} | {status} | {n_sig}/{n} |")
lines.append("")
lines.append("If the same extremes-significant / middle-null pattern holds across all three noise "
              "types, that is evidence the H1 effect is about degradation severity generally, not "
              "an artifact of the gaussian noise model specifically. If it does not hold, that is "
              "itself worth reporting rather than hiding.")

report = "\n".join(lines)
(OUT / "07_h1_by_noise_type.md").write_text(report, encoding="utf-8")
print(report)
print("\nsaved reports/partial_results/07_h1_by_noise_type.md")
