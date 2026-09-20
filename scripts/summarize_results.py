"""Summarise a results database: global versus ship-region quality, and per-ship contrast error with
cluster-robust 95% intervals (clusters = overlap groups of tiles from the reference database)."""
import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.stats import cluster_robust_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="db/results_baselines.sqlite")
    ap.add_argument("--reference", default="db/reference.sqlite")
    ap.add_argument("--baseline", default="identity", help="model used as the reference for differences")
    args = ap.parse_args()

    db = sqlite3.connect(args.results)
    ref = sqlite3.connect(args.reference)
    cluster = {t: g or t for t, g in ref.execute("SELECT tile, overlap_group FROM tiles")}

    def mean_metric(run, noise, snr, region, metric):
        return db.execute(
            "SELECT AVG(value) FROM metrics m JOIN runs r USING(run_id) WHERE r.model=? AND noise_type=? AND "
            "snr_db=? AND region=? AND metric=? AND band IS NULL", (run, noise, snr, region, metric)).fetchone()[0]

    models = [r[0] for r in db.execute("SELECT model FROM runs ORDER BY run_id")]
    conds = db.execute("SELECT DISTINCT noise_type, snr_db FROM metrics ORDER BY noise_type, snr_db").fetchall()
    n_tiles = db.execute("SELECT COUNT(DISTINCT tile) FROM metrics").fetchone()[0]

    print(f"Mean over {n_tiles} tiles. PSNR in dB against a per-band peak; SAM in degrees.")
    print("gain = PSNR(model) - PSNR(identity); 'ship lag' = global gain - ship gain (positive: the ship region gains less).\n")
    print(f"{'noise':19s} {'SNR':>4s} {'model':>8s} {'PSNR all':>9s} {'PSNR ship':>10s} {'gain all':>9s} {'gain ship':>10s} "
          f"{'ship lag':>9s} {'SAM all':>8s} {'SAM ship':>9s}")
    for noise, snr in conds:
        base_all = mean_metric(args.baseline, noise, snr, "all", "psnr")
        base_ship = mean_metric(args.baseline, noise, snr, "ship", "psnr")
        for m in models:
            pa, ps = mean_metric(m, noise, snr, "all", "psnr"), mean_metric(m, noise, snr, "ship", "psnr")
            ga, gs = pa - base_all, ps - base_ship
            print(f"{noise:19s} {snr:4.0f} {m:>8s} {pa:9.2f} {ps:10.2f} {ga:+9.2f} {gs:+10.2f} {ga - gs:+9.2f} "
                  f"{mean_metric(m, noise, snr, 'all', 'sam'):8.2f} {mean_metric(m, noise, snr, 'ship', 'sam'):9.2f}")
        print()

    print("Per-ship contrast error (normalised by the band peak; lower is better), mean with cluster-robust 95% CI,")
    print(f"and the paired difference to '{args.baseline}' (negative = the model preserves ship contrast better).\n")
    print(f"{'noise':19s} {'SNR':>4s} {'model':>8s} {'contrast error [95% CI]':>28s} {'diff vs base [95% CI]':>30s}")
    for noise, snr in conds:
        per_model = {}
        for m in models:
            rows = db.execute(
                "SELECT i.instance_id, i.value FROM instance_metrics i JOIN runs r USING(run_id) WHERE r.model=? AND "
                "i.noise_type=? AND i.snr_db=? AND i.metric='contrast_error'", (m, noise, snr)).fetchall()
            per_model[m] = {iid: v for iid, v in rows}
        for m in models:
            by_tile = defaultdict(list)
            diff_by_tile = defaultdict(list)
            for iid, v in per_model[m].items():
                tile = iid.rsplit("_s", 1)[0]
                by_tile[tile].append(v)
                if iid in per_model[args.baseline]:
                    diff_by_tile[tile].append(v - per_model[args.baseline][iid])
            mean, lo, hi = cluster_robust_ci(by_tile, {t: cluster[t] for t in by_tile})
            if m == args.baseline:
                dtxt = "-"
            else:
                dm, dlo, dhi = cluster_robust_ci(diff_by_tile, {t: cluster[t] for t in diff_by_tile})
                dtxt = f"{dm:+.4f} [{dlo:+.4f}, {dhi:+.4f}]"
            n = sum(len(v) for v in by_tile.values())
            print(f"{noise:19s} {snr:4.0f} {m:>8s} {mean:9.4f} [{lo:.4f}, {hi:.4f}] n={n:<4d} {dtxt:>30s}")
        print()
    print("Note: with few tiles the number of clusters is small, so the intervals are wide and the t-based")
    print("interval still under-covers somewhat (see scripts/check_stats.py).")


if __name__ == "__main__":
    main()
