"""Evaluate the BM3D classical baseline on the fixed test noise (gaussian only -- BM3D assumes
i.i.d. noise, so correlated_gaussian is out of scope for it). Same tiles/metrics/results-DB
pattern as scripts/eval_baselines.py (identity, gaussian smoothing); this is the slow one
(~50s per band per tile/SNR on CPU), so it's a separate script with its own output DB rather
than folded into eval_baselines.py.

Default tiles/SNR are the original 6-tile/5-SNR smoke test. To match H1's own protocol exactly
(all 16 tiles, all 9 SNR levels) so the BM3D-vs-learned-model comparison is finally apples-to-
apples, pass --tiles (all 16, see configs/splits.json) and --snr 0 5 10 15 20 25 30 35 40. At
~50s/band x 12 bands = ~10 min/(tile, SNR) that is ~24h of CPU time -- too long for one Kaggle
session (budgeted ~9h), so this is resumable: conditions already present for the run name are
skipped, same pattern as scripts/evaluate_run.py, and --time-budget-hours stops cleanly with time
left on the clock rather than letting a session get killed mid-tile. Re-run (same --out, same
--run-name) across however many sessions it takes; it always only adds what's still missing.

Run: python scripts/eval_bm3d.py --out db/results_bm3d.sqlite
Matched protocol: python scripts/eval_bm3d.py --out db/results_bm3d_matched.sqlite \
    --run-name baseline_bm3d_matched --tiles <all 16> --snr 0 5 10 15 20 25 30 35 40 \
    --time-budget-hours 8.0
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.tiles import load_ship_centroids
from src.degrade.noise import test_noisy_tile
from src.degrade.stats import load_signal_stats
from src.eval import results
from src.eval.baselines import bm3d_denoise
from src.eval.metrics import instance_metrics, region_masks, tile_metrics

DEFAULT_TILES = ["toulon", "brest1", "rotterdam1", "marseille", "portsmouth", "panama"]

ALL_TILES = ["brest1", "marseille", "panama", "portsmouth", "rome", "rotterdam1", "rotterdam2",
             "rotterdam3", "southampton", "suez1", "suez2", "suez3", "suez4", "suez5", "suez6",
             "toulon"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 10, 20, 30, 40])
    ap.add_argument("--run-name", default="baseline_bm3d")
    ap.add_argument("--out", default="db/results_bm3d.sqlite")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--time-budget-hours", type=float, default=None,
                     help="stop cleanly (not mid-combo) once this much wall time has elapsed, "
                          "leaving whatever's left for the next session")
    ap.add_argument("--overwrite", action="store_true", help="wipe and restart instead of resuming")
    args = ap.parse_args()
    if args.tiles == ["all"]:
        args.tiles = ALL_TILES

    out, processed = Path(args.out), Path(args.processed_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite and out.exists():
        out.unlink()

    stats = load_signal_stats(processed / "signal_stats.json")
    peaks = {t: s["peak"] for t, s in json.loads((processed / "signal_stats.json").read_text())["tiles"].items()}
    centroids = load_ship_centroids(Path(args.raw_dir))

    conn = results.connect(out)
    row = conn.execute("SELECT run_id FROM runs WHERE name = ?", (args.run_name,)).fetchone()
    run_id = row[0] if row else results.add_run(
        conn, args.run_name, "bm3d", notes="classical reference (BM3D), no training, gaussian noise only")
    done = {(t, float(s)) for t, s in conn.execute(
        "SELECT DISTINCT tile, snr_db FROM metrics WHERE run_id = ? AND noise_type='gaussian'", (run_id,))}

    t0 = time.time()
    n_done, n_skipped = 0, 0
    stop = False
    for tile in args.tiles:
        if stop:
            break
        todo = [s for s in args.snr if (tile, float(s)) not in done]
        if not todo:
            n_skipped += len(args.snr)
            continue
        clean = np.load(processed / "tiles" / f"{tile}_image.npy")
        ship = np.load(processed / "tiles" / f"{tile}_ship.npy")
        water = np.load(processed / "tiles" / f"{tile}_water.npy")
        regions = region_masks(ship, water)
        s = stats[tile]
        ids = [f"{tile}_s{i:03d}" for i in range(len(centroids[tile]))]
        for snr in todo:
            if args.time_budget_hours and (time.time() - t0) / 3600 > args.time_budget_hours:
                print(f"[{time.time() - t0:6.0f}s] time budget reached, stopping before {tile} {snr:g} dB", flush=True)
                stop = True
                break
            noisy, params = test_noisy_tile(clean, s["power"], s["mean"], tile, snr, "gaussian")
            pred = bm3d_denoise(noisy, params["sigma_read"])
            results.add_metrics(conn, run_id, tile, "gaussian", snr, tile_metrics(clean, pred, peaks[tile], regions))
            inst = instance_metrics(clean, pred, peaks[tile], centroids[tile], ship)
            for metric, values in inst.items():
                results.add_instance_metrics(conn, run_id, "gaussian", snr, metric, dict(zip(ids, values)))
            n_done += 1
            print(f"[{time.time() - t0:6.0f}s] {tile:11s} gaussian {snr:5.1f} dB done", flush=True)
    conn.close()
    remaining = sum(1 for t in args.tiles for s in args.snr if (t, float(s)) not in done) - n_done
    print(f"\nwrote {out}: {n_done} new, {n_skipped} already done, {max(remaining,0)} left for next session")


if __name__ == "__main__":
    main()
