"""Evaluate classical reference restorers (identity, Gaussian smoothing) on the fixed test noise.

For each tile, noise type and SNR the fixed noisy tile (src.degrade.noise.test_noisy_tile) is restored
and scored on the full tile by region, plus per-ship instance metrics. Results go to a results database
(db/results_schema.sql) and a summary is printed. No training is involved.
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
from src.eval.baselines import gaussian_filter, identity
from src.eval.metrics import instance_metrics, region_masks, tile_metrics

DEFAULT_TILES = ["toulon", "brest1", "rotterdam1", "marseille", "portsmouth", "panama"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 10, 20, 30, 40])
    ap.add_argument("--noise-types", nargs="+", default=["gaussian", "correlated_gaussian"])
    ap.add_argument("--sigmas", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--out", default="db/results_baselines.sqlite")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out, processed = Path(args.out), Path(args.processed_dir)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; use --overwrite to replace it")
        out.unlink()

    stats = load_signal_stats(processed / "signal_stats.json")
    peaks = {t: s["peak"] for t, s in json.loads((processed / "signal_stats.json").read_text())["tiles"].items()}
    centroids = load_ship_centroids(Path(args.raw_dir))

    predictors = {"identity": identity}
    for sg in args.sigmas:
        predictors[f"gauss{sg:g}"] = (lambda x, sg=sg: gaussian_filter(x, sg))

    conn = results.connect(out)
    run_ids = {name: results.add_run(conn, f"baseline_{name}", name, notes="classical reference, no training")
               for name in predictors}

    t0 = time.time()
    for tile in args.tiles:
        clean = np.load(processed / "tiles" / f"{tile}_image.npy")
        ship = np.load(processed / "tiles" / f"{tile}_ship.npy")
        water = np.load(processed / "tiles" / f"{tile}_water.npy")
        regions = region_masks(ship, water)
        s = stats[tile]
        ids = [f"{tile}_s{i:03d}" for i in range(len(centroids[tile]))]
        for noise_type in args.noise_types:
            for snr in args.snr:
                noisy, _ = test_noisy_tile(clean, s["power"], s["mean"], tile, snr, noise_type)
                for name, fn in predictors.items():
                    pred = fn(noisy)
                    results.add_metrics(conn, run_ids[name], tile, noise_type, snr,
                                        tile_metrics(clean, pred, peaks[tile], regions))
                    inst = instance_metrics(clean, pred, peaks[tile], centroids[tile], ship)
                    for metric, values in inst.items():
                        results.add_instance_metrics(conn, run_ids[name], noise_type, snr, metric, dict(zip(ids, values)))
                print(f"[{time.time() - t0:6.0f}s] {tile:11s} {noise_type:19s} {snr:5.1f} dB done", flush=True)
    conn.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
