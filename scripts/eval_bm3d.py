"""Evaluate the BM3D classical baseline on the fixed test noise (gaussian only -- BM3D assumes
i.i.d. noise, so correlated_gaussian is out of scope for it). Same tiles/metrics/results-DB
pattern as scripts/eval_baselines.py (identity, gaussian smoothing); this is the slow one
(~50s per band per tile/SNR on CPU), so it's a separate script with its own output DB rather
than folded into eval_baselines.py.

Run: python scripts/eval_bm3d.py --out db/results_bm3d.sqlite
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", nargs="+", default=DEFAULT_TILES)
    ap.add_argument("--snr", nargs="+", type=float, default=[0, 10, 20, 30, 40])
    ap.add_argument("--out", default="db/results_bm3d.sqlite")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out, processed = Path(args.out), Path(args.processed_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if not args.overwrite:
            sys.exit(f"{out} exists; use --overwrite to replace it")
        out.unlink()

    stats = load_signal_stats(processed / "signal_stats.json")
    peaks = {t: s["peak"] for t, s in json.loads((processed / "signal_stats.json").read_text())["tiles"].items()}
    centroids = load_ship_centroids(Path(args.raw_dir))

    conn = results.connect(out)
    run_id = results.add_run(conn, "baseline_bm3d", "bm3d", notes="classical reference (BM3D), no training, gaussian noise only")

    t0 = time.time()
    for tile in args.tiles:
        clean = np.load(processed / "tiles" / f"{tile}_image.npy")
        ship = np.load(processed / "tiles" / f"{tile}_ship.npy")
        water = np.load(processed / "tiles" / f"{tile}_water.npy")
        regions = region_masks(ship, water)
        s = stats[tile]
        ids = [f"{tile}_s{i:03d}" for i in range(len(centroids[tile]))]
        for snr in args.snr:
            noisy, params = test_noisy_tile(clean, s["power"], s["mean"], tile, snr, "gaussian")
            pred = bm3d_denoise(noisy, params["sigma_read"])
            results.add_metrics(conn, run_id, tile, "gaussian", snr, tile_metrics(clean, pred, peaks[tile], regions))
            inst = instance_metrics(clean, pred, peaks[tile], centroids[tile], ship)
            for metric, values in inst.items():
                results.add_instance_metrics(conn, run_id, "gaussian", snr, metric, dict(zip(ids, values)))
            print(f"[{time.time() - t0:6.0f}s] {tile:11s} gaussian {snr:5.1f} dB done", flush=True)
    conn.close()
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
