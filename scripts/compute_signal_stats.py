"""Compute per-tile, per-band signal statistics (P_b etc.) into data/processed/signal_stats.json.

P_b is the mean square over valid pixels of the tile (not 0 and not saturated at 65535), computed
once per tile and band. Prints how much the exclusion changes P_b compared with a naive mean square.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bands import BANDS
from src.degrade.stats import band_signal_stats, write_signal_stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    processed = Path(args.processed_dir)
    tiles = [r["tile"] for r in json.loads((processed / "tiles_meta.json").read_text())["tiles"]]

    stats = {}
    print(f"{'tile':12s} {'min valid frac':>14s} {'max naive/valid P_b shift (dB)':>32s}  band")
    for tile in tiles:
        image = np.load(processed / "tiles" / f"{tile}_image.npy", mmap_mode="r")
        s = band_signal_stats(image)
        stats[tile] = s
        shift = 10 * np.log10(np.array(s["power_naive"]) / np.array(s["power"]))
        worst = int(np.argmax(np.abs(shift)))
        print(f"{tile:12s} {min(s['valid_fraction']):14.4f} {shift[worst]:32.2f}  {BANDS[worst]}")

    out = processed / "signal_stats.json"
    write_signal_stats(out, stats)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
