"""Build the patch index (128x128, stride 64) over the prepared tiles.

Writes data/processed/patch_index.csv with one row per patch: position, ship pixels, ship
instances (polygon centroids inside the patch), water fraction and saturation/zero fractions.
Imagery is not copied; patches are sliced from data/processed/tiles/ on the fly.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.patches import FIELDS, PATCH, STRIDE, clip_centroids, covered_extent, tile_patch_rows
from src.data.tiles import load_ship_centroids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--out", default="data/processed/patch_index.csv")
    args = ap.parse_args()

    processed, raw = Path(args.processed_dir), Path(args.raw_dir)
    tiles = [r["tile"] for r in json.loads((processed / "tiles_meta.json").read_text())["tiles"]]
    centroids = load_ship_centroids(raw)

    rows = []
    lost_instances = lost_pixels = total_instances = total_pixels = 0
    for tile in tiles:
        image = np.load(processed / "tiles" / f"{tile}_image.npy", mmap_mode="r")
        ship = np.load(processed / "tiles" / f"{tile}_ship.npy")
        water = np.load(processed / "tiles" / f"{tile}_water.npy")
        c = centroids[tile]
        rows += tile_patch_rows(tile, image, ship, water, c)

        h, w = image.shape[1:]
        cov_h, cov_w = covered_extent(h, w)
        cc = clip_centroids(c, h, w)
        outside = (cc[:, 0] >= cov_w) | (cc[:, 1] >= cov_h)
        lost_instances += int(outside.sum())
        total_instances += len(c)
        lost_pixels += int(ship.sum() - ship[:cov_h, :cov_w].sum())
        total_pixels += int(ship.sum())

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    n = len(rows)
    print(f"{n} patches ({PATCH}x{PATCH}, stride {STRIDE}) over {len(tiles)} tiles -> {out}")
    print(f"patches with ship pixels: {sum(r['ship_pixels'] > 0 for r in rows)}")
    print(f"patches with >5% saturated pixels (any band): {sum(r['sat_any_frac'] > 0.05 for r in rows)}")
    print(f"patches with any zero pixel: {sum(r['zero_any_frac'] > 0 for r in rows)}")
    print(f"ship instances outside every patch: {lost_instances}/{total_instances}; "
          f"ship pixels outside every patch: {lost_pixels}/{total_pixels}")


if __name__ == "__main__":
    main()
