"""Convert the raw S2-SHIPS tiles into compact, pickle-free arrays in canonical band order.

Per tile, writes to <out-dir>/tiles/:
  <tile>_image.npy  uint16 (12, H, W), channels in src.data.bands.BANDS order
  <tile>_ship.npy   uint8  (H, W), 1 = ship pixel
  <tile>_water.npy  uint8  (H, W), 1 = water, resampled onto the imagery grid
and a summary <out-dir>/tiles_meta.json.

The npy imagery is checked bit-for-bit against the per-band GeoTIFFs before anything is written.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bands import BANDS, NPY_TO_CANONICAL
from src.data.tiles import (
    discover_tiles, find_band_file, find_ship_npy, load_pickled_npy, load_water_mask, tile_date,
)


def prepare_tile(raw_dir: Path, tile: str, tiles_out: Path) -> dict:
    tile_dir = raw_dir / "dataset_tif" / tile
    d = load_pickled_npy(find_ship_npy(raw_dir, tile))

    data = d["data"]  # (H, W, 12) float64, npy channel order
    if not (np.array_equal(data, np.round(data)) and data.min() >= 0 and data.max() <= 65535):
        raise ValueError(f"{tile}: imagery is not integral uint16-range data")
    image = data.transpose(2, 0, 1)[NPY_TO_CANONICAL].astype(np.uint16)

    for i, band in enumerate(BANDS):
        with rasterio.open(find_band_file(tile_dir, band)) as ds:
            if not np.array_equal(image[i], ds.read(1)):
                raise ValueError(f"{tile}: npy channel for {band} differs from its GeoTIFF")
            if band == "B02":
                bounds, crs = ds.bounds, ds.crs.to_string()

    ship = d["label"][..., 0]
    if not np.isin(ship, (0, 1)).all():
        raise ValueError(f"{tile}: ship label is not binary")
    ship = ship.astype(np.uint8)

    water = load_water_mask(tile_dir / f"{tile}_water.tif", find_band_file(tile_dir, "B02"))
    if water.shape != image.shape[1:] or ship.shape != image.shape[1:]:
        raise ValueError(f"{tile}: mask shapes {water.shape}/{ship.shape} != image {image.shape[1:]}")

    np.save(tiles_out / f"{tile}_image.npy", image)
    np.save(tiles_out / f"{tile}_ship.npy", ship)
    np.save(tiles_out / f"{tile}_water.npy", water)

    ship_px = int(ship.sum())
    return {
        "tile": tile,
        "date": tile_date(tile_dir),
        "shape": list(image.shape),
        "crs": crs,
        "bounds": [bounds.left, bounds.bottom, bounds.right, bounds.top],
        "ship_pixels": ship_px,
        "water_fraction": float(water.mean()),
        "ship_pixels_on_water_fraction": float(water[ship == 1].mean()) if ship_px else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--out-dir", default="data/processed")
    args = ap.parse_args()

    raw_dir, out_dir = Path(args.raw_dir), Path(args.out_dir)
    tiles_out = out_dir / "tiles"
    tiles_out.mkdir(parents=True, exist_ok=True)

    meta = []
    for tile in discover_tiles(raw_dir):
        rec = prepare_tile(raw_dir, tile, tiles_out)
        meta.append(rec)
        print(f"[{tile}] {rec['date']} shape={rec['shape']} ship_px={rec['ship_pixels']} "
              f"water={rec['water_fraction']:.3f} ship_on_water={rec['ship_pixels_on_water_fraction']}")

    (out_dir / "tiles_meta.json").write_text(json.dumps({"bands": BANDS, "tiles": meta}, indent=2))
    print(f"\nwrote {len(meta)} tiles to {tiles_out} and {out_dir / 'tiles_meta.json'}")


if __name__ == "__main__":
    main()
