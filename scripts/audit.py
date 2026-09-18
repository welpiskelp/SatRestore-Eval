"""Audit the downloaded S2-SHIPS tiles before any preprocessing/modeling work.

Checks, per tile: band count/shape/dtype, per-band pixel stats, a rough
nodata/cloud estimate (no SCL band or nodata metadata is shipped with this
dataset, so this is a pixel-saturation heuristic, not a real cloud mask),
and presence of the water mask / ship-instance mask / COCO annotations.

Confirmed empirically (see scripts/download_dataset.py's extraction and a
rasterio spot-check) before writing this:
- Each tile is a directory under dataset_tif/<tile>/ with one single-band
  GeoTIFF per band (not one 12-band file), named like
  "..._Sentinel-2_L2A_B0X_(Raw).tiff" (B01's filename lacks the parens).
- 12 bands: B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12. All uint16,
  shape (938, 1783), CRS EPSG:3857, dataset.nodata is None for all of them.
- Water mask: dataset_tif/<tile>/<tile>_water.tif.
- Ship mask: dataset_npy/<NN>_mask_<tile>.npy.
- COCO annotations: coco-s2ships.json, images keyed by "<tile>_rgb.png".
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import rasterio

BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
EXPECTED_SHAPE = (938, 1783)
EXPECTED_DTYPE = "uint16"


def discover_tiles(data_dir: Path) -> list[str]:
    tif_dir = data_dir / "dataset_tif"
    return sorted(p.name for p in tif_dir.iterdir() if p.is_dir())


def find_band_file(tile_dir: Path, band: str) -> Path | None:
    matches = [
        f for f in tile_dir.glob("*.tiff")
        if re.search(rf"_{band}(_|\()", f.name) and "Raw" in f.name
    ]
    return matches[0] if matches else None


def estimate_nodata_fraction(arr: np.ndarray) -> float:
    # No SCL band or nodata metadata is shipped with this dataset; this is
    # a pixel-saturation heuristic (fraction at 0 or at the uint16 max),
    # not a real cloud/nodata mask.
    return float(np.mean((arr == 0) | (arr == 65535)))


def audit_tile(data_dir: Path, tile: str) -> dict:
    tile_dir = data_dir / "dataset_tif" / tile
    result = {"tile": tile, "bands": {}, "issues": []}

    for band in BANDS:
        f = find_band_file(tile_dir, band)
        if f is None:
            result["issues"].append(f"missing band {band}")
            continue
        with rasterio.open(f) as ds:
            arr = ds.read(1)
            band_info = {
                "shape": list(ds.shape),
                "dtype": ds.dtypes[0],
                "min": int(arr.min()),
                "max": int(arr.max()),
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "nodata_frac_est": estimate_nodata_fraction(arr),
            }
            if tuple(ds.shape) != EXPECTED_SHAPE:
                result["issues"].append(f"{band} shape {ds.shape} != {EXPECTED_SHAPE}")
            if ds.dtypes[0] != EXPECTED_DTYPE:
                result["issues"].append(f"{band} dtype {ds.dtypes[0]} != {EXPECTED_DTYPE}")
            result["bands"][band] = band_info

    result["band_count"] = len(result["bands"])
    if result["band_count"] != len(BANDS):
        result["issues"].append(f"found {result['band_count']}/{len(BANDS)} expected bands")

    water_mask = tile_dir / f"{tile}_water.tif"
    result["has_water_mask"] = water_mask.exists()
    if not result["has_water_mask"]:
        result["issues"].append("missing water mask")

    npy_masks = list((data_dir / "dataset_npy").glob(f"*_mask_{tile}.npy"))
    result["has_ship_mask_npy"] = len(npy_masks) == 1
    if not result["has_ship_mask_npy"]:
        result["issues"].append(f"expected 1 ship-mask npy for {tile}, found {len(npy_masks)}")

    return result


def load_coco_annotation_counts(data_dir: Path) -> dict:
    coco_path = data_dir / "coco-s2ships.json"
    coco = json.loads(coco_path.read_text())
    image_id_to_tile = {}
    for img in coco["images"]:
        m = re.match(r"(.+)_rgb\.png$", img["file_name"])
        if m:
            image_id_to_tile[img["id"]] = m.group(1)

    counts = {tile: {"annotation_records": 0, "segmentation_polygons": 0} for tile in image_id_to_tile.values()}
    for ann in coco["annotations"]:
        tile = image_id_to_tile.get(ann["image_id"])
        if tile is None:
            continue
        counts[tile]["annotation_records"] += 1
        counts[tile]["segmentation_polygons"] += len(ann.get("segmentation", []))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--out", default="reports/audit_report.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    tiles = discover_tiles(data_dir)
    print(f"discovered {len(tiles)} tiles: {tiles}")

    ship_counts = load_coco_annotation_counts(data_dir)

    results = []
    n_clean = 0
    for tile in tiles:
        r = audit_tile(data_dir, tile)
        r["ship_annotations"] = ship_counts.get(tile, {"annotation_records": 0, "segmentation_polygons": 0})
        results.append(r)
        status = "OK" if not r["issues"] else "ISSUES: " + "; ".join(r["issues"])
        print(f"[{tile}] bands={r['band_count']}/{len(BANDS)} water_mask={r['has_water_mask']} "
              f"ship_mask_npy={r['has_ship_mask_npy']} "
              f"annotations={r['ship_annotations']['annotation_records']} "
              f"polygons={r['ship_annotations']['segmentation_polygons']} -> {status}")
        if not r["issues"]:
            n_clean += 1

    total_annotation_records = sum(r["ship_annotations"]["annotation_records"] for r in results)
    total_polygons = sum(r["ship_annotations"]["segmentation_polygons"] for r in results)

    print(f"\n{n_clean}/{len(tiles)} tiles pass all checks")
    print(f"total COCO annotation records: {total_annotation_records}")
    print(f"total segmentation polygons (candidate 'ship instance' count): {total_polygons}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "tiles": results,
        "summary": {
            "n_tiles": len(tiles),
            "n_clean": n_clean,
            "total_annotation_records": total_annotation_records,
            "total_segmentation_polygons": total_polygons,
        },
    }, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
