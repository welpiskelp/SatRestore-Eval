"""Write the tile-level 4-fold split to configs/splits.json (or --check an existing one).

Inputs (produced earlier): data/processed/tiles_meta.json (bounds, ship pixels) from
scripts/prepare_tiles.py and reports/audit_report.json (ship polygon counts) from scripts/audit.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import make_folds, overlap_groups, validate_folds


def load_inputs(meta_path: Path, audit_path: Path):
    for p in (meta_path, audit_path):
        if not p.exists():
            sys.exit(f"missing {p}; run scripts/prepare_tiles.py and scripts/audit.py first")
    meta = {r["tile"]: r for r in json.loads(meta_path.read_text())["tiles"]}
    audit = {r["tile"]: r for r in json.loads(audit_path.read_text())["tiles"]}
    tiles = sorted(meta)
    bounds = {t: meta[t]["bounds"] for t in tiles}
    ship_pixels = {t: meta[t]["ship_pixels"] for t in tiles}
    polygons = {t: audit[t]["ship_annotations"]["segmentation_polygons"] for t in tiles}
    return tiles, bounds, polygons, ship_pixels


def add_stats(folds, polygons, ship_pixels):
    for f in folds:
        f["stats"] = {
            split: {
                "tiles": len(f[split]),
                "ship_polygons": sum(polygons[t] for t in f[split]),
                "ship_pixels": sum(ship_pixels[t] for t in f[split]),
            }
            for split in ("train", "val", "test")
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/processed/tiles_meta.json")
    ap.add_argument("--audit", default="reports/audit_report.json")
    ap.add_argument("--out", default="configs/splits.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--n-val", type=int, default=2)
    ap.add_argument("--check", action="store_true", help="validate the existing --out file and exit")
    args = ap.parse_args()

    tiles, bounds, polygons, ship_pixels = load_inputs(Path(args.meta), Path(args.audit))
    groups = overlap_groups(bounds)
    print("overlap groups:", [g for g in groups if len(g) > 1])

    out = Path(args.out)
    if args.check:
        saved = json.loads(out.read_text())
        validate_folds(saved["folds"], tiles, groups)
        print(f"{out}: valid ({len(saved['folds'])} folds)")
        return

    folds, groups = make_folds(tiles, bounds, polygons, ship_pixels, args.n_folds, args.n_val, args.seed)
    add_stats(folds, polygons, ship_pixels)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "seed": args.seed,
        "n_folds": args.n_folds,
        "overlap_groups": [g for g in groups if len(g) > 1],
        "folds": folds,
    }, indent=2))

    print(f"{'fold':>4} {'split':>5} {'tiles':>5} {'polygons':>8} {'ship_px':>8}  members")
    for f in folds:
        for split in ("train", "val", "test"):
            s = f["stats"][split]
            print(f"{f['fold']:>4} {split:>5} {s['tiles']:>5} {s['ship_polygons']:>8} {s['ship_pixels']:>8}  "
                  f"{', '.join(f[split]) if split != 'train' else '(rest)'}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
