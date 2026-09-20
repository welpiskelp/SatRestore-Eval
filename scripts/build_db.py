"""Build the reference database db/reference.sqlite from the prepared files.

Inputs (all produced by earlier scripts): data/processed/{tiles_meta,signal_stats}.json,
data/processed/patch_index.csv, reports/audit_report.json, configs/splits.json and the raw COCO file.
Purely derived data, so an existing file is simply replaced. Results live in separate databases.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from src.data import refdb
from src.data.patches import PATCH
from src.data.tiles import load_ship_centroids
from src.degrade.noise import NOISE_TYPES, TEST_BASE_SEED, TEST_SNR_LEVELS, degradation_seed, noise_params
from src.eval.contrast import cnr, ship_contrast


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="db/reference.sqlite")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--splits", default="configs/splits.json")
    ap.add_argument("--audit", default="reports/audit_report.json")
    args = ap.parse_args()

    out, processed = Path(args.out), Path(args.processed_dir)
    if out.exists():
        out.unlink()

    meta = json.loads((processed / "tiles_meta.json").read_text())["tiles"]
    stats = json.loads((processed / "signal_stats.json").read_text())["tiles"]
    splits = json.loads(Path(args.splits).read_text())
    audit = {r["tile"]: r["ship_annotations"]["segmentation_polygons"]
             for r in json.loads(Path(args.audit).read_text())["tiles"]}
    patches = list(csv.DictReader(open(processed / "patch_index.csv")))
    centroids = load_ship_centroids(Path(args.raw_dir))
    shapes = {m["tile"]: (m["shape"][1], m["shape"][2]) for m in meta}

    contrast = {}
    for m in meta:
        t = m["tile"]
        tiles_dir = processed / "tiles"
        contrast[t] = ship_contrast(np.load(tiles_dir / f"{t}_image.npy", mmap_mode="r"),
                                    np.load(tiles_dir / f"{t}_ship.npy"), np.load(tiles_dir / f"{t}_water.npy"))

    out.parent.mkdir(parents=True, exist_ok=True)
    conn = refdb.connect(out)
    refdb.create_schema(conn)
    with conn:
        refdb.insert_tiles(conn, meta, audit, splits["overlap_groups"])
        refdb.insert_band_stats(conn, stats)
        refdb.insert_ship_contrast(conn, contrast)
        refdb.insert_folds(conn, splits)
        refdb.insert_patches(conn, patches)
        refdb.insert_ship_instances(conn, centroids, shapes)
        refdb.insert_test_degradations(conn, stats)

    q = lambda sql: conn.execute(sql).fetchone()[0]
    counts = {t: q(f"SELECT COUNT(*) FROM {t}") for t in
              ("tiles", "tile_band_stats", "tile_ship_contrast", "folds", "patches", "ship_instances",
               "degradations")}
    print("row counts:", counts)

    # Consistency checks.
    assert conn.execute("PRAGMA foreign_key_check").fetchall() == [], "foreign key violations"
    assert q("PRAGMA integrity_check") == "ok"
    assert counts["tiles"] == 16 and counts["tile_band_stats"] == 16 * 12 and counts["folds"] == 16 * 4
    assert counts["tile_ship_contrast"] == 16 * 12
    assert counts["degradations"] == 16 * len(NOISE_TYPES) * len(TEST_SNR_LEVELS)
    for tile, n in audit.items():
        assert q(f"SELECT COUNT(*) FROM ship_instances WHERE tile='{tile}'") == n, f"{tile}: instance count"
    assert counts["ship_instances"] == q("SELECT SUM(ship_instances) FROM tiles") == 1053
    assert q("SELECT COUNT(*) FROM (SELECT tile FROM folds WHERE split='test' GROUP BY tile HAVING COUNT(*)=1)") == 16

    # Home patch margin: distance from the ship centroid to the nearest border of its home patch.
    margins = conn.execute(
        "SELECT s.cx - p.x, p.x + ? - s.cx, s.cy - p.y, p.y + ? - s.cy "
        "FROM ship_instances s JOIN patches p ON p.patch_id = s.home_patch_id", (PATCH, PATCH)).fetchall()
    worst = [min(m) for m in margins]
    print(f"home-patch margin (px from centroid to nearest patch border): min {min(worst):.1f}, "
          f"ships with margin < 32 px: {sum(w < 32 for w in worst)} of {len(worst)}")

    # Reproducibility: every stored degradation must match the parameters recomputed from the stored stats.
    for tile, noise_type, snr, seed, sfrac, pj in conn.execute(
            "SELECT tile, noise_type, snr_db, seed, shot_fraction, params_json FROM degradations"):
        rows = conn.execute("SELECT power, mean FROM tile_band_stats WHERE tile=? ORDER BY rowid", (tile,)).fetchall()
        power, mean = [r[0] for r in rows], [r[1] for r in rows]
        assert seed == degradation_seed(TEST_BASE_SEED, tile, round(snr * 100))
        assert json.loads(pj) == noise_params(power, mean, snr, noise_type, seed, 0.5), (tile, noise_type, snr)
    print(f"all {counts['degradations']} stored degradations reproduce from the stored tile statistics")

    # Ship visibility: contrast-to-noise per pixel in B04 at three SNR levels (sigma = sqrt(P_b / 10^(SNR/10))).
    print("\nship contrast-to-noise in B04 (contrast from the ship pixels to nearby open water):")
    print(f"{'tile':12s} {'contrast DN':>11s} {'bg px':>7s} {'CNR@10dB':>9s} {'@20dB':>7s} {'@40dB':>7s}")
    for tile, contrast_dn, n_bg, power in conn.execute(
            "SELECT c.tile, c.contrast_dn, c.n_bg_px, s.power FROM tile_ship_contrast c "
            "JOIN tile_band_stats s ON s.tile = c.tile AND s.band = c.band WHERE c.band = 'B04' ORDER BY c.tile"):
        if contrast_dn is None:
            print(f"{tile:12s} {'n/a':>11s} {n_bg:7d}   (fewer than 30 water pixels near ships)")
            continue
        vals = [cnr(contrast_dn, (power / 10 ** (s / 10)) ** 0.5) for s in (10, 20, 40)]
        print(f"{tile:12s} {contrast_dn:11.0f} {n_bg:7d} {vals[0]:9.2f} {vals[1]:7.2f} {vals[2]:7.2f}")
    conn.close()
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
