"""Reference database (SQLite): schema creation and population of the reference tables."""
import json
import sqlite3
from pathlib import Path

from ..degrade.noise import NOISE_TYPES, TEST_BASE_SEED, TEST_SNR_LEVELS, degradation_seed, noise_params
from .bands import BANDS
from .patches import PATCH, patch_grid

SCHEMA_PATH = Path(__file__).resolve().parents[2] / "db" / "schema.sql"
REFERENCE_TABLES = ("degradations", "ship_instances", "patches", "folds", "tile_band_stats", "tiles")


def connect(path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text())


def insert_tiles(conn, meta_tiles: list[dict], polygons: dict[str, int], overlap_groups: list[list[str]]) -> None:
    group_of = {t: "+".join(g) for g in overlap_groups if len(g) > 1 for t in g}
    conn.executemany(
        "INSERT INTO tiles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (
                m["tile"], m["date"], m["shape"][1], m["shape"][2], m["crs"], *m["bounds"],
                m["ship_pixels"], polygons[m["tile"]], m["water_fraction"],
                m["ship_pixels_on_water_fraction"], group_of.get(m["tile"]),
            )
            for m in meta_tiles
        ],
    )


def insert_band_stats(conn, stats_tiles: dict[str, dict]) -> None:
    rows = []
    for tile, s in stats_tiles.items():
        for i, band in enumerate(BANDS):
            rows.append((tile, band, s["power"][i], s["mean"][i], s["valid_fraction"][i], s["power_naive"][i],
                         s["peak"][i]))
    conn.executemany("INSERT INTO tile_band_stats VALUES (?,?,?,?,?,?,?)", rows)


def insert_ship_contrast(conn, contrast_by_tile: dict[str, list[dict]]) -> None:
    """contrast_by_tile: tile -> per-band dicts from src.eval.contrast.ship_contrast (canonical band order)."""
    rows = []
    for tile, per_band in contrast_by_tile.items():
        for band, c in zip(BANDS, per_band):
            rows.append((tile, band, c["contrast_dn"], c["ship_mean"], c["bg_mean"], c["n_ship_px"], c["n_bg_px"]))
    conn.executemany("INSERT INTO tile_ship_contrast VALUES (?,?,?,?,?,?,?)", rows)


def insert_folds(conn, splits: dict) -> None:
    conn.executemany(
        "INSERT INTO folds VALUES (?,?,?)",
        [(f["fold"], t, split) for f in splits["folds"] for split in ("train", "val", "test") for t in f[split]],
    )


def insert_patches(conn, rows: list[dict]) -> None:
    conn.executemany(
        "INSERT INTO patches VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                r["patch_id"], r["tile"], int(r["y"]), int(r["x"]), int(r["ship_pixels"]), int(r["ship_instances"]),
                float(r["water_frac"]), float(r["sat_any_frac"]), float(r["sat_b09_frac"]), float(r["zero_any_frac"]),
            )
            for r in rows
        ],
    )


def home_patch(cx: float, cy: float, grid: list[tuple[int, int]], size: int = PATCH):
    """The patch containing (cx, cy) whose centre is nearest; ties go to the smaller (y, x)."""
    best = None
    for y, x in grid:
        if y <= cy < y + size and x <= cx < x + size:
            d = (cx - (x + size / 2)) ** 2 + (cy - (y + size / 2)) ** 2
            if best is None or d < best[0]:
                best = (d, y, x)
    if best is None:
        raise ValueError(f"no patch contains ({cx}, {cy})")
    return best[1], best[2]


def insert_ship_instances(conn, centroids: dict, shapes: dict[str, tuple[int, int]]) -> None:
    """centroids: tile -> (n, 2) array of (x, y); shapes: tile -> (height, width). Centroids are
    clipped into the image first (one polygon hangs off the bottom edge of a tile)."""
    import numpy as np

    rows = []
    for tile, c in centroids.items():
        h, w = shapes[tile]
        grid = patch_grid(h, w)
        c = np.clip(c, 0, [w - 1e-6, h - 1e-6])
        for i, (cx, cy) in enumerate(c):
            y, x = home_patch(cx, cy, grid)
            rows.append((f"{tile}_s{i:03d}", tile, float(cx), float(cy), f"{tile}_y{y:04d}_x{x:04d}"))
    conn.executemany("INSERT INTO ship_instances VALUES (?,?,?,?,?)", rows)


def insert_test_degradations(conn, stats_tiles: dict[str, dict], shot_fraction: float = 0.5,
                             base_seed: int = TEST_BASE_SEED) -> int:
    """One row per (tile, noise type, fixed test SNR level): the seed and full parameter record of
    the noisy tile that src.degrade.noise.test_noisy_tile generates for that combination."""
    rows = []
    for tile, s in stats_tiles.items():
        for noise_type in NOISE_TYPES:
            for snr in TEST_SNR_LEVELS:
                seed = degradation_seed(base_seed, tile, round(snr * 100))
                params = noise_params(s["power"], s["mean"], snr, noise_type, seed, shot_fraction)
                rows.append((tile, noise_type, snr, seed, params["shot_fraction"], json.dumps(params)))
    conn.executemany(
        "INSERT INTO degradations (tile, noise_type, snr_db, seed, shot_fraction, params_json) VALUES (?,?,?,?,?,?)",
        rows,
    )
    return len(rows)
