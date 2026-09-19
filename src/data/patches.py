"""128x128 patch grid (stride 64) over the prepared tiles, and per-patch statistics."""
import numpy as np

from .bands import BANDS

PATCH = 128
STRIDE = 64
SATURATED = 65535
B09 = BANDS.index("B09")

FIELDS = [
    "patch_id", "tile", "y", "x",
    "ship_pixels", "ship_instances", "water_frac",
    "sat_any_frac", "sat_b09_frac", "zero_any_frac",
]


def _starts(n: int, size: int, stride: int) -> list[int]:
    starts = list(range(0, n - size + 1, stride))
    if starts[-1] != n - size:
        starts.append(n - size)  # edge-aligned patch so the last rows/columns are covered
    return starts


def patch_grid(h: int, w: int, size: int = PATCH, stride: int = STRIDE) -> list[tuple[int, int]]:
    """Top-left (y, x) of every patch: a regular stride grid plus one edge-aligned row and column,
    so every pixel of the tile is covered. The edge patches overlap their neighbours more."""
    return [(y, x) for y in _starts(h, size, stride) for x in _starts(w, size, stride)]


def covered_extent(h: int, w: int, size: int = PATCH, stride: int = STRIDE) -> tuple[int, int]:
    """(rows, cols) covered by the grid; equals (h, w) because of the edge-aligned patches."""
    grid = patch_grid(h, w, size, stride)
    return max(y for y, _ in grid) + size, max(x for _, x in grid) + size


def clip_centroids(centroids: np.ndarray, h: int, w: int) -> np.ndarray:
    """Pull centroids into the image. One portsmouth polygon hangs off the bottom edge
    (centroid y = 938.3 on a 938-row tile) and would otherwise belong to no patch."""
    return np.clip(centroids, 0, [w - 1e-6, h - 1e-6])


def tile_patch_rows(tile, image, ship, water, centroids, size=PATCH, stride=STRIDE) -> list[dict]:
    """One row per patch. ship_instances counts ship polygons whose centroid lies inside the patch
    (a ship near a patch border can therefore appear in up to four overlapping patches)."""
    n_bands, h, w = image.shape
    centroids = clip_centroids(centroids, h, w)
    sat_any = np.zeros((h, w), dtype=bool)
    zero_any = np.zeros((h, w), dtype=bool)
    for i in range(n_bands):
        band = np.asarray(image[i])
        sat_any |= band == SATURATED
        zero_any |= band == 0
    sat_b09 = np.asarray(image[B09]) == SATURATED

    rows = []
    for y, x in patch_grid(h, w, size, stride):
        sl = (slice(y, y + size), slice(x, x + size))
        inside = (
            (centroids[:, 0] >= x) & (centroids[:, 0] < x + size)
            & (centroids[:, 1] >= y) & (centroids[:, 1] < y + size)
        )
        rows.append({
            "patch_id": f"{tile}_y{y:04d}_x{x:04d}",
            "tile": tile, "y": y, "x": x,
            "ship_pixels": int(ship[sl].sum()),
            "ship_instances": int(inside.sum()),
            "water_frac": round(float(water[sl].mean()), 6),
            "sat_any_frac": round(float(sat_any[sl].mean()), 6),
            "sat_b09_frac": round(float(sat_b09[sl].mean()), 6),
            "zero_any_frac": round(float(zero_any[sl].mean()), 6),
        })
    return rows
