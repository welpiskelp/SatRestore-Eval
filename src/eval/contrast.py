"""Ship visibility: contrast between ship pixels and nearby open water, and contrast-to-noise."""
import numpy as np

from ..data.patches import SATURATED

MIN_BG_PIXELS = 30


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(n):
        step = out.copy()
        step[1:] |= out[:-1]
        step[:-1] |= out[1:]
        step[:, 1:] |= out[:, :-1]
        step[:, :-1] |= out[:, 1:]
        out = step
    return out


def ship_contrast(image, ship, water, inner: int = 3, outer: int = 10) -> list[dict]:
    """Per band: contrast = mean(ship pixels) - mean(water pixels between `inner` and `outer` pixels
    from any ship pixel). Zero and saturated pixels are ignored. contrast_dn is None when fewer than
    MIN_BG_PIXELS background pixels exist."""
    ship = ship.astype(bool)
    water = water.astype(bool)
    ring = dilate(ship, outer) & ~dilate(ship, inner) & water
    out = []
    for i in range(image.shape[0]):
        x = np.asarray(image[i]).astype(np.float64)
        ok = (x != 0) & (x != SATURATED)
        s, b = x[ship & ok], x[ring & ok]
        enough = len(s) > 0 and len(b) >= MIN_BG_PIXELS
        out.append({
            "contrast_dn": float(s.mean() - b.mean()) if enough else None,
            "ship_mean": float(s.mean()) if len(s) else None,
            "bg_mean": float(b.mean()) if len(b) else None,
            "n_ship_px": int(len(s)),
            "n_bg_px": int(len(b)),
        })
    return out


def cnr(contrast_dn: float, sigma: float) -> float:
    """Per-pixel contrast-to-noise ratio |contrast| / sigma. Below 1, a ship pixel is under one
    noise standard deviation."""
    return abs(contrast_dn) / sigma
