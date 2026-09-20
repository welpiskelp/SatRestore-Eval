"""Restoration metrics on full tiles, by region, ignoring invalid pixels.

Conventions (frozen in configs/protocol.json):
- A pixel is invalid in a band when the clean reference is 0 or 65535; it is excluded from that band's
  metrics. SAM uses pixels valid in all bands.
- PSNR uses a per-band peak (99.9th percentile of valid clean pixels, tile_band_stats.peak), never 65535.
- SSIM: uniform 7 x 7 window, K1 = 0.01, K2 = 0.03, data range = the band's peak, sample covariance.
- SAM in degrees; ERGAS with a resolution ratio of 1 (same grid); aggregates over bands are plain means
  (ERGAS is its own aggregate).
- Regions: all (every pixel), ship (ship-mask pixels), water (water mask minus a 3 px halo around ships),
  background (everything outside the 3 px halo around ships).
"""
import numpy as np

from ..data.bands import BANDS, band_index
from ..data.patches import SATURATED
from .contrast import dilate

HALO_PX = 3
NATIVE_BANDS = ["B02", "B03", "B04", "B08"]
SSIM_WIN = 7


def valid_mask(clean: np.ndarray) -> np.ndarray:
    """(C, H, W) bool, True where the clean reference is neither 0 nor saturated."""
    return (clean != 0) & (clean != SATURATED)


def region_masks(ship: np.ndarray, water: np.ndarray) -> dict[str, np.ndarray]:
    ship = ship.astype(bool)
    water = water.astype(bool)
    halo = dilate(ship, HALO_PX)
    return {"all": np.ones_like(ship), "ship": ship, "water": water & ~halo, "background": ~halo}


def _box_mean(a: np.ndarray, win: int) -> np.ndarray:
    c = np.pad(np.cumsum(np.cumsum(a, axis=0), axis=1), ((1, 0), (1, 0)))
    s = c[win:, win:] - c[:-win, win:] - c[win:, :-win] + c[:-win, :-win]
    return s / (win * win)


def ssim_map(x: np.ndarray, y: np.ndarray, data_range: float, win: int = SSIM_WIN) -> np.ndarray:
    """Structural similarity at every window centre; NaN within win//2 of the border."""
    x = np.asarray(x, dtype=np.float64) / data_range
    y = np.asarray(y, dtype=np.float64) / data_range
    n = win * win
    cov = n / (n - 1)
    ux, uy = _box_mean(x, win), _box_mean(y, win)
    vx = cov * (_box_mean(x * x, win) - ux * ux)
    vy = cov * (_box_mean(y * y, win) - uy * uy)
    vxy = cov * (_box_mean(x * y, win) - ux * uy)
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / ((ux * ux + uy * uy + c1) * (vx + vy + c2))
    out = np.full(x.shape, np.nan)
    p = win // 2
    out[p : p + s.shape[0], p : p + s.shape[1]] = s
    return out


def spectral_angle_deg(clean: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Per-pixel angle (degrees) between the clean and restored spectra, (H, W)."""
    dot = np.sum(clean * pred, axis=0)
    norm = np.sqrt(np.sum(clean * clean, axis=0)) * np.sqrt(np.sum(pred * pred, axis=0))
    return np.degrees(np.arccos(np.clip(dot / (norm + 1e-12), -1.0, 1.0)))


def tile_metrics(clean, pred, peak, regions: dict[str, np.ndarray]) -> list[dict]:
    """clean, pred: (C, H, W) arrays in DN units (band order = BANDS); peak: (C,) PSNR peaks.
    Returns one dict per (region, metric, band) with keys region, metric, band (None for aggregates),
    value, n_pixels."""
    clean = np.asarray(clean, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    peak = np.asarray(peak, dtype=np.float64)
    valid = valid_mask(clean)
    valid_all = valid.all(axis=0)
    err = pred - clean
    ssim = np.stack([ssim_map(clean[i], pred[i], peak[i]) for i in range(len(BANDS))])
    sam = spectral_angle_deg(clean, pred)

    rows = []
    for name, reg in regions.items():
        psnr_b, ssim_b, nrmse_b, ratio_b = [], [], [], []
        for i, band in enumerate(BANDS):
            m = reg & valid[i]
            n = int(m.sum())
            if n == 0:
                continue
            mse = float(np.mean(err[i][m].astype(np.float64) ** 2))
            rmse = mse**0.5
            s = ssim[i][m]
            s = s[~np.isnan(s)]
            vals = {
                "rmse": rmse,
                "nrmse": rmse / peak[i],
                "psnr": 10 * np.log10(peak[i] ** 2 / mse) if mse > 0 else float("inf"),
                "ssim": float(s.mean()) if len(s) else float("nan"),
            }
            for metric, v in vals.items():
                rows.append({"region": name, "metric": metric, "band": band, "value": float(v), "n_pixels": n})
            psnr_b.append(vals["psnr"])
            ssim_b.append(vals["ssim"])
            nrmse_b.append(vals["nrmse"])
            mean_clean = float(clean[i][m].astype(np.float64).mean())
            ratio_b.append(rmse / mean_clean)
        m_all = reg & valid_all
        n_all = int(m_all.sum())
        if n_all == 0 or not psnr_b:
            continue
        aggregates = {
            "psnr": float(np.mean(psnr_b)),
            "ssim": float(np.nanmean(ssim_b)),
            "nrmse": float(np.mean(nrmse_b)),
            "ergas": float(100 * np.sqrt(np.mean(np.square(ratio_b)))),
            "sam": float(sam[m_all].mean()),
        }
        for metric, v in aggregates.items():
            rows.append({"region": name, "metric": metric, "band": None, "value": v, "n_pixels": n_all})
    return rows


def instance_metrics(clean, pred, peak, centroids, ship, bands=NATIVE_BANDS, radius: int = 4, ring: int = 10):
    """Per ship instance, in the native 10 m bands. centroids: (n, 2) array of (x, y).
    Ship pixels of an instance are the ship-mask pixels within `radius` px (Chebyshev) of its centroid;
    the local background is the valid non-ship pixels within `ring` px, outside a 2 px halo of ship pixels.
    Returns {"ship_nrmse": (n,), "contrast_error": (n,)}, NaN where an instance has no ship pixel or
    no background pixel.
      ship_nrmse     root mean square error over the instance's ship pixels, normalised by each band's peak
      contrast_error mean over bands of |pred contrast - clean contrast| / peak, where contrast is
                     mean(ship pixels) - mean(local background); blurring a ship shrinks its contrast"""
    clean = np.asarray(clean, dtype=np.float32)
    pred = np.asarray(pred, dtype=np.float32)
    ship = ship.astype(bool)
    halo = dilate(ship, 2)
    idx = [band_index(b) for b in bands]
    pk = np.asarray(peak, dtype=np.float64)[idx]
    valid = valid_mask(clean)[idx]
    h, w = ship.shape
    nrmse = np.full(len(centroids), np.nan)
    cerr = np.full(len(centroids), np.nan)
    for k, (cx, cy) in enumerate(np.clip(centroids, 0, [w - 1e-6, h - 1e-6])):
        x, y = int(cx), int(cy)
        y0, y1, x0, x1 = max(0, y - ring), min(h, y + ring + 1), max(0, x - ring), min(w, x + ring + 1)
        win_ship = ship[y0:y1, x0:x1] & (np.abs(np.arange(y0, y1)[:, None] - y) <= radius) \
            & (np.abs(np.arange(x0, x1)[None, :] - x) <= radius)
        win_bg = ~halo[y0:y1, x0:x1]
        if not win_ship.any():
            continue
        sq, dc = [], []
        for j, b in enumerate(idx):
            c = clean[b, y0:y1, x0:x1].astype(np.float64)
            p = pred[b, y0:y1, x0:x1].astype(np.float64)
            v = valid[j, y0:y1, x0:x1]
            s_m, b_m = win_ship & v, win_bg & v
            if not s_m.any() or not b_m.any():
                sq = None
                break
            sq.append(np.mean(((p[s_m] - c[s_m]) / pk[j]) ** 2))
            dc.append(abs((p[s_m].mean() - p[b_m].mean()) - (c[s_m].mean() - c[b_m].mean())) / pk[j])
        if sq is None:
            continue
        nrmse[k] = float(np.sqrt(np.mean(sq)))
        cerr[k] = float(np.mean(dc))
    return {"ship_nrmse": nrmse, "contrast_error": cerr}
