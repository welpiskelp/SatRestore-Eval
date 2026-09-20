"""Per-tile, per-band signal statistics that define SNR (P_b in SNR = 10*log10(P_b / sigma_b^2))."""
import json
from pathlib import Path

import numpy as np

from ..data.bands import BANDS
from ..data.patches import SATURATED


def band_signal_stats(image) -> dict:
    """image: (C, H, W) uint16 tile. Per band, over valid pixels only (not 0 and not saturated):
    power = mean square (P_b), mean = mean signal (sets the Poisson gain), valid_fraction.
    power_naive is the mean square over all pixels, kept only to show what the exclusion changes."""
    out = {"power": [], "mean": [], "valid_fraction": [], "power_naive": [], "peak": []}
    for i in range(image.shape[0]):
        x = np.asarray(image[i]).astype(np.float64)
        ok = (x != 0) & (x != SATURATED)
        v = x[ok]
        out["power"].append(float(np.mean(v * v)))
        out["mean"].append(float(v.mean()))
        out["valid_fraction"].append(float(ok.mean()))
        out["power_naive"].append(float(np.mean(x * x)))
        out["peak"].append(float(np.percentile(v, 99.9)))  # robust PSNR peak (a fixed 65535 would inflate PSNR)
    return out


def write_signal_stats(path: Path, stats_by_tile: dict) -> None:
    path.write_text(json.dumps({"bands": BANDS, "tiles": stats_by_tile}, indent=2))


def load_signal_stats(path: Path) -> dict[str, dict[str, np.ndarray]]:
    """tile -> {"power": (C,), "mean": (C,)} float64 arrays in canonical band order."""
    data = json.loads(path.read_text())
    if data["bands"] != BANDS:
        raise ValueError(f"{path} was written for a different band order")
    return {
        tile: {"power": np.asarray(s["power"]), "mean": np.asarray(s["mean"])}
        for tile, s in data["tiles"].items()
    }
