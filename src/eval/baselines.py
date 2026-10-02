"""Classical reference restorers used to test the harness and to give first reference numbers."""
import numpy as np


def identity(noisy: np.ndarray) -> np.ndarray:
    """No restoration: the noisy input is the prediction."""
    return noisy


def gaussian_filter(x: np.ndarray, sigma: float) -> np.ndarray:
    """Per-band separable Gaussian smoothing of a (C, H, W) array, reflect padding."""
    if sigma <= 0:
        return x
    r = max(1, int(np.ceil(3 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = x.astype(np.float32)
    for axis in (1, 2):
        pad = [(0, 0)] * 3
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="reflect")
        n = out.shape[axis]
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None)] * 3
            sl[axis] = slice(i, i + n)
            acc += np.float32(w) * padded[tuple(sl)]
        out = acc
    return out


def bm3d_denoise(x: np.ndarray, sigma) -> np.ndarray:
    """Per-band BM3D (classical, non-learned denoiser) on a (C, H, W) array, each band
    independently normalised to [0, 1] by its own clean-signal range before denoising, then
    scaled back. sigma: per-band noise std in the same (raw DN) units as x, shape (C,)."""
    import bm3d as _bm3d

    sigma = np.asarray(sigma, dtype=np.float64)
    out = np.empty(x.shape, dtype=np.float32)
    for c in range(x.shape[0]):
        band = x[c].astype(np.float64)
        lo, hi = band.min(), band.max()
        scale = max(hi - lo, 1e-6)
        band01 = (band - lo) / scale
        sigma01 = sigma[c] / scale
        denoised01 = _bm3d.bm3d(band01, sigma_psd=float(sigma01))
        out[c] = (denoised01 * scale + lo).astype(np.float32)
    return out
