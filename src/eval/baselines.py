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
