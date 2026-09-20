"""Full-tile prediction: cut the noisy tile into the project's overlapping patch grid, restore each
patch, and average the overlaps. Also builds the conditioning vector for a tile."""
import numpy as np
import torch

from ..data.dataset import condition_vector
from ..data.patches import PATCH, STRIDE, patch_grid


def tile_condition(params: dict, scale, cond_mode: str, offset_db: float = 0.0) -> torch.Tensor:
    """Conditioning vector from a degradation's parameter record (src.degrade.noise.noise_params).
    offset_db tells the model an SNR that is off by that many dB (E4): sigma is rescaled consistently."""
    snr = np.asarray(params["snr_db"], dtype=np.float64) + offset_db
    sigma = np.asarray(params["sigma"], dtype=np.float64) * 10.0 ** (-offset_db / 20.0)
    return torch.from_numpy(condition_vector(snr, sigma, np.asarray(scale), cond_mode))


@torch.no_grad()
def predict_tile(model, noisy_dn, cond, scale, device="cpu", batch_size: int = 16, patch: int = PATCH,
                 stride: int = STRIDE) -> np.ndarray:
    """noisy_dn: (C, H, W) array in DN; cond: (D,) tensor shared by every patch of the tile;
    scale: (C,) normalisation used in training. Returns the restored tile in DN, float32."""
    model.eval()
    scale = np.asarray(scale, dtype=np.float32)
    c, h, w = noisy_dn.shape
    x = torch.from_numpy(np.asarray(noisy_dn, dtype=np.float32) / scale[:, None, None])
    acc = torch.zeros(c, h, w)
    cnt = torch.zeros(1, h, w)
    grid = patch_grid(h, w, patch, stride)
    cond = cond.to(device).float()
    for i in range(0, len(grid), batch_size):
        chunk = grid[i : i + batch_size]
        xb = torch.stack([x[:, y : y + patch, xx : xx + patch] for y, xx in chunk]).to(device)
        out = model(xb, cond.expand(len(chunk), -1)).float().cpu()
        for k, (y, xx) in enumerate(chunk):
            acc[:, y : y + patch, xx : xx + patch] += out[k]
            cnt[:, y : y + patch, xx : xx + patch] += 1
    return ((acc / cnt).numpy() * scale[:, None, None]).astype(np.float32)
