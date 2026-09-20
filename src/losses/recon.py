"""Reconstruction loss: masked L1 + lambda_ship * ship-weighted L1 + lambda_sam * spectral angle.

All terms ignore invalid pixels (clean reference 0 or saturated). Inputs are in the dataset's
normalised units; the spectral angle is computed on denormalised (DN) spectra so it matches the SAM
metric. Ship masks are used only here, never as network input.

  l1    mean |pred - clean| over valid pixels
  ship  the same mean restricted to ship pixels dilated by `ship_dilate` px (0 when the batch has no ship)
  sam   mean (1 - cos angle) between pred and clean spectra over pixels valid in every band
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_l1(pred, clean, valid):
    return (torch.abs(pred - clean) * valid).sum() / valid.sum().clamp_min(1.0)


def ship_l1(pred, clean, valid, ship, dilate: int = 1):
    w = F.max_pool2d(ship, 2 * dilate + 1, stride=1, padding=dilate) * valid
    return (torch.abs(pred - clean) * w).sum() / w.sum().clamp_min(1.0)


def sam_loss(pred, clean, valid, scale):
    """scale: (C,) tensor converting normalised values back to DN."""
    s = scale.view(1, -1, 1, 1)
    p, c = pred * s, clean * s
    all_valid = valid.min(dim=1, keepdim=True).values
    cos = (p * c).sum(1, keepdim=True) / (p.norm(dim=1, keepdim=True) * c.norm(dim=1, keepdim=True) + 1e-12)
    return ((1.0 - cos) * all_valid).sum() / all_valid.sum().clamp_min(1.0)


class ReconLoss(nn.Module):
    def __init__(self, scale, lambda_ship: float = 0.0, lambda_sam: float = 0.0, ship_dilate: int = 1):
        super().__init__()
        self.register_buffer("scale", torch.as_tensor(scale, dtype=torch.float32))
        self.lambda_ship, self.lambda_sam, self.ship_dilate = lambda_ship, lambda_sam, ship_dilate

    def forward(self, pred, batch):
        clean, valid, ship = batch["clean"], batch["valid"], batch["ship"]
        parts = {"l1": masked_l1(pred, clean, valid)}
        total = parts["l1"]
        if self.lambda_ship:
            parts["ship"] = ship_l1(pred, clean, valid, ship, self.ship_dilate)
            total = total + self.lambda_ship * parts["ship"]
        if self.lambda_sam:
            parts["sam"] = sam_loss(pred, clean, valid, self.scale)
            total = total + self.lambda_sam * parts["sam"]
        return total, {k: float(v.detach()) for k, v in parts.items()}
