"""Verify src.losses.recon: masked L1, ship-weighted L1, spectral angle, composition and gradients."""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.losses.recon import ReconLoss, masked_l1, sam_loss, ship_l1

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


B, C, H, W = 2, 12, 32, 32
clean = torch.rand(B, C, H, W) + 0.1
pred = clean + 0.05 * torch.randn(B, C, H, W)
valid = torch.ones(B, C, H, W)
ship = torch.zeros(B, 1, H, W)
ship[0, 0, 10:12, 10:12] = 1
scale = torch.linspace(1000, 5000, C)

# Masked L1.
check("l1 equals the plain mean when all valid", torch.allclose(masked_l1(pred, clean, valid), (pred - clean).abs().mean()))
v2 = valid.clone()
v2[:, :, :5, :] = 0
p2 = pred.clone()
p2[:, :, :5, :] += 100
check("l1 ignores invalid pixels wherever the prediction is", torch.allclose(masked_l1(p2, clean, v2), masked_l1(pred, clean, v2)))
check("l1 is 0 when nothing is valid", masked_l1(pred, clean, torch.zeros_like(valid)) == 0)

# Ship-weighted L1.
dil = torch.nn.functional.max_pool2d(ship, 3, 1, 1)
manual = ((pred - clean).abs() * dil).sum() / (dil.expand(B, C, H, W).sum())
check("ship l1 equals the mean over the dilated ship region", torch.allclose(ship_l1(pred, clean, valid, ship, 1), manual, atol=1e-6))
check("ship l1 is 0 with no ship in the batch", ship_l1(pred, clean, valid, torch.zeros_like(ship), 1) == 0)
p3 = pred.clone()
p3[:, :, 20:30, 20:30] += 50
check("ship l1 ignores errors far from ships", torch.allclose(ship_l1(p3, clean, valid, ship, 1), ship_l1(pred, clean, valid, ship, 1)))
check("more dilation covers more pixels", (torch.nn.functional.max_pool2d(ship, 5, 1, 2) > 0).sum() > (dil > 0).sum())

# SAM loss.
check("sam loss is 0 for a scaled spectrum", sam_loss(clean * 1.7, clean, valid, scale).abs() < 1e-6)
a = torch.zeros(1, C, 1, 1)
a[0, 0] = 1
b = torch.zeros(1, C, 1, 1)
b[0, 1] = 1
check("sam loss is 1 for orthogonal spectra", abs(float(sam_loss(a, b, torch.ones(1, C, 1, 1), torch.ones(C))) - 1.0) < 1e-6)
p5 = pred.clone()
v4 = valid.clone()
v4[:, 2, 8:12, 8:12] = 0
p5[:, :, 8:12, 8:12] = torch.rand(B, C, 4, 4) * 9
check("sam ignores those pixels exactly", torch.allclose(sam_loss(p5, clean, v4, scale),
      sam_loss(torch.where(v4.min(1, keepdim=True).values.bool(), p5, pred), clean, v4, scale)))
# Scale matters: the angle is computed on DN spectra, not on the normalised ones.
check("sam uses the band scale", abs(float(sam_loss(pred, clean, valid, scale)) - float(sam_loss(pred, clean, valid, torch.ones(C)))) > 1e-6)

# Composition and gradients.
batch = {"clean": clean, "valid": valid, "ship": ship}
crit = ReconLoss(scale, lambda_ship=0.5, lambda_sam=2.0)
p = pred.clone().requires_grad_(True)
total, parts = crit(p, batch)
expect = parts["l1"] + 0.5 * parts["ship"] + 2.0 * parts["sam"]
check("total = l1 + lambda_ship * ship + lambda_sam * sam", abs(float(total) - expect) < 1e-5, f"({float(total):.4f})")
total.backward()
check("gradients are finite", torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
plain, parts0 = ReconLoss(scale)(pred, batch)
check("with zero lambdas only l1 remains", set(parts0) == {"l1"} and torch.allclose(plain, masked_l1(pred, clean, valid)))

# The ship term's gradient lives only in the dilated ship region.
p = pred.clone().requires_grad_(True)
ship_l1(p, clean, valid, ship, 1).backward()
outside = (p.grad.abs().sum(1, keepdim=True) * (1 - dil)).sum()
check("ship-term gradient is confined to the ship neighbourhood", outside == 0 and p.grad.abs().sum() > 0)

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall loss checks passed")
