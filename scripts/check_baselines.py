"""Verify src.models.baselines and the registry: identity at initialisation, shapes, conditioning use, gradient
flow, fitting a fixed batch, parameter counts and the parameter-matched control."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.nafnet import build_model, count_params
from src.models.registry import BASELINE_SIZES, build_any, build_matched_control

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


x = torch.randn(2, 12, 32, 40)
cond = torch.randn(2, 24)
setups = [("dncnn", "none"), ("ffdnet", "sigma_snr"), ("ffdnet", "snr"), ("swinir", "none")]

for arch, cm in setups:
    m = build_any(arch, "tiny", cm).eval()
    c = torch.randn(2, {"sigma_snr": 24, "snr": 1, "none": 1}[cm])
    with torch.no_grad():
        y = m(x, c)
    check(f"{arch}/{cm}: identity at init, shape kept", y.shape == x.shape and torch.equal(y, x))
    with torch.no_grad():
        y = m(torch.randn(1, 12, 33, 45), c[:1])
    check(f"{arch}/{cm}: odd input size", y.shape == (1, 12, 33, 45))

# Blind models reject conditioning modes, non-blind needs one.
for arch, bad in (("dncnn", "sigma_snr"), ("swinir", "snr"), ("ffdnet", "none")):
    try:
        build_any(arch, "tiny", bad)
        check(f"{arch} rejects cond_mode {bad}", False)
    except ValueError:
        check(f"{arch} rejects cond_mode {bad}", True)


def perturb(model):
    for p in model.parameters():
        if p.ndim >= 2 and torch.count_nonzero(p) == 0:
            torch.nn.init.normal_(p, std=0.05)


# FFDNet-like uses its conditioning; the blind models ignore it.
ff = build_any("ffdnet", "tiny", "sigma_snr")
perturb(ff)
ff.eval()
with torch.no_grad():
    d = (ff(x, cond) - ff(x, cond + 1.0)).abs().max().item()
check("ffdnet output depends on the conditioning", d > 1e-5, f"(max diff {d:.4f})")
for arch in ("dncnn", "swinir"):
    b = build_any(arch, "tiny", "none")
    perturb(b)
    b.eval()
    with torch.no_grad():
        check(f"{arch} ignores conditioning", torch.equal(b(x, cond), b(x, cond + 5)))

# Gradients flow and a fixed batch can be fitted (clean = smoothed input, a denoising-like target).
target = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
for arch, cm in setups:
    m = build_any(arch, "tiny", cm)
    c = torch.randn(2, {"sigma_snr": 24, "snr": 1, "none": 1}[cm])
    opt = torch.optim.AdamW(m.parameters(), 3e-3)
    first = last = None
    for _ in range(120):
        opt.zero_grad()
        loss = (m(x, c) - target).abs().mean()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
        last = loss.item()
    grads = sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    check(f"{arch}/{cm}: gradients flow and a fixed batch is fitted", grads > 0 and last < 0.85 * first, f"({first:.3f} -> {last:.3f})")

# Parameter counts at every size, and CPU speed for the medium/base sizes on a small batch.
print("\nparameters (M):")
print(f"  {'':12s} " + " ".join(f"{s:>8s}" for s in ("tiny", "small", "medium", "base")))
for arch in BASELINE_SIZES:
    cm = {"dncnn": "none", "swinir": "none", "ffdnet": "sigma_snr"}[arch]
    print(f"  {arch:12s} " + " ".join(f"{count_params(build_any(arch, s, cm)) / 1e6:8.2f}" for s in ("tiny", "small", "medium", "base")))
for cm, cb, label in (("none", False, "reconnet backbone"), ("sigma_snr", True, "reconnet full")):
    print(f"  {label:12s} " + " ".join(f"{count_params(build_model(s, cm, cb)) / 1e6:8.2f}" for s in ("tiny", "small", "medium", "base")))

# Parameter-matched control.
full = count_params(build_model("medium", "sigma_snr", True))
ctrl = build_matched_control("medium")
n = count_params(ctrl)
check("matched control is within 3% of the full model's parameters", abs(n - full) / full < 0.03,
      f"({n / 1e6:.3f} M vs {full / 1e6:.3f} M, width {ctrl.stem.out_channels})")
check("matched control has no FiLM or attention", ctrl.cba is None and ctrl.cond_mlp is None)
back = count_params(build_model("medium", "none", False))
check("matched control is wider than the plain backbone", n > back, f"({back / 1e6:.2f} M backbone -> {n / 1e6:.2f} M)")
try:
    build_any("matched_control", "medium", "sigma_snr")
    check("matched control rejects conditioning", False)
except ValueError:
    check("matched control rejects conditioning", True)

print("\nparameter-matched baselines (target = full ReconNet medium):")
for arch, cm in (("dncnn", "none"), ("ffdnet", "sigma_snr"), ("swinir", "none")):
    mm = build_any(arch, "matched", cm)
    nn_ = count_params(mm)
    check(f"{arch} matched to the full model within 6%", abs(nn_ - full) / full < 0.06, f"({nn_ / 1e6:.2f} M vs {full / 1e6:.2f} M)")
    with torch.no_grad():
        check(f"{arch} matched still starts as the identity", torch.equal(mm.eval()(x, torch.randn(2, 24 if cm == 'sigma_snr' else 1)), x))

print("\nCPU time per training step, batch 2, 64x64:")
xs, cs = torch.randn(2, 12, 64, 64), torch.randn(2, 24)
for arch, cm in (("dncnn", "none"), ("ffdnet", "sigma_snr"), ("swinir", "none")):
    m = build_any(arch, "base", cm)
    c = cs if cm != "none" else None
    t = time.perf_counter()
    m(xs, c).abs().mean().backward()
    print(f"  {arch:8s} base: {time.perf_counter() - t:5.2f} s")

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall baseline checks passed")
