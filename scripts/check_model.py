"""Verify src.models.nafnet: shapes, identity at initialisation, conditioning and cross-band attention
actually used, gradient flow, arbitrary sizes, parameter counts and CPU speed."""
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.nafnet import CONFIGS, COND_DIMS, CrossBandAttention, build_model, count_params

torch.manual_seed(0)
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


x = torch.randn(2, 12, 64, 64)
cond = torch.randn(2, 24)

# Identity at initialisation for every variant.
for cm in ("sigma_snr", "snr", "none"):
    for cb in (False, True):
        m = build_model("tiny", cm, cb).eval()
        c = torch.randn(2, COND_DIMS[cm]) if COND_DIMS[cm] else None
        with torch.no_grad():
            y = m(x, c)
        check(f"identity at init (cond={cm}, cross_band={cb})", y.shape == x.shape and torch.equal(y, x))

# Non-square and non-multiple sizes.
m = build_model("tiny", "sigma_snr", True).eval()
with torch.no_grad():
    check("odd input size 50x70", m(torch.randn(1, 12, 50, 70), torch.randn(1, 24)).shape == (1, 12, 50, 70))

# After perturbing the head, the output depends on the input, the conditioning, and (if on) the attention.
def perturb(model):
    torch.nn.init.normal_(model.head.weight, std=0.05)
    for mod in model.modules():
        if hasattr(mod, "proj") and mod.__class__.__name__ == "FiLM":
            torch.nn.init.normal_(mod.proj.weight, std=0.05)
        if hasattr(mod, "beta") and hasattr(mod, "gamma"):
            torch.nn.init.normal_(mod.beta, std=0.1)
            torch.nn.init.normal_(mod.gamma, std=0.1)

m = build_model("tiny", "sigma_snr", True)
perturb(m)
m.eval()
with torch.no_grad():
    y1, y2 = m(x, cond), m(x, cond + 1.0)
check("conditioning changes the output", (y1 - y2).abs().max() > 1e-4, f"(max diff {(y1 - y2).abs().max():.4f})")
blind = build_model("tiny", "none", True)
perturb(blind)
blind.eval()
with torch.no_grad():
    check("blind model ignores any conditioning passed", torch.equal(blind(x, cond), blind(x, cond + 5)))

# Cross-band attention: mixes bands (changing one band changes the others' outputs) when on.
cba = CrossBandAttention(12, 4)
with torch.no_grad():
    a = cba(x)
    x2 = x.clone()
    x2[:, 3] += 5.0
    b = cba(x2)
check("attention output shape", a.shape == (2, 12 * 4, 64, 64))
band_out = lambda t, i: t.reshape(2, 12, 4, 64, 64)[:, i]
check("a change in one band reaches other bands", (band_out(a, 7) - band_out(b, 7)).abs().max() > 1e-6)
attn_only = build_model("tiny", "none", False)
perturb(attn_only)
attn_only.eval()
with torch.no_grad():
    d = attn_only(x2, None) - attn_only(x, None)
check("without attention there is still spatial-conv mixing (sanity, not a claim)", d.abs().max() > 0)

# Gradients reach FiLM, the attention block and the stem after one step.
m = build_model("tiny", "sigma_snr", True)
perturb(m)
opt = torch.optim.AdamW(m.parameters(), 1e-3)
loss = (m(x, cond) - x).abs().mean() + 0.1 * m(x, cond).pow(2).mean()
loss.backward()
film_g = [p.grad.abs().sum().item() for n, p in m.named_parameters() if "film" in n and p.grad is not None]
check("gradients reach FiLM", len(film_g) > 0 and sum(film_g) > 0)
check("gradients reach cross-band attention", m.cba.qkv.weight.grad.abs().sum() > 0 and m.cba.band_embedding.grad.abs().sum() > 0)
check("gradients reach the stem and head", m.stem.weight.grad.abs().sum() > 0 and m.head.weight.grad.abs().sum() > 0)

# A few steps on a fixed batch should fit it (loss falls).
m = build_model("tiny", "sigma_snr", True)
target = x * 0.5
opt = torch.optim.AdamW(m.parameters(), 3e-3)
first = last = None
curve = []
for step in range(150):
    opt.zero_grad()
    loss = (m(x, cond) - target).abs().mean()
    loss.backward()
    opt.step()
    first = loss.item() if first is None else first
    last = loss.item()
    curve.append(last)
check("training fits a fixed batch", last < 0.5 * first, f"({first:.3f} -> {last:.3f} in 150 steps)")
check("loss falls steadily", curve[-1] < curve[len(curve) // 2] < curve[len(curve) // 8])

# Parameter counts and CPU speed for the named sizes (forward+backward, batch 4, 128x128).
print("\nparameters and CPU speed (batch 4, 128x128, cross-band on, sigma_snr):")
for size in CONFIGS:
    m = build_model(size, "sigma_snr", True)
    n = count_params(m)
    if size == "base":
        print(f"  {size:7s} {n / 1e6:6.2f} M params (speed not timed on CPU)")
        continue
    xb, cb_ = torch.randn(4, 12, 128, 128), torch.randn(4, 24)
    t = time.perf_counter()
    m(xb, cb_).abs().mean().backward()
    print(f"  {size:7s} {n / 1e6:6.2f} M params, {time.perf_counter() - t:5.2f} s per step")
n_blind = count_params(build_model("medium", "none", False))
n_full = count_params(build_model("medium", "sigma_snr", True))
print(f"  medium backbone only {n_blind / 1e6:.2f} M vs full {n_full / 1e6:.2f} M (+{(n_full - n_blind) / 1e6:.2f} M for FiLM + attention)")

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall model checks passed")
