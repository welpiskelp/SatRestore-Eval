"""End-to-end smoke test on CPU with a tiny model: training runs, checkpoints are written, an interrupted run
resumes to exactly the same weights as an uninterrupted one, an untrained network reproduces the identity
baseline through tile prediction and the metrics, and a short training run beats the identity baseline."""
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import PatchDataset
from src.degrade.noise import test_noisy_tile
from src.degrade.stats import load_signal_stats
from src.eval.metrics import region_masks, tile_metrics
from src.eval.predict import predict_tile, tile_condition
from src.models.nafnet import build_model
from src.train.loop import build_run, train

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


base = json.loads(Path("configs/smoke.json").read_text())
out = Path("runs_smoke")
if out.exists():
    shutil.rmtree(out)


def cfg_for(name, epochs):
    c = json.loads(json.dumps(base))
    c.update(run_name=name, out_dir=str(out))
    c["optim"]["epochs"] = epochs
    return c


# 1. Training runs and writes its files.
t0 = time.time()
train(cfg_for("a", 2))
run_a = out / "a"
lines = [json.loads(l) for l in (run_a / "metrics.jsonl").read_text().splitlines()]
check("wrote last.pt, best.pt, metrics.jsonl", all((run_a / f).exists() for f in ("last.pt", "best.pt", "metrics.jsonl")))
check("two epochs logged with finite values", len(lines) == 2 and all(np.isfinite(r["val_l1"]) and np.isfinite(r["train_l1"]) for r in lines))
print(f"     ({(time.time() - t0) / 2:.0f} s per epoch on 96 patches, tiny model, CPU)")

# 2. Interrupt and resume equals an uninterrupted run (same config; the first call stops after epoch 0).
interrupted = cfg_for("b", 2)
interrupted["stop_after_epoch"] = 0
train(interrupted)
check("stop_after_epoch stops after one complete epoch",
      len((out / "b" / "metrics.jsonl").read_text().splitlines()) == 1)
train(cfg_for("b", 2))  # resumes from b/last.pt (epoch 0) and runs epoch 1
wa = torch.load(run_a / "last.pt", weights_only=False)["model"]
wb = torch.load(out / "b" / "last.pt", weights_only=False)["model"]
diff = max((wa[k].float() - wb[k].float()).abs().max().item() for k in wa)
check("resumed run equals the uninterrupted run", diff < 1e-6, f"(max weight difference {diff:.2e})")

if "--quick" in sys.argv:
    shutil.rmtree(out)
    if failures:
        print("\nFAILED:", failures)
        sys.exit(1)
    print("\nquick smoke test passed (training, checkpoints, resume)")
    sys.exit(0)

# 3. An untrained network is the identity: tile prediction and metrics reproduce the identity baseline.
processed = Path("data/processed")
train_ds, val_ds, model = build_run(cfg_for("c", 1))
scale = train_ds.scale
stats = load_signal_stats(processed / "signal_stats.json")
peaks = json.loads((processed / "signal_stats.json").read_text())["tiles"]["brest1"]["peak"]
clean = np.load(processed / "tiles" / "brest1_image.npy")
ship = np.load(processed / "tiles" / "brest1_ship.npy")
water = np.load(processed / "tiles" / "brest1_water.npy")
s = stats["brest1"]
noisy, params = test_noisy_tile(clean, s["power"], s["mean"], "brest1", 20.0, "gaussian")
cond = tile_condition(params, scale, "sigma_snr")
t0 = time.time()
pred0 = predict_tile(model, noisy, cond, scale)
print(f"     (full-tile prediction with the tiny model: {time.time() - t0:.0f} s on CPU)")
check("untrained network returns the noisy tile", np.allclose(pred0, noisy, rtol=1e-4, atol=1e-2),
      f"(max |diff| {np.abs(pred0 - noisy).max():.3g} DN)")
regions = region_masks(ship, water)
rows = tile_metrics(clean, pred0, peaks, regions)
got = next(r["value"] for r in rows if r["region"] == "all" and r["metric"] == "psnr" and r["band"] is None)
expected = np.mean(20.0 + 10 * np.log10(np.asarray(peaks) ** 2 / s["power"]))
check("identity PSNR through predict_tile matches the analytic value", abs(got - expected) < 0.1,
      f"(got {got:.2f}, expected {expected:.2f})")
# Conditioning offsets (E4) change the vector consistently.
c_off = tile_condition(params, scale, "sigma_snr", offset_db=6.0)
check("offset conditioning shifts the SNR entries by 6 dB", np.allclose((c_off[:12] - cond[:12]).numpy() * 40, 6.0, atol=1e-3))

# 4. A short training run beats the identity baseline on the fixed validation noise.
longer = cfg_for("d", 12)
longer["data"]["max_train_patches"] = 160
longer["optim"]["lr"] = 0.004
t0 = time.time()
train(longer)
lines = [json.loads(l) for l in (out / "d" / "metrics.jsonl").read_text().splitlines()]
identity_l1 = None
with torch.no_grad():
    from src.losses.recon import ReconLoss

    crit = ReconLoss(scale)
    tot = 0.0
    loader = torch.utils.data.DataLoader(val_ds, batch_size=8)
    for b in loader:
        _, p = crit(model(b["noisy"], b["cond"]), b)
        tot += p["l1"]
    identity_l1 = tot / len(loader)
best = min(r["val_l1"] for r in lines)
check("trained model beats the identity baseline on validation", best < identity_l1,
      f"(val L1 {identity_l1:.5f} identity -> {best:.5f} trained, {time.time() - t0:.0f} s)")
check("training loss fell", lines[-1]["train_l1"] < lines[0]["train_l1"], f"({lines[0]['train_l1']:.5f} -> {lines[-1]['train_l1']:.5f})")

shutil.rmtree(out)
if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nsmoke test passed")
