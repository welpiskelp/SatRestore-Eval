"""Verify src.data.dataset: shapes, noise level versus the target SNR, determinism, epoch variation,
invalid-pixel mask, conditioning vectors and split hygiene."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import COND_DIMS, PatchDataset
from src.data.bands import BANDS

processed = Path("data/processed")
splits = json.loads(Path("configs/splits.json").read_text())
fold0 = splits["folds"][0]
failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


ds = PatchDataset(processed, fold0["train"], "train", noise_types=("gaussian", "poisson_gaussian", "correlated_gaussian"))
item = ds[0]
check("shapes", item["noisy"].shape == (12, 128, 128) and item["clean"].shape == (12, 128, 128)
      and item["valid"].shape == (12, 128, 128) and item["ship"].shape == (1, 128, 128), str(tuple(item["noisy"].shape)))
check("cond dimension", item["cond"].shape == (COND_DIMS["sigma_snr"],))
check("dtype float32", item["noisy"].dtype == torch.float32 and item["cond"].dtype == torch.float32)
check("inputs are O(1)", 0 < float(item["clean"].mean()) < 3, f"(clean mean {float(item['clean'].mean()):.3f})")

# Same (epoch, index) reproduces; another epoch changes the noise, not the clean patch.
a, b = ds[5], ds[5]
check("deterministic for the same epoch and index", torch.equal(a["noisy"], b["noisy"]) and torch.equal(a["cond"], b["cond"]))
ds.set_epoch(1)
c = ds[5]
check("clean patch unchanged across epochs", torch.equal(a["clean"], c["clean"]))
check("noise differs across epochs", not torch.equal(a["noisy"], c["noisy"]))
ds.set_epoch(0)

# Realised noise power tracks the target SNR (mean over many patches, one noise type at a time).
for nt in ("gaussian", "correlated_gaussian", "poisson_gaussian"):
    d = PatchDataset(processed, fold0["train"], "train", noise_types=(nt,), snr_range=(20.0, 20.0))
    p, s = [], []
    idx = np.random.default_rng(0).choice(len(d), 150, replace=False)
    for i in idx:
        it = d[int(i)]
        v = it["valid"].bool()
        e = ((it["noisy"] - it["clean"]) ** 2)[v]
        p.append(float(e.mean()))
        s.append(float((it["clean"][v] ** 2).mean()))
    snr_real = 10 * np.log10(np.mean(s) / np.mean(p))
    # patches differ from the tile average signal power, so allow a few dB
    check(f"{nt}: realised SNR near 20 dB (patch-level)", abs(snr_real - 20) < 3, f"({snr_real:.1f} dB)")

# Invalid pixels are flagged in the mask.
sat = [r for r in ds.rows if float(r["sat_any_frac"]) > 0.05]
if sat:
    j = ds.rows.index(sat[0])
    check("saturated pixels are masked out", float(ds[j]["valid"].mean()) < 0.98, f"(valid fraction {float(ds[j]['valid'].mean()):.3f})")

# Ship mask matches the tile's ship pixels.
sh = [r for r in ds.rows if int(r["ship_pixels"]) > 0][0]
k = ds.rows.index(sh)
check("ship mask pixel count matches the index", int(ds[k]["ship"].sum()) == int(sh["ship_pixels"]))

# Split hygiene: train tiles only.
check("train dataset uses only the fold's train tiles", {r["tile"] for r in ds.rows} == set(fold0["train"]))
val = PatchDataset(processed, fold0["val"], "val", noise_types=("gaussian",))
check("val dataset uses only the fold's val tiles", {r["tile"] for r in val.rows} == set(fold0["val"]))
check("val noise is fixed across calls and epochs", torch.equal(val[3]["noisy"], val[3]["noisy"]))
val.set_epoch(7)
check("val noise ignores the epoch", torch.equal(val[3]["noisy"], PatchDataset(processed, fold0["val"], "val", noise_types=("gaussian",))[3]["noisy"]))
check("val cycles through the fixed SNR levels", sorted({round(float(val[i]["snr"])) for i in range(8)}) == [5, 15, 25, 35])

# Conditioning modes.
for mode in ("snr", "none"):
    m = PatchDataset(processed, fold0["train"], "train", cond_mode=mode)
    check(f"cond_mode {mode} dimension", m[0]["cond"].shape == (COND_DIMS[mode],))
snr_cond = PatchDataset(processed, fold0["train"], "train", cond_mode="snr")[0]
check("snr cond equals SNR / 40", abs(float(snr_cond["cond"][0]) - float(snr_cond["snr"]) / 40) < 1e-5)

# Ship oversampling repeats only ship patches.
rep = PatchDataset(processed, fold0["train"], "train", ship_patch_repeat=3)
check("ship_patch_repeat grows the dataset by the ship patches", len(rep) == len(ds) + 2 * sum(int(r["ship_pixels"]) > 0 for r in ds.rows))
check("dataset size", len(ds) > 3000, f"({len(ds)} train patches in fold 0, {len(val)} val)")

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall dataset checks passed")
