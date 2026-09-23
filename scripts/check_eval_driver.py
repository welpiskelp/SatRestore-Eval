"""Verify the evaluation driver end to end and the model registry inside the training loop.

1. A checkpoint of an identity-at-init FFDNet-style model goes through load_checkpoint and evaluate_model;
   its results must equal the classical identity baseline already in db/results_baselines.sqlite.
2. Re-running skips finished conditions; conditioning offsets create separate runs.
3. Every architecture trains for one short epoch through src.train.loop and its best.pt reloads.
"""
import json
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset import load_band_scale
from src.eval.evaluate import evaluate_model, load_checkpoint, run_name_for
from src.models.registry import build_any
from src.train.loop import train

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


processed = Path("data/processed")
scale = load_band_scale(processed)
tmp = Path(tempfile.mkdtemp())

# 1. Identity checkpoint through the driver.
cfg = {"run_name": "identity_ffdnet", "fold": 0, "seed": 0,
       "model": {"arch": "ffdnet", "size": "tiny", "cond_mode": "sigma_snr"}}
model = build_any("ffdnet", "tiny", "sigma_snr")
ckpt = tmp / "best.pt"
torch.save({"model": model.state_dict(), "config": cfg, "scale": scale}, ckpt)
loaded, cfg2, scale2 = load_checkpoint(ckpt)
check("checkpoint round trip", type(loaded).__name__ == "FFDNetLike" and cfg2["run_name"] == "identity_ffdnet" and np.allclose(scale, scale2))

db = tmp / "results.sqlite"
tile, noises, snrs = "portsmouth", ["gaussian", "correlated_gaussian"], [10.0, 30.0]
n = evaluate_model(loaded, scale2, "sigma_snr", [tile], db, "identity_ffdnet", "ffdnet", noise_types=noises, snr_levels=snrs,
                   fold=0, seed=0, config=cfg, log=lambda s: None)
check("evaluated every condition", n == 4, f"({n} conditions)")

baseline_db = Path("db/results_baselines.sqlite")
if baseline_db.exists():
    a, b = sqlite3.connect(db), sqlite3.connect(baseline_db)
    q = ("SELECT noise_type, snr_db, region, metric, band, value FROM metrics m JOIN runs r USING(run_id) "
         "WHERE r.name = ? AND tile = ?")
    got = {(r[0], r[1], r[2], r[3], r[4]): r[5] for r in a.execute(q, ("identity_ffdnet", tile))}
    ref = {(r[0], r[1], r[2], r[3], r[4]): r[5] for r in b.execute(q, ("baseline_identity", tile))}
    keys = [k for k in got if k in ref and k[1] in snrs and k[0] in noises]
    worst = {}
    for k in keys:
        tol = 1e-4 if k[3] in ("ssim", "nrmse") else 2e-3
        d = abs(got[k] - ref[k])
        worst[k[3]] = max(worst.get(k[3], 0), d)
        if d > tol:
            check(f"identity model equals the classical identity baseline: {k}", False, f"({got[k]} vs {ref[k]})")
    check("identity model reproduces the classical identity baseline", len(keys) > 100 and not failures,
          f"({len(keys)} values compared, worst abs diff per metric: " + ", ".join(f"{m} {v:.1e}" for m, v in sorted(worst.items())) + ")")
    qi = ("SELECT instance_id, noise_type, snr_db, metric, value FROM instance_metrics i JOIN runs r USING(run_id) "
          "WHERE r.name = ?")
    gi = {(r[0], r[1], r[2], r[3]): r[4] for r in a.execute(qi, ("identity_ffdnet",))}
    ri = {(r[0], r[1], r[2], r[3]): r[4] for r in b.execute(qi, ("baseline_identity",)) if r[0].startswith(tile + "_s")}
    common = [k for k in gi if k in ri]
    md = max(abs(gi[k] - ri[k]) for k in common)
    check("per-ship metrics match the baseline", len(common) > 100 and md < 1e-4, f"({len(common)} values, max diff {md:.1e})")
    a.close()
    b.close()
else:
    print("skip  classical baseline database not present; cross-check skipped")

# 2. Resume and offsets.
n2 = evaluate_model(loaded, scale2, "sigma_snr", [tile], db, "identity_ffdnet", "ffdnet", noise_types=noises, snr_levels=snrs,
                    fold=0, seed=0, config=cfg, log=lambda s: None)
check("finished conditions are skipped on rerun", n2 == 0)
name6 = run_name_for("identity_ffdnet", 6.0)
check("offset run name", name6 == "identity_ffdnet_offset+6dB" and run_name_for("x", 0) == "x")
evaluate_model(loaded, scale2, "sigma_snr", [tile], db, name6, "ffdnet", noise_types=["gaussian"], snr_levels=[10.0],
               offset_db=6.0, fold=0, seed=0, log=lambda s: None)
c = sqlite3.connect(db)
runs = [r[0] for r in c.execute("SELECT name FROM runs ORDER BY run_id")]
check("offset evaluation is a separate run", runs == ["identity_ffdnet", name6], str(runs))
psnr = lambda name: c.execute("SELECT value FROM metrics m JOIN runs r USING(run_id) WHERE r.name=? AND tile=? AND noise_type='gaussian' AND "
                              "snr_db=10 AND region='all' AND metric='psnr' AND band IS NULL", (name, tile)).fetchone()[0]
check("an identity model is unaffected by a wrong SNR input", abs(psnr("identity_ffdnet") - psnr(name6)) < 1e-3)
c.close()

# 3. Every architecture trains through the loop and reloads.
base = json.loads(Path("configs/smoke.json").read_text())
for arch, cm, size in (("dncnn", "none", "tiny"), ("ffdnet", "sigma_snr", "tiny"), ("swinir", "none", "tiny"),
                       ("matched_control", "none", "tiny"), ("reconnet", "sigma_snr", "tiny")):
    c_ = json.loads(json.dumps(base))
    c_.update(run_name=f"t_{arch}", out_dir=str(tmp / "runs"))
    c_["model"] = {"arch": arch, "size": size, "cond_mode": cm, "cross_band": arch == "reconnet"}
    c_["data"].update(max_train_patches=16, max_val_patches=8)
    c_["optim"].update(epochs=1, batch_size=8)
    train(c_)
    m, cfg_r, _ = load_checkpoint(tmp / "runs" / f"t_{arch}" / "best.pt")
    lines = (tmp / "runs" / f"t_{arch}" / "metrics.jsonl").read_text().splitlines()
    ok = len(lines) == 1 and np.isfinite(json.loads(lines[0])["val_l1"])
    check(f"{arch}: trains one epoch and reloads", ok, f"(val l1 {json.loads(lines[0])['val_l1']:.4f})")

shutil.rmtree(tmp, ignore_errors=True)
if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall evaluation-driver checks passed")
