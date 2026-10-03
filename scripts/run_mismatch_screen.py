"""SNR-mismatch screening (protocol E4): does telling a conditioned model the wrong noise level
degrade it sensibly, confirming the FiLM conditioning signal is actually doing something?

No training involved -- this runs existing checkpoints (runs/full_medium_fold{0-3}_seed0/best.pt)
through scripts/evaluate_run.py's own offset mechanism (src/eval/predict.py:tile_condition already
supports offset_db; this was simply never run with a non-zero offset before). Scoped down to one
test tile per fold and 3 SNR levels so a full CPU pass finishes in hours, not days -- a screening
pass, not the full protocol grid. Resumable: reruns skip conditions already in the output DB.

Run: python scripts/run_mismatch_screen.py
"""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.evaluate import evaluate_model, load_checkpoint, run_name_for

OUT_DB = "db/results_mismatch.sqlite"
OFFSETS = [-10.0, -5.0, 5.0, 10.0]
SNR_LEVELS = [0.0, 20.0, 40.0]
NOISE_TYPES = ["gaussian"]

FOLDS = json.load(open("configs/splits.json"))["folds"]

t_start = time.time()
for fold in range(4):
    ckpt = f"runs/full_medium_fold{fold}_seed0/best.pt"
    if not Path(ckpt).exists():
        print(f"skip fold {fold}: no local checkpoint at {ckpt}")
        continue
    tile = FOLDS[fold]["test"][0]
    model, cfg, scale = load_checkpoint(ckpt, "cpu")
    base_name = cfg["run_name"]
    for off in OFFSETS:
        n = evaluate_model(
            model, scale, cfg["model"]["cond_mode"], [tile], OUT_DB, run_name_for(base_name, off),
            cfg["model"].get("arch", "reconnet"), noise_types=NOISE_TYPES, snr_levels=SNR_LEVELS,
            offset_db=off, device="cpu", batch_size=16, fold=fold, seed=cfg.get("seed"), config=cfg,
            log=lambda s: print(f"[{time.time()-t_start:7.0f}s] {s}", flush=True),
        )
        print(f"fold {fold} tile {tile} offset {off:+g} dB: {n} new conditions", flush=True)

print(f"\ndone in {(time.time()-t_start)/3600:.2f}h, wrote {OUT_DB}")
