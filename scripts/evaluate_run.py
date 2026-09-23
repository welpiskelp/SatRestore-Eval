"""Evaluate a checkpoint on its fold's test tiles (or chosen tiles) over the fixed test noise.

  python scripts/evaluate_run.py --checkpoint runs/<run>/best.pt --out db/results_<person>.sqlite

Run name defaults to the config's run_name and must be unique across everyone's results. --offsets adds the
SNR-mismatch runs of experiment E4 (the model is told an SNR that is off by that many dB); blind models
ignore the conditioning, so only offset 0 is evaluated for them. Conditions already recorded are skipped.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.degrade.noise import NOISE_TYPES, TEST_SNR_LEVELS
from src.eval.evaluate import evaluate_model, load_checkpoint, run_name_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="results database (created if missing)")
    ap.add_argument("--name", help="run name (default: the config's run_name)")
    ap.add_argument("--tiles", nargs="+", help="default: the test tiles of the checkpoint's fold")
    ap.add_argument("--noise-types", nargs="+", default=list(NOISE_TYPES))
    ap.add_argument("--snr", nargs="+", type=float, default=list(TEST_SNR_LEVELS))
    ap.add_argument("--offsets", nargs="+", type=float, default=[0.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--crops-dir", help="save small clean/noisy/restored ship crops for figures")
    ap.add_argument("--crop-conditions", nargs="*", default=["gaussian:10", "gaussian:30"])
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    args = ap.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    model, cfg, scale = load_checkpoint(args.checkpoint, device)
    cond_mode = cfg["model"]["cond_mode"]
    tiles = args.tiles or json.loads(Path(cfg.get("splits", "configs/splits.json")).read_text())["folds"][cfg["fold"]]["test"]
    name = args.name or cfg["run_name"]
    offsets = args.offsets if cond_mode != "none" else [0.0]
    if cond_mode == "none" and args.offsets != [0.0]:
        print("blind model: offsets ignored, evaluating offset 0 only")

    for off in offsets:
        n = evaluate_model(model, scale, cond_mode, tiles, args.out, run_name_for(name, off), cfg["model"].get("arch", "reconnet"),
                           noise_types=args.noise_types, snr_levels=args.snr, offset_db=off,
                           processed_dir=args.processed_dir, raw_dir=args.raw_dir, device=device,
                           batch_size=args.batch_size, fold=cfg.get("fold"), seed=cfg.get("seed"), config=cfg,
                           crops_dir=args.crops_dir, crop_conditions=tuple(args.crop_conditions),
                           log=lambda s: print(s, flush=True))
        print(f"offset {off:+g} dB: {n} new conditions evaluated")


if __name__ == "__main__":
    main()
