"""SNR-mismatch screening (protocol E4): does telling a conditioned model the wrong noise level
degrade it sensibly, confirming the FiLM conditioning signal is actually doing something?

No training involved -- this runs existing checkpoints (full_medium_fold{0-3}_seed0/best.pt)
through scripts/evaluate_run.py's own offset mechanism (src/eval/predict.py:tile_condition already
supports offset_db; this was simply never run with a non-zero offset before). Scoped down to one
test tile per fold and 3 SNR levels so a full CPU pass finishes in hours, not days -- a screening
pass, not the full protocol grid. Resumable within one process: reruns skip conditions already in
the output DB (does not persist across separate Kaggle sessions, same caveat as scripts/eval_bm3d.py).

Run locally:  python scripts/run_mismatch_screen.py --checkpoint-dir runs
Run on Kaggle: python scripts/run_mismatch_screen.py --checkpoint-dir /kaggle/input/satrestore-mismatch-checkpoints --flat-names
  (--flat-names: checkpoints uploaded as <run_name>_best.pt directly in the dataset root, rather
  than runs/<run_name>/best.pt)
"""
import argparse
import json
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.eval.evaluate import evaluate_model, load_checkpoint, run_name_for

OFFSETS = [-10.0, -5.0, 5.0, 10.0]
SNR_LEVELS = [0.0, 20.0, 40.0]
NOISE_TYPES = ["gaussian"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", default="runs")
    ap.add_argument("--flat-names", action="store_true",
                     help="checkpoints are <checkpoint-dir>/<run_name>_best.pt instead of "
                          "<checkpoint-dir>/<run_name>/best.pt")
    ap.add_argument("--out", default="db/results_mismatch.sqlite")
    ap.add_argument("--processed-dir", default="data/processed")
    ap.add_argument("--raw-dir", default="data/S2-SHIPS/S2SHIPS")
    ap.add_argument("--time-budget-hours", type=float, default=None)
    args = ap.parse_args()

    folds = json.load(open("configs/splits.json"))["folds"]
    ckpt_dir = Path(args.checkpoint_dir)

    t_start = time.time()
    stop = False
    for fold in range(4):
        if stop:
            break
        run_name = f"full_medium_fold{fold}_seed0"
        ckpt = ckpt_dir / f"{run_name}_best.pt" if args.flat_names else ckpt_dir / run_name / "best.pt"
        if not ckpt.exists():
            print(f"skip fold {fold}: no checkpoint at {ckpt}")
            continue
        tile = folds[fold]["test"][0]
        model, cfg, scale = load_checkpoint(str(ckpt), "cpu")
        for off in OFFSETS:
            if args.time_budget_hours and (time.time() - t_start) / 3600 > args.time_budget_hours:
                print(f"[{time.time() - t_start:7.0f}s] time budget reached, stopping before "
                      f"fold {fold} offset {off:+g}", flush=True)
                stop = True
                break
            n = evaluate_model(
                model, scale, cfg["model"]["cond_mode"], [tile], args.out, run_name_for(cfg["run_name"], off),
                cfg["model"].get("arch", "reconnet"), noise_types=NOISE_TYPES, snr_levels=SNR_LEVELS,
                offset_db=off, processed_dir=args.processed_dir, raw_dir=args.raw_dir, device="cpu",
                batch_size=16, fold=fold, seed=cfg.get("seed"), config=cfg,
                log=lambda s: print(f"[{time.time()-t_start:7.0f}s] {s}", flush=True),
            )
            print(f"fold {fold} tile {tile} offset {off:+g} dB: {n} new conditions", flush=True)

    print(f"\ndone in {(time.time()-t_start)/3600:.2f}h, wrote {args.out}")


if __name__ == "__main__":
    main()
