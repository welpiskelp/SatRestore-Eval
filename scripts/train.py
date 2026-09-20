"""Train one run from a config file: python scripts/train.py configs/<run>.json [--no-resume]

Resumes from <out_dir>/<run_name>/last.pt when present, so a run interrupted by a session limit
continues where it stopped.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.train.loop import train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], help="override, e.g. optim.epochs=2 data.max_train_patches=64")
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    for item in args.set:
        key, value = item.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node[p]
        node[parts[-1]] = json.loads(value)
    print(train(cfg, resume=not args.no_resume))


if __name__ == "__main__":
    main()
