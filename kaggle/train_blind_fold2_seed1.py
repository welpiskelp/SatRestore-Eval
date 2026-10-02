"""Kaggle kernel: single run, the one gap left over from seeds-batch2 (skipped by its time
budget): blind-model fold2 seed1. Same staging pattern as the other training kernels."""
import os
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/welpiskelp/SatRestore-Eval.git"
REPO_DIR = Path("/kaggle/working/SatRestore-Eval")
INPUT_ROOT = Path("/kaggle/input")
CONFIG = "configs/e1_blind_fold2_seed1.json"


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find_dataset_dir():
    print("contents of /kaggle/input:", list(INPUT_ROOT.iterdir()), flush=True)
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "patch_index.csv" in files:
            return Path(root)
    raise FileNotFoundError(f"could not find patch_index.csv under {INPUT_ROOT}")


def main():
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    run(["pip", "install", "-q", "rasterio==1.5.1"])

    dataset_dir = find_dataset_dir()
    print("using dataset dir:", dataset_dir, flush=True)

    processed = REPO_DIR / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "db").mkdir(exist_ok=True)

    for name in ("patch_index.csv", "signal_stats.json", "tiles_meta.json"):
        shutil.copy(dataset_dir / name, processed / name)
    shutil.copy(dataset_dir / "reference.sqlite", REPO_DIR / "db" / "reference.sqlite")

    raw_dir = REPO_DIR / "data" / "S2-SHIPS" / "S2SHIPS"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset_dir / "coco-s2ships.json", raw_dir / "coco-s2ships.json")

    tiles_src = dataset_dir / "tiles"
    if not tiles_src.exists():
        import zipfile
        (processed / "tiles").mkdir(exist_ok=True)
        with zipfile.ZipFile(dataset_dir / "tiles.zip") as z:
            z.extractall(processed / "tiles")
    else:
        os.symlink(tiles_src, processed / "tiles")
    n = len(list((processed / "tiles").glob("*")))
    print(f"staged {n} tile files", flush=True)

    import torch
    print("cuda available:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)

    run(["python", "scripts/train.py", CONFIG])
    import json
    run_name = json.loads(Path(CONFIG).read_text())["run_name"]
    ckpt = REPO_DIR / "runs" / run_name / "best.pt"
    run(["python", "scripts/evaluate_run.py", "--checkpoint", str(ckpt), "--out", "db/results_pilot_full.sqlite"])
    print(f"\n{run_name} finished; db/results_pilot_full.sqlite is under /kaggle/working/SatRestore-Eval", flush=True)


if __name__ == "__main__":
    main()
