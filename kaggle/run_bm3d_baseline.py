"""Kaggle kernel: BM3D classical baseline (no training, CPU-only, no GPU needed).
Stages the same dataset as the training kernels, then runs scripts/eval_bm3d.py on the 6
baseline tiles x 5 SNR levels (gaussian noise only) x 12 bands. ~5h on a single CPU core;
runs independently of GPU sessions so it can go in parallel with training batches.
"""
import os
import shutil
import subprocess
import traceback
from pathlib import Path

REPO_URL = "https://github.com/welpiskelp/SatRestore-Eval.git"
REPO_DIR = Path("/kaggle/working/SatRestore-Eval")
INPUT_ROOT = Path("/kaggle/input")
ERROR_FILE = Path("/kaggle/working/ERROR.txt")


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

    run(["pip", "install", "-q", "bm3d", "rasterio==1.5.1"])

    dataset_dir = find_dataset_dir()
    print("using dataset dir:", dataset_dir, flush=True)

    processed = REPO_DIR / "data" / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    (REPO_DIR / "db").mkdir(exist_ok=True)

    for name in ("patch_index.csv", "signal_stats.json", "tiles_meta.json"):
        shutil.copy(dataset_dir / name, processed / name)

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

    import sys
    log_path = Path("/kaggle/working/bm3d_run.log")
    with open(log_path, "a", buffering=1) as logf:
        proc = subprocess.Popen(
            ["python", "-u", "scripts/eval_bm3d.py", "--out", "db/results_bm3d.sqlite"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"eval_bm3d.py exited with code {rc}; see bm3d_run.log")
    print("\nbm3d baseline finished; db/results_bm3d.sqlite is under /kaggle/working/SatRestore-Eval", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERROR_FILE.write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        raise
