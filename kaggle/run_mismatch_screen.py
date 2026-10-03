"""Kaggle kernel: SNR-mismatch screening (protocol E4), CPU-only, no GPU needed -- pure inference
on already-trained checkpoints (full_medium_fold{0-3}_seed0), not training. Moved here from a
local CPU run (too slow to be worth blocking a local machine for ~4-6h) onto Kaggle so it can run
alongside the GPU training kernels. Stages two datasets: the usual s2ships-processed data, plus a
small (~48MB) checkpoint-only dataset (abhinavp10/satrestore-mismatch-checkpoints) holding the 4
fold-seed0 best.pt files flattened to <run_name>_best.pt.
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
SESSION_BUDGET_HOURS = 8.0


def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find_dataset_dir(marker):
    for root, dirs, files in os.walk(INPUT_ROOT):
        if marker in files:
            return Path(root)
    raise FileNotFoundError(f"could not find {marker} under {INPUT_ROOT}")


def main():
    print("contents of /kaggle/input:", list(INPUT_ROOT.iterdir()), flush=True)
    if REPO_DIR.exists():
        shutil.rmtree(REPO_DIR)
    run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
    os.chdir(REPO_DIR)

    run(["pip", "install", "-q", "rasterio==1.5.1"])

    dataset_dir = find_dataset_dir("patch_index.csv")
    ckpt_dir = find_dataset_dir("full_medium_fold0_seed0_best.pt")
    print("using data dir:", dataset_dir, "/ checkpoint dir:", ckpt_dir, flush=True)

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

    log_path = Path("/kaggle/working/mismatch_run.log")
    cmd = ["python", "-u", "scripts/run_mismatch_screen.py", "--checkpoint-dir", str(ckpt_dir),
           "--flat-names", "--out", "db/results_mismatch.sqlite",
           "--time-budget-hours", str(SESSION_BUDGET_HOURS)]
    with open(log_path, "a", buffering=1) as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"run_mismatch_screen.py exited with code {rc}; see mismatch_run.log")
    print("\nmismatch screen finished; db/results_mismatch.sqlite is under "
          "/kaggle/working/SatRestore-Eval", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERROR_FILE.write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        raise
