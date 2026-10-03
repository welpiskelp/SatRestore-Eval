"""Kaggle kernel: BM3D classical baseline on H1's EXACT protocol (all 16 tiles, all 9 SNR
levels, gaussian noise), so the BM3D-vs-conditioned-model comparison is finally apples-to-apples
(the original baseline_bm3d run used only 6 tiles x 5 SNR levels). CPU-only, no GPU needed.

At ~10 min/(tile, SNR) this is ~24h of CPU time -- too long for one ~8.5h Kaggle session, so this
is split into batches of tiles, same pattern as kaggle/train_h2_crossband.py: edit BATCH_TILES to
whichever tiles are still missing (check db/results_bm3d_matched.sqlite locally) and re-run
("Save & Run All") this same kernel across however many sessions it takes. Each session evaluates
its batch at all 9 SNR levels; download and merge db/results_bm3d_matched.sqlite after every run.
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

# Round 1: first 5 tiles (largest/most ship-relevant first). Trim/replace each round with
# whatever's still missing from db/results_bm3d_matched.sqlite.
BATCH_TILES = ["rotterdam1", "rotterdam2", "rotterdam3", "toulon", "brest1"]
SNR_LEVELS = ["0", "5", "10", "15", "20", "25", "30", "35", "40"]
SESSION_BUDGET_HOURS = 8.0


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

    log_path = Path("/kaggle/working/bm3d_matched_run.log")
    cmd = ["python", "-u", "scripts/eval_bm3d.py", "--out", "db/results_bm3d_matched.sqlite",
           "--run-name", "baseline_bm3d_matched", "--tiles", *BATCH_TILES, "--snr", *SNR_LEVELS,
           "--time-budget-hours", str(SESSION_BUDGET_HOURS)]
    with open(log_path, "a", buffering=1) as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            print(line, end="", flush=True)
            logf.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"eval_bm3d.py exited with code {rc}; see bm3d_matched_run.log")
    print("\nbm3d matched-protocol batch finished; db/results_bm3d_matched.sqlite is under "
          "/kaggle/working/SatRestore-Eval", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        ERROR_FILE.write_text(traceback.format_exc(), encoding="utf-8")
        print(traceback.format_exc(), flush=True)
        raise
