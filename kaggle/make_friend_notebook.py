"""Builds a self-contained Kaggle .ipynb that a friend can upload to THEIR OWN Kaggle
account, press "Save & Run All" on, and that produces a results.zip you can either have
them send you directly, or pull yourself via `kaggle kernels output` if they make the
kernel public / add you as a viewer.

Usage: python kaggle/make_friend_notebook.py <out.ipynb> <label> <config1> [config2] ...
Example:
  python kaggle/make_friend_notebook.py kaggle/friend_notebooks/batch_A.ipynb "Batch A" \
      configs/e1_blind_fold3_seed1.json configs/run_full_fold0_seed2.json configs/run_full_fold1_seed2.json
"""
import json
import sys
from pathlib import Path

REPO_URL = "https://github.com/welpiskelp/SatRestore-Eval.git"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def build(label, configs):
    cfg_list_py = ",\n    ".join(f'"{c}"' for c in configs)

    cells = []

    cells.append(md(f"""# SatRestore training run — {label}

Thanks for lending GPU time! Steps to run this:

1. Click **Add Data** (right sidebar) and search for **`s2ships-processed`** (owner:
   `abhinavp10`). If you can't find it, the owner hasn't shared it with your account yet —
   ping them first.
2. In the **Settings** panel (right sidebar): turn **Accelerator** to a GPU (T4 x2 or P100),
   and turn **Internet** ON.
3. Click **Save Version -> Save & Run All (Commit)**. It will take a few hours — you can
   close the tab, Kaggle keeps it running.
4. When it finishes, go to the notebook's **Output** tab and download `results.zip`, then
   send that file back. (Or: make the notebook public / share it, and the owner can pull it
   directly with the Kaggle API.)

No local setup needed — everything below runs inside Kaggle."""))

    cells.append(code(f"""import json, os, shutil, subprocess, time, zipfile
from pathlib import Path

REPO_URL = "{REPO_URL}"
REPO_DIR = Path("/kaggle/working/SatRestore-Eval")
INPUT_ROOT = Path("/kaggle/input")

SESSION_BUDGET_HOURS = 8.3
EST_HOURS_PER_RUN = 2.3

CONFIGS = [
    {cfg_list_py},
]

def run(cmd, **kw):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)

def find_dataset_dir():
    print("contents of /kaggle/input:", list(INPUT_ROOT.iterdir()), flush=True)
    for root, dirs, files in os.walk(INPUT_ROOT):
        if "patch_index.csv" in files:
            return Path(root)
    raise FileNotFoundError(
        f"could not find patch_index.csv under {{INPUT_ROOT}} -- did you Add Data for "
        "'s2ships-processed'? See step 1 above."
    )
"""))

    cells.append(code("""if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
run(["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)])
os.chdir(REPO_DIR)
run(["pip", "install", "-q", "rasterio==1.5.1"])
"""))

    cells.append(code("""dataset_dir = find_dataset_dir()
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
    (processed / "tiles").mkdir(exist_ok=True)
    with zipfile.ZipFile(dataset_dir / "tiles.zip") as z:
        z.extractall(processed / "tiles")
else:
    os.symlink(tiles_src, processed / "tiles")
n = len(list((processed / "tiles").glob("*")))
print(f"staged {{n}} tile files", flush=True)

import torch
print("cuda available:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-", flush=True)
"""))

    cells.append(code("""t0 = time.time()
done, skipped = [], []
for config in CONFIGS:
    elapsed_h = (time.time() - t0) / 3600
    if elapsed_h + EST_HOURS_PER_RUN > SESSION_BUDGET_HOURS:
        print(f"[{elapsed_h:.2f}h elapsed] not enough session time left for {config}, stopping here", flush=True)
        skipped.append(config)
        continue
    print(f"[{elapsed_h:.2f}h elapsed] starting {config}", flush=True)
    run(["python", "scripts/train.py", config])
    run_name = json.loads(Path(config).read_text())["run_name"]
    ckpt = REPO_DIR / "runs" / run_name / "best.pt"
    run(["python", "scripts/evaluate_run.py", "--checkpoint", str(ckpt), "--out", "db/results_pilot_full.sqlite"])
    done.append(config)

print("\\n=== batch summary ===", flush=True)
print("completed:", done, flush=True)
print("left unfinished (ran out of session time):", skipped, flush=True)
"""))

    cells.append(code("""# package everything needed to merge these results back, into one zip in /kaggle/working
out_zip = Path("/kaggle/working/results.zip")
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    db_path = REPO_DIR / "db" / "results_pilot_full.sqlite"
    if db_path.exists():
        z.write(db_path, "results_pilot_full.sqlite")
    for run_dir in (REPO_DIR / "runs").glob("*"):
        for f in ("best.pt", "metrics.jsonl", "config.json"):
            fp = run_dir / f
            if fp.exists():
                z.write(fp, f"runs/{run_dir.name}/{f}")
print("wrote", out_zip, out_zip.stat().st_size / 1e6, "MB")
print("\\nDone. Go to the Output tab of this notebook and download results.zip, then send it back.")
"""))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    out_path, label, *configs = sys.argv[1:]
    nb = build(label, configs)
    Path(out_path).write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print(f"wrote {out_path} ({len(configs)} configs)")
