"""Resumable training loop.

One config dict describes a run (see configs/*.json). The loop writes, into <out_dir>/<run_name>/:
  last.pt        model, optimiser, scheduler, scaler, epoch, best metric: written every epoch; a run resumed
                 from it continues exactly as an uninterrupted run would (data order and noise are seeded
                 from the seed and the epoch)
  best.pt        model weights with the lowest validation L1 (fixed validation noise, so comparable)
  metrics.jsonl  one line per epoch
Splits come from configs/splits.json: train tiles for training, val tiles for checkpoint selection.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..data.dataset import PatchDataset
from ..losses.recon import ReconLoss
from ..models.registry import build_any


def _subset(ds: PatchDataset, n, seed):
    if n and n < len(ds.rows):
        idx = np.random.default_rng(seed).choice(len(ds.rows), n, replace=False)
        ds.rows = [ds.rows[i] for i in sorted(idx)]
    return ds


def _to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def build_run(cfg: dict):
    splits = json.loads(Path(cfg.get("splits", "configs/splits.json")).read_text())
    fold = splits["folds"][cfg["fold"]]
    d, m = cfg["data"], cfg["model"]
    common = dict(processed_dir=d["processed_dir"], noise_types=d["noise_types"], base_seed=cfg["seed"],
                  cond_mode=m["cond_mode"], shot_fraction=d.get("shot_fraction", 0.5),
                  max_sat_frac=d.get("max_sat_frac", 1.0))
    train_ds = _subset(PatchDataset(tiles=fold["train"], mode="train", snr_range=tuple(d["snr_range"]),
                                    ship_patch_repeat=d.get("ship_patch_repeat", 1), **common),
                       d.get("max_train_patches"), cfg["seed"])
    val_ds = _subset(PatchDataset(tiles=fold["val"], mode="val", **common), d.get("max_val_patches"), cfg["seed"])
    model = build_any(m.get("arch", "reconnet"), m["size"], m["cond_mode"], m.get("cross_band", False))
    return train_ds, val_ds, model


@torch.no_grad()
def validate(model, val_loader, crit_ship, crit_l1, device):
    model.eval()
    l1 = ship = n = 0.0
    for batch in val_loader:
        batch = _to_device(batch, device)
        pred = model(batch["noisy"], batch["cond"])
        _, a = crit_l1(pred, batch)
        _, b = crit_ship(pred, batch)
        l1 += a["l1"]
        ship += b["ship"]
        n += 1
    return l1 / n, ship / n


def train(cfg: dict, resume: bool = True) -> dict:
    t_start = time.time()
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.get("device", "auto") != "cpu" else "cpu")
    o = cfg["optim"]
    run_dir = Path(cfg["out_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, model = build_run(cfg)
    model.to(device)
    lw = cfg["loss"]
    scale = train_ds.scale
    crit = ReconLoss(scale, lw.get("lambda_ship", 0.0), lw.get("lambda_sam", 0.0), lw.get("ship_dilate", 1)).to(device)
    crit_l1 = ReconLoss(scale).to(device)
    crit_ship = ReconLoss(scale, lambda_ship=1.0, ship_dilate=lw.get("ship_dilate", 1)).to(device)

    workers = cfg.get("num_workers", 0)
    val_loader = DataLoader(val_ds, batch_size=o["batch_size"], shuffle=False, num_workers=workers)
    steps_per_epoch = math.ceil(len(train_ds) / o["batch_size"])
    total_steps = steps_per_epoch * o["epochs"]
    opt = torch.optim.AdamW(model.parameters(), lr=o["lr"], weight_decay=o.get("weight_decay", 1e-4))
    warm = o.get("warmup_steps", 0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps)))) if warm
        else 0.5 * (1 + math.cos(math.pi * min(1.0, s / total_steps))))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best = 0, float("inf")
    last = run_dir / "last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from epoch {ck['epoch']} (best val l1 {best:.5f})", flush=True)

    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    for epoch in range(start_epoch, o["epochs"]):
        t0 = time.time()
        model.train()
        train_ds.set_epoch(epoch)
        gen = torch.Generator().manual_seed(cfg["seed"] * 1000 + epoch)
        loader = DataLoader(train_ds, batch_size=o["batch_size"], shuffle=True, generator=gen, num_workers=workers,
                            drop_last=False, persistent_workers=False)
        sums, count = {}, 0
        for step, batch in enumerate(loader):
            batch = _to_device(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                pred = model(batch["noisy"], batch["cond"])
            loss, parts = crit(pred.float(), batch)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if o.get("grad_clip"):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), o["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            count += 1
            if cfg.get("max_steps") and epoch * steps_per_epoch + step + 1 >= cfg["max_steps"]:
                break
        val_l1, val_ship = validate(model, val_loader, crit_ship, crit_l1, device)
        rec = {"epoch": epoch, "lr": sched.get_last_lr()[0], "val_l1": val_l1, "val_ship_l1": val_ship,
               "seconds": round(time.time() - t0, 1), **{f"train_{k}": v / max(count, 1) for k, v in sums.items()}}
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
        if val_l1 < best:
            best = val_l1
            torch.save({"model": model.state_dict(), "config": cfg, "scale": scale, "epoch": epoch, "val_l1": val_l1},
                       run_dir / "best.pt")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "scaler": scaler.state_dict(), "epoch": epoch, "best": best, "config": cfg}, last)
        # Graceful stops: the checkpoint above is complete, so the run resumes from here later.
        if cfg.get("stop_after_epoch") is not None and epoch >= cfg["stop_after_epoch"]:
            break
        if cfg.get("max_hours") and time.time() - t_start > cfg["max_hours"] * 3600:
            print("time budget reached; stopping after a complete epoch (resume to continue)", flush=True)
            break
    return {"run_dir": str(run_dir), "best_val_l1": best, "finished": epoch + 1 >= o["epochs"]}
