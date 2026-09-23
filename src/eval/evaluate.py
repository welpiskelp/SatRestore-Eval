"""Evaluate a trained model on the fixed test noise and record the results.

For each test tile, noise type and SNR the fixed noisy tile (src.degrade.noise.test_noisy_tile) is
restored with predict_tile and scored by region plus per ship (src.eval.metrics). Conditions already in
the results database for the run are skipped, so an interrupted evaluation resumes where it stopped.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.tiles import load_ship_centroids
from ..degrade.noise import NOISE_TYPES, TEST_SNR_LEVELS, test_noisy_tile
from ..degrade.stats import load_signal_stats
from ..models.registry import build_any
from . import results
from .metrics import instance_metrics, region_masks, tile_metrics
from .predict import predict_tile, tile_condition


def load_checkpoint(path, device="cpu"):
    """Returns (model in eval mode, config dict, band scale)."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    m = cfg["model"]
    model = build_any(m.get("arch", "reconnet"), m["size"], m["cond_mode"], m.get("cross_band", False))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), cfg, np.asarray(ck["scale"], dtype=np.float32)


def run_name_for(base: str, offset_db: float) -> str:
    return base if offset_db == 0 else f"{base}_offset{offset_db:+g}dB"


def _save_crops(path, clean, noisy, pred, centroids, size=48, n=6):
    h, w = clean.shape[1:]
    crops = {"clean": [], "noisy": [], "pred": [], "xy": []}
    for cx, cy in centroids[:: max(1, len(centroids) // n)][:n]:
        y0 = int(np.clip(cy - size // 2, 0, h - size))
        x0 = int(np.clip(cx - size // 2, 0, w - size))
        for k, arr in (("clean", clean), ("noisy", noisy), ("pred", pred)):
            crops[k].append(arr[:, y0 : y0 + size, x0 : x0 + size].astype(np.float32))
        crops["xy"].append((x0, y0))
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in crops.items()})


def evaluate_model(model, scale, cond_mode, tiles, out_db, run_name, model_label, *, noise_types=NOISE_TYPES,
                   snr_levels=TEST_SNR_LEVELS, offset_db=0.0, processed_dir="data/processed",
                   raw_dir="data/S2-SHIPS/S2SHIPS", device="cpu", batch_size=16, fold=None, seed=None,
                   config=None, crops_dir=None, crop_conditions=(), log=print):
    processed = Path(processed_dir)
    stats = load_signal_stats(processed / "signal_stats.json")
    peaks = {t: s["peak"] for t, s in json.loads((processed / "signal_stats.json").read_text())["tiles"].items()}
    centroids = load_ship_centroids(Path(raw_dir))

    conn = results.connect(out_db)
    row = conn.execute("SELECT run_id FROM runs WHERE name = ?", (run_name,)).fetchone()
    run_id = row[0] if row else results.add_run(
        conn, run_name, model_label, fold=fold, seed=seed, config_json=json.dumps(config) if config else None,
        notes=f"cond offset {offset_db:+g} dB" if offset_db else None)
    done = {(t, n, s) for t, n, s in conn.execute(
        "SELECT DISTINCT tile, noise_type, snr_db FROM metrics WHERE run_id = ?", (run_id,))}

    t0 = time.time()
    n_new = 0
    for tile in tiles:
        todo = [(n, s) for n in noise_types for s in snr_levels if (tile, n, float(s)) not in done]
        if not todo:
            continue
        clean = np.load(processed / "tiles" / f"{tile}_image.npy")
        ship = np.load(processed / "tiles" / f"{tile}_ship.npy")
        water = np.load(processed / "tiles" / f"{tile}_water.npy")
        regions = region_masks(ship, water)
        s_ = stats[tile]
        ids = [f"{tile}_s{i:03d}" for i in range(len(centroids[tile]))]
        for noise_type, snr in todo:
            noisy, params = test_noisy_tile(clean, s_["power"], s_["mean"], tile, snr, noise_type)
            cond = tile_condition(params, scale, cond_mode, offset_db)
            pred = predict_tile(model, noisy, cond, scale, device, batch_size)
            results.add_metrics(conn, run_id, tile, noise_type, float(snr), tile_metrics(clean, pred, peaks[tile], regions))
            inst = instance_metrics(clean, pred, peaks[tile], centroids[tile], ship)
            for metric, values in inst.items():
                results.add_instance_metrics(conn, run_id, noise_type, float(snr), metric, dict(zip(ids, values)))
            if crops_dir and f"{noise_type}:{snr:g}" in crop_conditions:
                Path(crops_dir).mkdir(parents=True, exist_ok=True)
                _save_crops(Path(crops_dir) / f"{run_name}_{tile}_{noise_type}_{snr:g}dB.npz", clean, noisy, pred, centroids[tile])
            n_new += 1
            log(f"[{time.time() - t0:6.0f}s] {run_name} {tile:11s} {noise_type:19s} {snr:5.1f} dB done")
    conn.close()
    return n_new
