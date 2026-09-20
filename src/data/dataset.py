"""Training and validation patches with on-the-fly noise (torch Dataset).

Clean patches are sliced from the prepared tile arrays (data/processed/tiles, memory-mapped). Noise is
drawn per patch from the configured types and SNR range, seeded from (base_seed, patch, epoch), so runs
are reproducible. Everything is normalised by a fixed per-band scale (mean over tiles of the per-band
PSNR peak) so network inputs are O(1); the same scale converts predictions back to DN.

Conditioning vector (cond_mode):
  "sigma_snr": 24 values, [SNR_b / 40] and [(log10(sigma_b / scale_b) + 2) / 2] for the 12 bands
  "snr":       1 value, mean SNR / 40
  "none":      1 zero (unused by the model)
The vector is computed from the true noise parameters; mismatch experiments perturb it afterwards.
"""
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..degrade.noise import NOISE_TYPES, TRAIN_SNR_RANGE, degrade, degradation_seed, sample_train_snr
from .patches import PATCH

COND_DIMS = {"sigma_snr": 24, "snr": 1, "none": 1}
VAL_SNRS = (5.0, 15.0, 25.0, 35.0)


def load_band_scale(processed_dir) -> np.ndarray:
    tiles = json.loads((Path(processed_dir) / "signal_stats.json").read_text())["tiles"]
    return np.mean([t["peak"] for t in tiles.values()], axis=0).astype(np.float32)


def condition_vector(snr_db, sigma, scale, cond_mode: str) -> np.ndarray:
    snr = np.asarray(snr_db, dtype=np.float64)
    if cond_mode == "sigma_snr":
        sigma_n = np.asarray(sigma, dtype=np.float64) / scale
        return np.concatenate([snr / 40.0, (np.log10(sigma_n) + 2.0) / 2.0]).astype(np.float32)
    if cond_mode == "snr":
        return np.array([snr.mean() / 40.0], dtype=np.float32)
    if cond_mode == "none":
        return np.zeros(1, dtype=np.float32)
    raise ValueError(f"unknown cond_mode {cond_mode!r}")


class PatchDataset(Dataset):
    def __init__(self, processed_dir, tiles, mode="train", noise_types=("gaussian",), snr_range=TRAIN_SNR_RANGE,
                 base_seed=0, cond_mode="sigma_snr", shot_fraction=0.5, max_sat_frac=1.0, val_snrs=VAL_SNRS,
                 ship_patch_repeat=1):
        if mode not in ("train", "val"):
            raise ValueError("mode must be 'train' or 'val'")
        for nt in noise_types:
            if nt not in NOISE_TYPES:
                raise ValueError(f"unknown noise type {nt!r}")
        self.processed = Path(processed_dir)
        self.mode, self.noise_types, self.snr_range = mode, tuple(noise_types), snr_range
        self.base_seed, self.cond_mode, self.shot_fraction = base_seed, cond_mode, shot_fraction
        self.val_snrs, self.epoch = tuple(val_snrs), 0

        with open(self.processed / "patch_index.csv") as f:
            rows = [r for r in csv.DictReader(f) if r["tile"] in set(tiles) and float(r["sat_any_frac"]) <= max_sat_frac]
        # Optionally repeat patches that contain ships so the rare positives are seen more often.
        self.rows = [r for r in rows for _ in range(ship_patch_repeat if int(r["ship_pixels"]) > 0 else 1)]
        stats = json.loads((self.processed / "signal_stats.json").read_text())["tiles"]
        self.stats = {t: (np.asarray(s["power"]), np.asarray(s["mean"])) for t, s in stats.items()}
        self.scale = load_band_scale(self.processed)
        self._arrays: dict = {}

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _tile_arrays(self, tile):
        if tile not in self._arrays:
            d = self.processed / "tiles"
            self._arrays[tile] = (np.load(d / f"{tile}_image.npy", mmap_mode="r"),
                                  np.load(d / f"{tile}_ship.npy", mmap_mode="r"))
        return self._arrays[tile]

    def __getitem__(self, i):
        r = self.rows[i]
        tile, y, x = r["tile"], int(r["y"]), int(r["x"])
        image, ship = self._tile_arrays(tile)
        clean = np.asarray(image[:, y : y + PATCH, x : x + PATCH]).astype(np.float32)
        ship_p = np.asarray(ship[y : y + PATCH, x : x + PATCH]).astype(np.float32)

        key = self.epoch if self.mode == "train" else 0
        rng = np.random.default_rng(degradation_seed(self.base_seed, r["patch_id"], key))
        if self.mode == "train":
            snr = sample_train_snr(rng, *self.snr_range)
            noise_type = self.noise_types[int(rng.integers(len(self.noise_types)))]
        else:
            snr = self.val_snrs[i % len(self.val_snrs)]
            noise_type = self.noise_types[(i // len(self.val_snrs)) % len(self.noise_types)]
        power, mean = self.stats[tile]
        noisy, params = degrade(clean, power, mean, snr, noise_type, int(rng.integers(2**62)), self.shot_fraction)

        valid = ((clean != 0) & (clean != 65535)).astype(np.float32)
        s = self.scale[:, None, None]
        return {
            "noisy": torch.from_numpy(noisy / s),
            "clean": torch.from_numpy(clean / s),
            "valid": torch.from_numpy(valid),
            "ship": torch.from_numpy(ship_p[None]),
            "cond": torch.from_numpy(condition_vector(params["snr_db"], params["sigma"], self.scale, self.cond_mode)),
            "snr": torch.tensor(snr, dtype=torch.float32),
            "noise_type": torch.tensor(NOISE_TYPES.index(noise_type)),
        }
