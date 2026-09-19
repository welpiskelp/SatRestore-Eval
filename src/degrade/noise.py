"""Synthetic noise at a target per-band SNR.

SNR_b = 10*log10(P_b / N_b), where P_b is the tile-level signal power (src.degrade.stats) and N_b
is the average noise power (variance) added to band b. Both noise types are scaled to the same N_b,
so a nominal SNR means the same thing for either:

  gaussian:           y = x + n,                       n ~ N(0, N_b)
  poisson_gaussian:   y = a*Poisson(x/a) + n,          n ~ N(0, (1-rho)*N_b), a = rho*N_b / mean_b
                      pixel variance a*x + (1-rho)*N_b, whose average over the tile is N_b.

rho (shot_fraction) is the share of N_b from signal-dependent shot noise. It is an assumption, not
calibrated to the sensor. Output is float32 in the input's digital-number units and is not clipped,
so the realised SNR is not distorted at low signal levels.
"""
import hashlib

import numpy as np

NOISE_TYPES = ("gaussian", "poisson_gaussian")
TRAIN_SNR_RANGE = (5.0, 30.0)
TEST_SNR_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


def degradation_seed(base_seed: int, patch_id: str, key: int) -> int:
    """Stable (platform- and process-independent) seed for one patch and one degradation draw.
    key is round(snr * 100) for fixed test levels or a draw counter for training."""
    digest = hashlib.sha256(f"{base_seed}|{patch_id}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def sample_train_snr(rng: np.random.Generator, low: float = TRAIN_SNR_RANGE[0], high: float = TRAIN_SNR_RANGE[1]) -> float:
    return float(rng.uniform(low, high))


def degrade(clean, power, mean, snr_db, noise_type: str, seed: int, shot_fraction: float = 0.5):
    """clean: (C, H, W) array in DN units. power, mean: (C,) tile statistics for the bands.
    snr_db: scalar (same SNR for every band) or (C,). Returns (noisy float32, params dict).
    params holds everything needed to log or reproduce the degradation."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
    clean = np.asarray(clean, dtype=np.float64)
    power = np.asarray(power, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    if clean.ndim != 3 or clean.shape[0] != power.shape[0]:
        raise ValueError(f"clean {clean.shape} does not match {power.shape[0]} bands")

    snr = np.broadcast_to(np.asarray(snr_db, dtype=np.float64), power.shape)
    noise_power = power / 10.0 ** (snr / 10.0)
    rng = np.random.default_rng(seed)
    chan = (slice(None), None, None)

    if noise_type == "gaussian":
        gain = np.zeros_like(power)
        sigma_read = np.sqrt(noise_power)
        noisy = clean + rng.standard_normal(clean.shape) * sigma_read[chan]
    else:
        gain = shot_fraction * noise_power / mean
        sigma_read = np.sqrt((1.0 - shot_fraction) * noise_power)
        noisy = gain[chan] * rng.poisson(clean / gain[chan]) + rng.standard_normal(clean.shape) * sigma_read[chan]

    params = {
        "noise_type": noise_type,
        "seed": int(seed),
        "snr_db": snr.tolist(),
        "sigma": np.sqrt(noise_power).tolist(),
        "gain": gain.tolist(),
        "sigma_read": sigma_read.tolist(),
        "shot_fraction": shot_fraction if noise_type == "poisson_gaussian" else 0.0,
    }
    return noisy.astype(np.float32), params
