"""Synthetic noise at a target per-band SNR.

SNR_b = 10*log10(P_b / N_b), where P_b is the tile-level signal power (src.degrade.stats) and N_b
is the average noise power (variance) added to band b. All noise types are scaled to the same N_b,
so a nominal SNR means the same thing for each:

  gaussian:            y = x + n,                       n ~ N(0, N_b), independent per pixel
  poisson_gaussian:    y = a*Poisson(x/a) + n,          n ~ N(0, (1-rho)*N_b), a = rho*N_b / mean_b
                       pixel variance a*x + (1-rho)*N_b, whose average over the tile is N_b.
  correlated_gaussian: y = x + n,                       n ~ N(0, N_b) with spatial correlation
                       (Gaussian sd CORRELATION_SIGMA_PX per band): noise in the 20 m and 60 m bands
                       is smooth after resampling to the 10 m grid, unlike independent pixel noise.

rho (shot_fraction) is the share of N_b from signal-dependent shot noise. It is an assumption, not
calibrated to the sensor. Output is float32 in the input's digital-number units and is not clipped,
so the realised SNR is not distorted at low signal levels.

SNR range: the reference images carry their own noise, estimated at about 47-72 dB in this
convention (median 56 dB), and ESA's requirements put real Sentinel-2 at about 40-45 dB. Levels of
35-40 dB are therefore near nominal sensor noise, and 0-20 dB is a stress regime.
"""
import hashlib

import numpy as np

from ..data.bands import CORRELATION_SIGMA_PX

NOISE_TYPES = ("gaussian", "poisson_gaussian", "correlated_gaussian")
TRAIN_SNR_RANGE = (5.0, 40.0)
TEST_SNR_LEVELS = (0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0)
TEST_BASE_SEED = 0


def degradation_seed(base_seed: int, patch_id: str, key: int) -> int:
    """Stable (platform- and process-independent) seed for one patch and one degradation draw.
    key is round(snr * 100) for fixed test levels or a draw counter for training.
    63-bit, so it fits a signed SQLite INTEGER."""
    digest = hashlib.sha256(f"{base_seed}|{patch_id}|{key}".encode()).digest()
    return int.from_bytes(digest[:8], "little") >> 1


def sample_train_snr(rng: np.random.Generator, low: float = TRAIN_SNR_RANGE[0], high: float = TRAIN_SNR_RANGE[1]) -> float:
    return float(rng.uniform(low, high))


def noise_params(power, mean, snr_db, noise_type: str, seed: int, shot_fraction: float = 0.5) -> dict:
    """Deterministic noise parameters for a target SNR (no random numbers involved).
    power, mean: (C,) tile statistics. snr_db: scalar (same for every band) or (C,).
    sigma is the per-band noise standard deviation sqrt(N_b); gain and sigma_read are the
    Poisson-Gaussian components (gain is 0 for gaussian noise)."""
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
    power = np.asarray(power, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    snr = np.broadcast_to(np.asarray(snr_db, dtype=np.float64), power.shape)
    noise_power = power / 10.0 ** (snr / 10.0)

    corr = np.zeros_like(power)
    if noise_type == "poisson_gaussian":
        gain = shot_fraction * noise_power / mean
        sigma_read = np.sqrt((1.0 - shot_fraction) * noise_power)
    else:
        gain = np.zeros_like(power)
        sigma_read = np.sqrt(noise_power)
        if noise_type == "correlated_gaussian":
            corr = np.asarray(CORRELATION_SIGMA_PX, dtype=np.float64)
            if corr.shape != power.shape:
                raise ValueError("correlated_gaussian is defined for the 12 canonical bands")

    return {
        "noise_type": noise_type,
        "seed": int(seed),
        "snr_db": snr.tolist(),
        "sigma": np.sqrt(noise_power).tolist(),
        "gain": gain.tolist(),
        "sigma_read": sigma_read.tolist(),
        "corr_sigma_px": corr.tolist(),
        "shot_fraction": shot_fraction if noise_type == "poisson_gaussian" else 0.0,
    }


def _correlated_unit_noise(rng: np.random.Generator, shape, sigma_px) -> np.ndarray:
    """(C, H, W) zero-mean, unit-variance Gaussian noise; band c is white noise filtered by a Gaussian
    of sd sigma_px[c] pixels (periodic boundaries), rescaled so its expected variance stays 1."""
    c, h, w = shape
    out = rng.standard_normal(shape)
    fy = np.fft.fftfreq(h)[:, None]
    for i in range(c):
        s = float(sigma_px[i])
        if s <= 0:
            continue
        half = np.exp(-2 * np.pi**2 * s**2 * (fy**2 + np.fft.rfftfreq(w)[None, :] ** 2))
        full = np.exp(-2 * np.pi**2 * s**2 * (fy**2 + np.fft.fftfreq(w)[None, :] ** 2))
        out[i] = np.fft.irfft2(np.fft.rfft2(out[i]) * half, s=(h, w)) / np.sqrt(np.mean(full**2))
    return out


def degrade(clean, power, mean, snr_db, noise_type: str, seed: int, shot_fraction: float = 0.5):
    """clean: (C, H, W) array in DN units. power, mean: (C,) tile statistics for the bands.
    snr_db: scalar (same SNR for every band) or (C,). Returns (noisy float32, params dict).
    params holds everything needed to log or reproduce the degradation."""
    params = noise_params(power, mean, snr_db, noise_type, seed, shot_fraction)
    clean = np.asarray(clean, dtype=np.float64)
    gain = np.asarray(params["gain"])
    sigma_read = np.asarray(params["sigma_read"])
    if clean.ndim != 3 or clean.shape[0] != gain.shape[0]:
        raise ValueError(f"clean {clean.shape} does not match {gain.shape[0]} bands")

    rng = np.random.default_rng(seed)
    chan = (slice(None), None, None)
    if noise_type == "gaussian":
        noisy = clean + rng.standard_normal(clean.shape) * sigma_read[chan]
    elif noise_type == "correlated_gaussian":
        noisy = clean + _correlated_unit_noise(rng, clean.shape, params["corr_sigma_px"]) * sigma_read[chan]
    else:
        noisy = gain[chan] * rng.poisson(clean / gain[chan]) + rng.standard_normal(clean.shape) * sigma_read[chan]
    return noisy.astype(np.float32), params


def test_noisy_tile(image, power, mean, tile: str, snr_db: float, noise_type: str,
                    shot_fraction: float = 0.5, base_seed: int = TEST_BASE_SEED):
    """The fixed test degradation of a whole tile: one noisy realisation per (tile, noise type, SNR),
    shared by every model. Test patches are crops of this array, so overlapping patches agree on
    their noise and full-tile stitched evaluation is consistent."""
    seed = degradation_seed(base_seed, tile, round(snr_db * 100))
    return degrade(image, power, mean, snr_db, noise_type, seed, shot_fraction)
