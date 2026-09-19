"""Verify the noise engine on real tiles: realised SNR versus target, signal dependence, determinism.

Realised SNR is measured per band over valid pixels (not 0, not saturated), using the same tile-level
P_b that defined the target. Exits non-zero if any check fails.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bands import BANDS, band_index
from src.data.patches import SATURATED
from src.degrade.noise import NOISE_TYPES, degrade, degradation_seed
from src.degrade.stats import load_signal_stats

TILES = ("toulon", "panama", "rotterdam3")  # clean, heavily saturated, has zero pixels
LEVELS = (0.0, 10.0, 30.0)
TOL_DB = 0.1


def realised_snr_db(clean, noisy, power):
    out = []
    for i in range(clean.shape[0]):
        x = clean[i].astype(np.float64)
        ok = (x != 0) & (x != SATURATED)
        noise_power = np.mean((noisy[i].astype(np.float64)[ok] - x[ok]) ** 2)
        out.append(10 * np.log10(power[i] / noise_power))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed")
    args = ap.parse_args()
    processed = Path(args.processed_dir)
    stats = load_signal_stats(processed / "signal_stats.json")
    failures = []

    print(f"{'tile':11s} {'noise':17s} {'target dB':>9s} {'worst |error| dB':>17s}  worst band")
    for tile in TILES:
        clean = np.load(processed / "tiles" / f"{tile}_image.npy")
        s = stats[tile]
        for noise_type in NOISE_TYPES:
            for snr in LEVELS:
                noisy, _ = degrade(clean, s["power"], s["mean"], snr, noise_type, seed=123)
                err = np.abs(realised_snr_db(clean, noisy, s["power"]) - snr)
                worst = int(np.argmax(err))
                print(f"{tile:11s} {noise_type:17s} {snr:9.1f} {err[worst]:17.4f}  {BANDS[worst]}")
                if err[worst] > TOL_DB:
                    failures.append(f"{tile} {noise_type} {snr} dB: error {err[worst]:.3f} dB in {BANDS[worst]}")

    # Signal dependence: Poisson-Gaussian noise power must grow with signal, Gaussian must not.
    clean = np.load(processed / "tiles" / "toulon_image.npy")
    s = stats["toulon"]
    b = band_index("B08")
    x = clean[b].astype(np.float64)
    ok = (x != 0) & (x != SATURATED)
    lo = ok & (x <= np.median(x[ok]))
    hi = ok & (x > np.median(x[ok]))
    print("\nnoise power, bright half / dark half of B08 (toulon, 10 dB):")
    for noise_type, want_high in (("gaussian", False), ("poisson_gaussian", True)):
        noisy, _ = degrade(clean, s["power"], s["mean"], 10.0, noise_type, seed=1)
        d = noisy[b].astype(np.float64) - x
        ratio = np.mean(d[hi] ** 2) / np.mean(d[lo] ** 2)
        print(f"  {noise_type:17s} ratio {ratio:6.2f}")
        if want_high and ratio < 1.5:
            failures.append(f"poisson_gaussian noise not signal dependent (ratio {ratio:.2f})")
        if not want_high and not 0.9 < ratio < 1.1:
            failures.append(f"gaussian noise not signal independent (ratio {ratio:.2f})")

    # Determinism and seeding.
    patch = clean[:, :128, :128]
    a, pa = degrade(patch, s["power"], s["mean"], 12.5, "poisson_gaussian", seed=7)
    b2, _ = degrade(patch, s["power"], s["mean"], 12.5, "poisson_gaussian", seed=7)
    c, _ = degrade(patch, s["power"], s["mean"], 12.5, "poisson_gaussian", seed=8)
    if not np.array_equal(a, b2):
        failures.append("same seed gave different noise")
    if np.array_equal(a, c):
        failures.append("different seeds gave identical noise")
    seeds = {degradation_seed(0, pid, k) for pid in ("toulon_y0000_x0000", "toulon_y0000_x0064") for k in (0, 1250)}
    if len(seeds) != 4 or degradation_seed(0, "toulon_y0000_x0000", 0) != degradation_seed(0, "toulon_y0000_x0000", 0):
        failures.append("degradation_seed not stable/distinct")
    per_band = np.linspace(5, 30, len(BANDS))
    _, pb = degrade(patch, s["power"], s["mean"], per_band, "gaussian", seed=1)
    if not np.allclose(pb["snr_db"], per_band):
        failures.append("per-band SNR array not honoured")
    print("\ndeterminism / seeding / per-band SNR checks done")

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print(f"\nall checks passed (realised SNR within {TOL_DB} dB of target)")


if __name__ == "__main__":
    main()
