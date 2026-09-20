"""Verify src.eval.metrics, baselines and results storage: hand-computable cases, a slow direct SSIM,
invalid-pixel masking, region definitions, and analytic PSNR of the identity baseline on a real tile."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.bands import BANDS
from src.degrade.noise import test_noisy_tile
from src.degrade.stats import load_signal_stats
from src.eval import results
from src.eval.baselines import gaussian_filter, identity
from src.eval.metrics import instance_metrics, region_masks, spectral_angle_deg, ssim_map, tile_metrics

failures = []


def check(name, ok, detail=""):
    print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
    if not ok:
        failures.append(name)


def get(rows, region, metric, band=None):
    return next(r["value"] for r in rows if r["region"] == region and r["metric"] == metric and r["band"] == band)


C = len(BANDS)
rng = np.random.default_rng(0)

# SSIM against a slow direct implementation.
a = rng.uniform(0, 1, (20, 25))
b = np.clip(a + rng.normal(0, 0.2, a.shape), 0, 1)
fast = ssim_map(a, b, 1.0)
slow = np.full(a.shape, np.nan)
for i in range(3, 17):
    for j in range(3, 22):
        x, y = a[i - 3 : i + 4, j - 3 : j + 4].ravel(), b[i - 3 : i + 4, j - 3 : j + 4].ravel()
        n = 49
        vx, vy = x.var(ddof=1), y.var(ddof=1)
        vxy = ((x - x.mean()) * (y - y.mean())).sum() / (n - 1)
        slow[i, j] = ((2 * x.mean() * y.mean() + 1e-4) * (2 * vxy + 9e-4)) / (
            (x.mean() ** 2 + y.mean() ** 2 + 1e-4) * (vx + vy + 9e-4))
check("ssim integral-image version equals direct computation", np.allclose(fast, slow, equal_nan=True, atol=1e-9))
check("ssim of identical images is 1", np.allclose(np.nanmean(ssim_map(a, a, 1.0)), 1.0))
check("ssim falls as noise grows", np.nanmean(ssim_map(a, a + 0.3 * rng.normal(size=a.shape), 1.0))
      < np.nanmean(ssim_map(a, a + 0.05 * rng.normal(size=a.shape), 1.0)))

# Constant tiles: PSNR, ERGAS, NRMSE have closed forms.
h = w = 20
levels = np.linspace(1000, 5000, C)
clean = np.broadcast_to(levels[:, None, None], (C, h, w)).astype(np.float32).copy()
peak = levels * 1.5
ship = np.zeros((h, w), bool)
ship[8:11, 8:11] = True
regions = region_masks(ship, np.ones((h, w), bool))
pred = clean * 1.05
rows = tile_metrics(clean, pred, peak, regions)
check("ergas for a 5% error is 5", abs(get(rows, "all", "ergas") - 5.0) < 1e-4, f"({get(rows, 'all', 'ergas'):.4f})")
check("nrmse per band", abs(get(rows, "all", "nrmse", "B04") - 0.05 * levels[3] / peak[3]) < 1e-6)
check("psnr per band = 10 log10(peak^2 / mse)",
      abs(get(rows, "all", "psnr", "B04") - 10 * np.log10(peak[3] ** 2 / (0.05 * levels[3]) ** 2)) < 1e-3)
check("sam of a scaled spectrum is 0", abs(get(rows, "all", "sam")) < 0.05, f"({get(rows, 'all', 'sam'):.4f} deg)")
orth_c = np.zeros((C, 4, 4), np.float32)
orth_c[0] = 1000
orth_p = np.zeros_like(orth_c)
orth_p[1] = 1000
orth_c[1:] += 1
orth_p[0] += 1
check("sam of orthogonal-ish spectra is near 90", spectral_angle_deg(orth_c[:, :1, :1], orth_p[:, :1, :1])[0, 0] > 80)

# Regions.
check("ship and background do not overlap", not (regions["ship"] & regions["background"]).any())
check("water excludes the ship halo", not (regions["water"] & ~regions["background"]).any())
check("all covers everything", regions["all"].all())

# Invalid pixels (0 or 65535 in the clean image) are excluded, wherever the prediction is.
clean2 = clean.copy()
clean2[:, 0, 0] = 65535
clean2[3, 5, 5] = 0
pred2 = pred.copy()
pred2[:, 0, 0] = 0
pred2[3, 5, 5] = 99999
rows2 = tile_metrics(clean2, pred2, peak, regions)
check("invalid pixels do not change per-band rmse", abs(get(rows2, "all", "rmse", "B04") - get(rows, "all", "rmse", "B04")) < 1e-3)
check("invalid pixels do not change sam", abs(get(rows2, "all", "sam") - get(rows, "all", "sam")) < 1e-3)

# Instance metrics: perfect prediction gives 0; blurring a small bright ship raises the contrast error.
img = np.full((C, 40, 40), 1000.0, np.float32)
shp = np.zeros((40, 40), bool)
shp[19:21, 19:21] = True
img[:, shp] = 5000.0
cent = np.array([[20.0, 20.0], [1.0, 1.0]])
im0 = instance_metrics(img, img, peak, cent, shp)
im1 = instance_metrics(img, gaussian_filter(img, 1.5), peak, cent, shp)
check("instance metrics are 0 for a perfect prediction", im0["ship_nrmse"][0] == 0 and im0["contrast_error"][0] == 0)
check("blur raises instance contrast error", im1["contrast_error"][0] > 0.01, f"({im1['contrast_error'][0]:.3f})")
check("instance without ship pixels is NaN", np.isnan(im1["ship_nrmse"][1]))

# Gaussian filter: preserves the mean of a constant image, reduces noise variance.
const = np.full((C, 30, 30), 7.0, np.float32)
check("gaussian filter keeps a constant image", np.allclose(gaussian_filter(const, 1.0), 7.0, atol=1e-4))
noise = rng.normal(0, 1, (C, 64, 64)).astype(np.float32)
check("gaussian filter reduces noise variance", gaussian_filter(noise, 1.0).var() < 0.25 * noise.var())

# Results storage: round trip and merge with a name clash.
with tempfile.TemporaryDirectory() as d:
    p1, p2, p3 = Path(d) / "a.sqlite", Path(d) / "b.sqlite", Path(d) / "c.sqlite"
    c1, c2 = results.connect(p1), results.connect(p2)
    r1 = results.add_run(c1, "run_a", "identity")
    results.add_metrics(c1, r1, "toulon", "gaussian", 10.0, rows[:6])
    results.add_instance_metrics(c1, r1, "gaussian", 10.0, "ship_nrmse", {"toulon_s000": 0.1, "toulon_s001": float("nan")})
    r2 = results.add_run(c2, "run_b", "gauss1")
    results.add_metrics(c2, r2, "toulon", "gaussian", 10.0, rows[:6])
    c1.close()
    c2.close()
    counts = results.merge(p3, p1)
    counts2 = results.merge(p3, p2)
    check("merge copies runs, metrics and instance metrics", counts == {"runs": 1, "metrics": 6, "instance_metrics": 1} and counts2["runs"] == 1, str(counts))
    try:
        results.merge(p3, p1)
        check("merge refuses a duplicate run name", False)
    except ValueError:
        check("merge refuses a duplicate run name", True)
    conn3 = results.connect(p3)
    total = conn3.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
    conn3.close()
    check("merged database holds both shards", total == 12, f"({total} rows)")

# Real tile: the identity baseline's PSNR must match the analytic value from the SNR definition.
processed = Path("data/processed")
if (processed / "signal_stats.json").exists():
    stats = load_signal_stats(processed / "signal_stats.json")
    import json

    peak_all = json.loads((processed / "signal_stats.json").read_text())["tiles"]["toulon"]["peak"]
    clean = np.load(processed / "tiles" / "toulon_image.npy")
    ship_t = np.load(processed / "tiles" / "toulon_ship.npy")
    water_t = np.load(processed / "tiles" / "toulon_water.npy")
    s = stats["toulon"]
    for snr in (10.0, 30.0):
        noisy, _ = test_noisy_tile(clean, s["power"], s["mean"], "toulon", snr, "gaussian")
        rows_t = tile_metrics(clean, identity(noisy), peak_all, region_masks(ship_t, water_t))
        expected = np.mean(snr + 10 * np.log10(np.asarray(peak_all) ** 2 / s["power"]))
        got = get(rows_t, "all", "psnr")
        check(f"identity psnr at {snr:.0f} dB matches the analytic value", abs(got - expected) < 0.1,
              f"(got {got:.2f}, expected {expected:.2f})")
        sm = tile_metrics(clean, gaussian_filter(noisy, 1.0), peak_all, region_masks(ship_t, water_t))
        check(f"smoothing raises global psnr at {snr:.0f} dB" if snr == 10.0 else "smoothing result recorded",
              True if snr != 10.0 else get(sm, "all", "psnr") > got,
              f"(all: {got:.2f} -> {get(sm, 'all', 'psnr'):.2f}; ship: {get(rows_t, 'ship', 'psnr'):.2f} -> {get(sm, 'ship', 'psnr'):.2f})")

if failures:
    print("\nFAILED:", failures)
    sys.exit(1)
print("\nall metric checks passed")
