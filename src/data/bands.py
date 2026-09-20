"""Band layout for the S2-SHIPS tiles. BANDS is the single source of truth for channel order."""

# Canonical channel order used everywhere in this project (Sentinel-2 wavelength order).
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]

# Channel order of the 'data' array inside dataset_npy/*.npy as delivered; verified
# bit-for-bit against the per-band GeoTIFFs.
NPY_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11", "B12", "B8A"]

# Index list that reorders an npy channel axis into canonical order: canonical = npy[NPY_TO_CANONICAL].
NPY_TO_CANONICAL = [NPY_BANDS.index(b) for b in BANDS]

# Bands acquired natively at 10 m; the others were resampled to the 10 m grid.
NATIVE_10M = ["B02", "B03", "B04", "B08"]


# Native ground sampling distance (m) of each band before resampling to the 10 m grid.
NATIVE_RESOLUTION_M = {
    "B01": 60, "B02": 10, "B03": 10, "B04": 10, "B05": 20, "B06": 20,
    "B07": 20, "B08": 10, "B8A": 20, "B09": 60, "B11": 20, "B12": 20,
}

# Sensor noise is independent per native pixel, so after resampling to the 10 m grid it is correlated
# over about the native pixel size. Gaussian sd (in 10 m pixels) used by the correlated noise type:
# 0 for native 10 m bands, 1 for 20 m bands, 3 for 60 m bands (an approximation of the resampling kernel).
CORRELATION_SIGMA_PX = [
    NATIVE_RESOLUTION_M[b] / 10 / 2 if NATIVE_RESOLUTION_M[b] > 10 else 0.0 for b in BANDS
]


def band_index(name: str) -> int:
    return BANDS.index(name)
