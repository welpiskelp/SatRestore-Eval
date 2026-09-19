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


def band_index(name: str) -> int:
    return BANDS.index(name)
