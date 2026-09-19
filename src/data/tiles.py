"""Readers for the raw S2-SHIPS download (data/S2-SHIPS/S2SHIPS)."""
import pickletools
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject

# The delivered dataset_npy files are pickled dicts. Unpickling can execute code, so the
# loader statically checks that the pickle imports nothing beyond numpy array internals.
_ALLOWED_PICKLE_GLOBALS = {
    "numpy dtype",
    "numpy ndarray",
    "numpy.core.multiarray _reconstruct",
    "numpy._core.multiarray _reconstruct",
}
_FORBIDDEN_OPCODES = {"STACK_GLOBAL", "INST", "OBJ"}


def discover_tiles(raw_dir: Path) -> list[str]:
    return sorted(p.name for p in (raw_dir / "dataset_tif").iterdir() if p.is_dir())


def find_band_file(tile_dir: Path, band: str) -> Path | None:
    # Filenames are inconsistent (B01 lacks the parentheses around "Raw").
    matches = [
        f for f in tile_dir.glob("*.tiff")
        if re.search(rf"_{band}(_|\()", f.name) and "Raw" in f.name
    ]
    return matches[0] if matches else None


def tile_date(tile_dir: Path) -> str:
    return next(tile_dir.glob("*.tiff")).name[:10]


def find_ship_npy(raw_dir: Path, tile: str) -> Path:
    matches = list((raw_dir / "dataset_npy").glob(f"*_mask_{tile}.npy"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one ship npy for {tile}, found {len(matches)}")
    return matches[0]


def load_pickled_npy(path: Path) -> dict:
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        header_reader = (
            np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        )
        header_reader(f)
        payload = f.read()
    for op, arg, _ in pickletools.genops(payload):
        if op.name in _FORBIDDEN_OPCODES:
            raise ValueError(f"{path}: refusing to unpickle, unexpected opcode {op.name}")
        if op.name == "GLOBAL" and arg not in _ALLOWED_PICKLE_GLOBALS:
            raise ValueError(f"{path}: refusing to unpickle, unexpected global {arg!r}")
    return np.load(path, allow_pickle=True).item()


def load_water_mask(water_tif: Path, reference_band_tif: Path) -> np.ndarray:
    """Water mask (0/255 on a 10.000 m grid, 1784 wide) resampled by nearest neighbour onto the
    imagery grid (10.0048 m, 1783 wide). Returns uint8 0/1 with the imagery's shape."""
    with rasterio.open(reference_band_tif) as ref, rasterio.open(water_tif) as src:
        out = np.zeros(ref.shape, dtype=np.uint8)
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            dst_transform=ref.transform,
            dst_crs=ref.crs,
            resampling=Resampling.nearest,
        )
    return (out > 0).astype(np.uint8)
