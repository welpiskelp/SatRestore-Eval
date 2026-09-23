"""One place to build any model from a config: model.arch, model.size, model.cond_mode, model.cross_band."""
from .baselines import DnCNN, FFDNetLike, SwinIRLite
from .nafnet import COND_DIMS, build_model, count_params

# Sizes per baseline architecture ("base" is the size compared against the NAFNet-family "medium").
BASELINE_SIZES = {
    "dncnn": {"tiny": dict(width=8, depth=4), "small": dict(width=32, depth=8),
              "medium": dict(width=64, depth=17), "base": dict(width=128, depth=17)},
    "ffdnet": {"tiny": dict(width=16, depth=4), "small": dict(width=48, depth=8),
               "medium": dict(width=96, depth=12), "base": dict(width=128, depth=15)},
    "swinir": {"tiny": dict(dim=12, depths=(1, 1), heads=2), "small": dict(dim=36, depths=(2, 2), heads=6),
               "medium": dict(dim=60, depths=(4, 4, 4, 4), heads=6), "base": dict(dim=96, depths=(4, 4, 4, 4), heads=6)},
}
BLIND = {"dncnn", "swinir"}
ARCHS = ("reconnet", "dncnn", "ffdnet", "swinir", "matched_control")


def build_any(arch: str, size: str, cond_mode: str, cross_band: bool = False, **overrides):
    """arch 'reconnet' is the NAFNet-family model (cond_mode and cross_band are its ablation switches).
    'dncnn' and 'swinir' are blind and require cond_mode 'none'; 'ffdnet' needs a conditioning vector.
    'matched_control' is the NAFNet backbone only, widened to match the parameter count of the full
    ReconNet of the same size (pass target_params to override)."""
    if arch == "reconnet":
        return build_model(size, cond_mode, cross_band, **overrides)
    if arch == "matched_control":
        if cond_mode != "none":
            raise ValueError("matched_control is blind: cond_mode must be 'none'")
        return build_matched_control(size, overrides.pop("target_params", None))
    if arch not in BASELINE_SIZES:
        raise ValueError(f"unknown arch {arch!r}; choose from {ARCHS}")
    if size == "matched":  # a baseline widened to the full ReconNet's parameter count
        if arch in BLIND and cond_mode != "none":
            raise ValueError(f"{arch} is blind: set model.cond_mode to 'none'")
        return build_param_matched(arch, cond_mode, overrides.pop("target_params", None))
    if arch in BLIND and cond_mode != "none":
        raise ValueError(f"{arch} is blind: set model.cond_mode to 'none'")
    if arch == "ffdnet" and cond_mode == "none":
        raise ValueError("ffdnet needs a conditioning vector: use cond_mode 'sigma_snr' or 'snr'")
    cfg = dict(BASELINE_SIZES[arch][size], **overrides)
    if arch == "dncnn":
        return DnCNN(**cfg)
    if arch == "ffdnet":
        return FFDNetLike(cond_dim=COND_DIMS[cond_mode], **cfg)
    return SwinIRLite(**cfg)


def build_param_matched(arch: str, cond_mode: str, target_params: int | None = None):
    """A baseline (dncnn, ffdnet or swinir) widened until its parameter count is as close as possible to
    target_params (default: the full ReconNet 'medium'), so that model size cannot explain a difference."""
    if arch not in BASELINE_SIZES:
        raise ValueError(f"unknown baseline {arch!r}")
    target = target_params or count_params(build_model("medium", "sigma_snr", True))
    key, step = ("dim", 6) if arch == "swinir" else ("width", 1)
    best = None
    for v in range(step * 2, 400, step):
        model = build_any(arch, "base", cond_mode, **{key: v})
        gap = abs(count_params(model) - target)
        if best is None or gap < best[0]:
            best = (gap, model)
    return best[1]


def build_matched_control(size: str = "medium", target_params: int | None = None):
    """NAFNet backbone without FiLM or cross-band attention, with the width chosen so that its parameter
    count is as close as possible to the full model's (default: the full ReconNet of the same size).
    Separates the effect of the added modules from the effect of simply having more parameters."""
    target = target_params or count_params(build_model(size, "sigma_snr", True))
    best = None
    for width in range(4, 200):
        model = build_model(size, "none", False, width=width)
        gap = abs(count_params(model) - target)
        if best is None or gap < best[0]:
            best = (gap, width, model)
    return best[2]
