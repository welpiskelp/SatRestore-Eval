"""NAFNet-style restoration network for 12-band Sentinel-2 patches, with two optional additions.

Backbone: a U-Net of NAFBlocks (nonlinear-activation-free blocks: depthwise conv, simple gate,
simplified channel attention), predicting a residual that is added to the noisy input.

Additions (ablation switches):
- FiLM conditioning (cond_dim > 0): a small MLP embeds the conditioning vector (per-band SNR and
  noise sigma from the dataset, or a scalar SNR); every block scales and shifts its normalised
  features by gamma(embedding), beta(embedding). The FiLM layers and the output head start at zero, so
  the untrained network is exactly the identity mapping.
- Cross-band attention (cross_band=True): the 12 bands are the tokens. Each band is embedded to
  cb_dim feature channels by a shared convolution plus a learned band embedding; attention is
  computed between bands (12 x 12, from queries and keys formed over the whole feature map) and mixes
  the band features before the U-Net. Its output is fused into the stem.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.weight[None, :, None, None] * x + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class FiLM(nn.Module):
    """x -> x * (1 + gamma(e)) + beta(e), zero-initialised so it starts as the identity."""

    def __init__(self, emb_dim: int, channels: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, 2 * channels)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, emb):
        gamma, beta = self.proj(emb).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class NAFBlock(nn.Module):
    def __init__(self, c: int, emb_dim: int = 0, dw_expand: int = 2, ffn_expand: int = 2):
        super().__init__()
        dw, ffn = c * dw_expand, c * ffn_expand
        self.norm1, self.norm2 = LayerNorm2d(c), LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)
        self.gate = SimpleGate()
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.film1 = FiLM(emb_dim, c) if emb_dim else None
        self.film2 = FiLM(emb_dim, c) if emb_dim else None

    def forward(self, x, emb=None):
        h = self.norm1(x)
        if self.film1 is not None:
            h = self.film1(h, emb)
        h = self.gate(self.conv2(self.conv1(h)))
        h = self.conv3(h * self.sca(h))
        y = x + h * self.beta
        h = self.norm2(y)
        if self.film2 is not None:
            h = self.film2(h, emb)
        h = self.conv5(self.gate(self.conv4(h)))
        return y + h * self.gamma


class CrossBandAttention(nn.Module):
    """Attention between bands. Input (B, n_bands, H, W); output (B, n_bands * d, H, W)."""

    def __init__(self, n_bands: int = 12, d: int = 8):
        super().__init__()
        self.n_bands, self.d = n_bands, d
        self.embed = nn.Conv2d(1, d, 3, padding=1)
        self.band_embedding = nn.Parameter(torch.zeros(n_bands, d, 1, 1))
        self.qkv = nn.Conv2d(d, 3 * d, 1)
        self.proj = nn.Conv2d(d, d, 1)
        self.temperature = nn.Parameter(torch.ones(1))
        nn.init.normal_(self.band_embedding, std=0.02)

    def forward(self, x):
        b, n, h, w = x.shape
        f = self.embed(x.reshape(b * n, 1, h, w)).reshape(b, n, self.d, h, w) + self.band_embedding[None]
        f = f.reshape(b * n, self.d, h, w)
        q, k, v = self.qkv(f).chunk(3, dim=1)
        q, k, v = (t.reshape(b, n, -1) for t in (q, k, v))
        attn = torch.softmax(F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(1, 2) * self.temperature, dim=-1)
        mixed = (attn @ v).reshape(b * n, self.d, h, w)
        out = f + self.proj(mixed)
        return out.reshape(b, n * self.d, h, w)


class ReconNet(nn.Module):
    def __init__(self, in_ch=12, width=32, enc=(1, 1, 2, 4), mid=4, dec=(1, 1, 1, 1), cond_dim=0,
                 emb_dim=64, cross_band=False, cb_dim=8):
        super().__init__()
        self.in_ch, self.depth = in_ch, len(enc)
        self.cond_mlp = (nn.Sequential(nn.Linear(cond_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
                         if cond_dim else None)
        e = emb_dim if cond_dim else 0
        self.cba = CrossBandAttention(in_ch, cb_dim) if cross_band else None
        self.stem = nn.Conv2d(in_ch * cb_dim if cross_band else in_ch, width, 3, padding=1)
        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encoders.append(nn.ModuleList([NAFBlock(c, e) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.middle = nn.ModuleList([NAFBlock(c, e) for _ in range(mid)])
        self.ups, self.decoders = nn.ModuleList(), nn.ModuleList()
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decoders.append(nn.ModuleList([NAFBlock(c, e) for _ in range(n)]))
        self.head = nn.Conv2d(width, in_ch, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, cond=None):
        emb = self.cond_mlp(cond) if self.cond_mlp is not None else None
        h, w = x.shape[-2:]
        m = 2**self.depth
        xp = F.pad(x, (0, (-w) % m, 0, (-h) % m), mode="reflect") if (h % m or w % m) else x
        f = self.stem(self.cba(xp) if self.cba is not None else xp)
        skips = []
        for blocks, down in zip(self.encoders, self.downs):
            for blk in blocks:
                f = blk(f, emb)
            skips.append(f)
            f = down(f)
        for blk in self.middle:
            f = blk(f, emb)
        for up, blocks, skip in zip(self.ups, self.decoders, reversed(skips)):
            f = up(f) + skip
            for blk in blocks:
                f = blk(f, emb)
        return x + self.head(f)[..., :h, :w]


CONFIGS = {
    "tiny": dict(width=8, enc=(1, 1), mid=1, dec=(1, 1), cb_dim=4, emb_dim=16),
    "small": dict(width=24, enc=(1, 1, 2), mid=2, dec=(1, 1, 1), cb_dim=6, emb_dim=32),
    "medium": dict(width=32, enc=(1, 2, 2), mid=4, dec=(1, 1, 1), cb_dim=8, emb_dim=64),
    "base": dict(width=32, enc=(2, 2, 4), mid=8, dec=(2, 2, 2), cb_dim=8, emb_dim=64),
    "large": dict(width=32, enc=(2, 2, 4, 8), mid=12, dec=(2, 2, 2, 2), cb_dim=8, emb_dim=64),  # NAFNet-width32 shape
}
COND_DIMS = {"sigma_snr": 24, "snr": 1, "none": 0}


def build_model(size: str = "medium", cond_mode: str = "sigma_snr", cross_band: bool = True, **overrides) -> ReconNet:
    """cond_mode: 'sigma_snr' (24-d), 'snr' (scalar) or 'none' (blind); cross_band toggles the attention block.
    (cond_mode names match src.data.dataset; the blind model ignores the conditioning vector.)"""
    cfg = dict(CONFIGS[size], cond_dim=COND_DIMS[cond_mode], cross_band=cross_band)
    cfg.update(overrides)
    return ReconNet(**cfg)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
