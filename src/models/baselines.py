"""Baseline restoration networks for 12-band patches, all with forward(x, cond).

- DnCNN: blind residual CNN (conv-BN-ReLU stack). Ignores cond.
- FFDNetLike: non-blind. The conditioning vector is broadcast to per-pixel noise-level maps and
  concatenated with the input, then a plain conv stack runs on a 2x pixel-unshuffled image (as FFDNet).
- SwinIRLite: blind Swin-transformer restoration (window attention with shifted windows, residual Swin
  blocks, global residual), a compact version of SwinIR. Ignores cond.

Every network predicts a residual added to the input, and its last layer starts at zero, so an untrained
network is exactly the identity (same convention as ReconNet).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DnCNN(nn.Module):
    def __init__(self, in_ch=12, width=64, depth=17):
        super().__init__()
        layers = [nn.Conv2d(in_ch, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        last = nn.Conv2d(width, in_ch, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.body = nn.Sequential(*layers, last)

    def forward(self, x, cond=None):
        return x + self.body(x)


class FFDNetLike(nn.Module):
    def __init__(self, in_ch=12, cond_dim=24, width=96, depth=12):
        super().__init__()
        self.in_ch, self.cond_dim = in_ch, cond_dim
        self.down, self.up = nn.PixelUnshuffle(2), nn.PixelShuffle(2)
        layers = [nn.Conv2d((in_ch + cond_dim) * 4, width, 3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(width, width, 3, padding=1, bias=False), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        last = nn.Conv2d(width, in_ch * 4, 3, padding=1)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.body = nn.Sequential(*layers, last)

    def forward(self, x, cond):
        b, c, h, w = x.shape
        xp = F.pad(x, (0, w % 2, 0, h % 2), mode="reflect") if (h % 2 or w % 2) else x
        maps = cond[:, :, None, None].expand(b, self.cond_dim, xp.shape[2], xp.shape[3])
        y = self.up(self.body(self.down(torch.cat([xp, maps], dim=1))))
        return x + y[..., :h, :w]


# ---- SwinIR-lite -------------------------------------------------------------------------------

def window_partition(x, ws):
    b, h, w, c = x.shape
    x = x.view(b, h // ws, ws, w // ws, ws, c)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, c)


def window_reverse(windows, ws, h, w):
    b = windows.shape[0] // ((h // ws) * (w // ws))
    x = windows.view(b, h // ws, w // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, -1)


class WindowAttention(nn.Module):
    def __init__(self, dim, ws, heads):
        super().__init__()
        self.ws, self.heads, self.scale = ws, heads, (dim // heads) ** -0.5
        self.qkv, self.proj = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.bias_table = nn.Parameter(torch.zeros((2 * ws - 1) ** 2, heads))
        nn.init.trunc_normal_(self.bias_table, std=0.02)
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing="ij")).flatten(1)
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0) + (ws - 1)
        self.register_buffer("bias_index", rel[..., 0] * (2 * ws - 1) + rel[..., 1], persistent=False)

    def forward(self, x, mask=None):
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn + self.bias_table[self.bias_index.view(-1)].view(n, n, -1).permute(2, 0, 1).unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.heads, n, n)
        return self.proj((attn.softmax(dim=-1) @ v).transpose(1, 2).reshape(b, n, c))


class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws, shift, mlp_ratio=2.0):
        super().__init__()
        self.ws, self.shift = ws, shift
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, ws, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim))
        self._masks: dict = {}

    def _mask(self, h, w, device):
        key = (h, w, str(device))
        if key not in self._masks:
            img = torch.zeros(1, h, w, 1)
            cnt = 0
            for hs in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                for ws_ in (slice(0, -self.ws), slice(-self.ws, -self.shift), slice(-self.shift, None)):
                    img[:, hs, ws_, :] = cnt
                    cnt += 1
            mw = window_partition(img, self.ws).squeeze(-1)
            diff = mw.unsqueeze(1) - mw.unsqueeze(2)
            self._masks[key] = diff.masked_fill(diff != 0, -100.0).masked_fill(diff == 0, 0.0).to(device)
        return self._masks[key]

    def forward(self, x, h, w):
        b, n, c = x.shape
        y = self.norm1(x).view(b, h, w, c)
        mask = None
        if self.shift:
            y = torch.roll(y, (-self.shift, -self.shift), (1, 2))
            mask = self._mask(h, w, x.device)
        y = window_reverse(self.attn(window_partition(y, self.ws), mask), self.ws, h, w)
        if self.shift:
            y = torch.roll(y, (self.shift, self.shift), (1, 2))
        x = x + y.reshape(b, n, c)
        return x + self.mlp(self.norm2(x))


class RSTB(nn.Module):
    def __init__(self, dim, depth, heads, ws):
        super().__init__()
        self.blocks = nn.ModuleList([SwinBlock(dim, heads, ws, 0 if i % 2 == 0 else ws // 2) for i in range(depth)])
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x, h, w):
        b, n, c = x.shape
        y = x
        for blk in self.blocks:
            y = blk(y, h, w)
        y = self.conv(y.transpose(1, 2).reshape(b, c, h, w)).flatten(2).transpose(1, 2)
        return x + y


class SwinIRLite(nn.Module):
    def __init__(self, in_ch=12, dim=60, depths=(2, 2, 2, 2), heads=6, ws=8):
        super().__init__()
        self.ws = ws
        self.conv_first = nn.Conv2d(in_ch, dim, 3, padding=1)
        self.layers = nn.ModuleList([RSTB(dim, d, heads, ws) for d in depths])
        self.norm = nn.LayerNorm(dim)
        self.conv_after = nn.Conv2d(dim, dim, 3, padding=1)
        self.conv_last = nn.Conv2d(dim, in_ch, 3, padding=1)
        nn.init.zeros_(self.conv_last.weight)
        nn.init.zeros_(self.conv_last.bias)

    def forward(self, x, cond=None):
        h0, w0 = x.shape[-2:]
        ph, pw = (-h0) % self.ws, (-w0) % self.ws
        xp = F.pad(x, (0, pw, 0, ph), mode="reflect") if (ph or pw) else x
        h, w = xp.shape[-2:]
        f0 = self.conv_first(xp)
        t = f0.flatten(2).transpose(1, 2)
        for layer in self.layers:
            t = layer(t, h, w)
        t = self.norm(t).transpose(1, 2).reshape(f0.shape)
        f = self.conv_after(t) + f0
        return x + self.conv_last(f)[..., :h0, :w0]
