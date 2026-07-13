"""Kyvo-style multi-view render auxiliary loss for training the reconstructive tokenizer.

Kyvo's central tokenizer finding: latent-space reconstruction alone is insufficient; a pixel-space loss
(L1 + D-SSIM + LPIPS) over ~150 rendered views is needed. Reproducing that finding (render-aux vs
latent-only) is the headline result. Rendering is TRAINING-TIME ONLY; inference encodes SLAT -> tokens with
no rendering.

The differentiable renderer and the TRELLIS.2 decoder (SLAT -> renderable asset) are external and injected
via the `Renderer` protocol; the default raises so the dependency is explicit.

Recommended off-the-shelf renderer (Slang-native): TRELLIS.2 decodes SLAT -> 3D Gaussians, then the Slang.D
3D Gaussian Splatting rasterizer `google/slang-gaussian-rasterization` renders differentiably (Slang has
first-class autodiff via [Differentiable]; PyTorch-integrated, compiles to CUDA/Vulkan/D3D/OptiX). No
mesh/nvdiffrast path is required since TRELLIS already emits Gaussian splats. OpenUSD's Hydra/Storm is NOT
differentiable and is not usable here. Training stays in PyTorch; the rasterizer is the Slang adapter.

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F

N_VIEWS_DEFAULT = 150


class Renderer(Protocol):
    """Differentiable SLAT -> multi-view images. Implemented by an nvdiffrast/gsplat + TRELLIS.2 decoder."""

    def __call__(self, slat: torch.Tensor, n_views: int) -> torch.Tensor:
        """slat [B, 8, 64, 64, 64] -> images [B, n_views, 3, H, W], differentiable w.r.t. slat."""
        ...


def _gaussian_window(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    return (g.t() @ g).unsqueeze(0).unsqueeze(0)  # [1,1,ws,ws]


def d_ssim(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """Structural dissimilarity (1 - SSIM)/2, averaged over channels. x, y: [N, C, H, W] in [0, 1]."""
    n, c, h, w = x.shape
    win = _gaussian_window(window_size, sigma, x.device, x.dtype).expand(c, 1, window_size, window_size)
    pad = window_size // 2
    mu_x = F.conv2d(x, win, padding=pad, groups=c)
    mu_y = F.conv2d(y, win, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, win, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, win, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, win, padding=pad, groups=c) - mu_xy
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    ssim = ((2 * mu_xy + c1) * (2 * sig_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sig_x + sig_y + c2))
    return ((1 - ssim.mean()) / 2)


class MultiViewRenderLoss(torch.nn.Module):
    """L1 + D-SSIM + LPIPS over rendered views of decoded vs ground-truth SLAT.

    `renderer` renders both the reconstructed and the target SLAT to `n_views`. `lpips_fn` is an injected
    LPIPS module (e.g. `lpips.LPIPS(net='vgg')`); if None, the LPIPS term is skipped (with a warning cost).
    """

    def __init__(self, renderer: Renderer, lpips_fn=None, n_views: int = N_VIEWS_DEFAULT,
                 w_l1: float = 1.0, w_ssim: float = 1.0, w_lpips: float = 1.0):
        super().__init__()
        self.renderer = renderer
        self.lpips_fn = lpips_fn
        self.n_views = n_views
        self.w_l1, self.w_ssim, self.w_lpips = w_l1, w_ssim, w_lpips

    def forward(self, recon_slat: torch.Tensor, target_slat: torch.Tensor) -> dict:
        img_recon = self.renderer(recon_slat, self.n_views)   # [B, V, 3, H, W]
        with torch.no_grad():
            img_target = self.renderer(target_slat, self.n_views)
        b, v = img_recon.shape[:2]
        r = img_recon.reshape(b * v, *img_recon.shape[2:])
        t = img_target.reshape(b * v, *img_target.shape[2:])
        l1 = F.l1_loss(r, t)
        ssim = d_ssim(r.clamp(0, 1), t.clamp(0, 1))
        lpips = self.lpips_fn(r, t).mean() if self.lpips_fn is not None else torch.zeros((), device=r.device)
        total = self.w_l1 * l1 + self.w_ssim * ssim + self.w_lpips * lpips
        return {"render_total": total, "l1": l1.detach(), "d_ssim": ssim.detach(), "lpips": lpips.detach()}
