"""Evaluation metrics + the render-aux vs latent-only reconstruction comparison (headline result).

Reproduces Kyvo's reporting: reconstruction error (L1 / SSIM / L2) over held-out SLAT rendered to multiple
views, plus task metrics (Jaccard for scene structure, text-accuracy). The key table compares a
render-aux-trained tokenizer against a latent-only ablation — render-aux should win.

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .render_loss import d_ssim


def l2(x: torch.Tensor, y: torch.Tensor) -> float:
    return F.mse_loss(x, y).item()


def l1(x: torch.Tensor, y: torch.Tensor) -> float:
    return F.l1_loss(x, y).item()


def ssim(x: torch.Tensor, y: torch.Tensor) -> float:
    return 1.0 - 2.0 * d_ssim(x.clamp(0, 1), y.clamp(0, 1)).item()  # invert D-SSIM back to SSIM


def jaccard(pred_idx: torch.Tensor, target_idx: torch.Tensor) -> float:
    """Token-set IoU — a proxy for scene-structure agreement on discrete tokens."""
    p, t = set(pred_idx.reshape(-1).tolist()), set(target_idx.reshape(-1).tolist())
    union = p | t
    return len(p & t) / len(union) if union else 1.0


def text_accuracy(pred_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
    return (pred_ids == target_ids).float().mean().item()


@torch.no_grad()
def reconstruction_report(tokenizer, slats, renderer, n_views: int = 150, device="cpu") -> dict:
    """Mean render-space L1/SSIM/L2 of tokenizer reconstructions over held-out SLAT."""
    accL1, accSSIM, accL2, n = 0.0, 0.0, 0.0, 0
    for slat in slats:
        slat = slat.to(device).unsqueeze(0)
        recon = tokenizer(slat)["recon"]
        img_r = renderer(recon, n_views)               # [1, V, 3, H, W]
        img_t = renderer(slat, n_views)
        img_r = img_r.reshape(-1, 3, *img_r.shape[-2:])
        img_t = img_t.reshape(-1, 3, *img_t.shape[-2:])
        accL1 += l1(img_r, img_t); accSSIM += ssim(img_r, img_t); accL2 += l2(img_r, img_t); n += 1
    return {"render_l1": accL1 / n, "render_ssim": accSSIM / n, "render_l2": accL2 / n, "n": n}


def compare_ablations(render_aux_report: dict, latent_only_report: dict) -> dict:
    """Headline: does render-aux beat latent-only on render-space reconstruction?"""
    return {
        "render_aux": render_aux_report,
        "latent_only": latent_only_report,
        "render_aux_wins": (render_aux_report["render_l1"] < latent_only_report["render_l1"]
                            and render_aux_report["render_ssim"] > latent_only_report["render_ssim"]),
    }
