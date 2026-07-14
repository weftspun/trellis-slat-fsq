"""Tokenizer training loop with the render-aux vs latent-only ABLATION (the headline reproduction).

Kyvo's central tokenizer finding: latent reconstruction alone is insufficient; adding the multi-view render
aux-loss improves reconstruction. We reproduce it by training two configs and comparing (see eval.py):

    ablation="latent_only"  -> loss = L_latent
    ablation="render_aux"   -> loss = L_latent + w_render * L_render (150 views, L1+D-SSIM+LPIPS)

Rendering is TRAINING-TIME ONLY. Training runs in PyTorch; the FSQ kernel and the 3DGS renderer are Slang
adapters (slangtorch). Requires the `train` extra + GPU for real runs; the synthetic path exercises the
graph on CPU (no renderer -> latent_only only).

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .render_loss import MultiViewRenderLoss
from .tokenizer import SlatFsqReconstructiveTokenizer


@dataclass
class TokenizerTrainConfig:
    ablation: str = "render_aux"      # "render_aux" | "latent_only"
    levels: tuple = (8, 8, 8, 16)
    num_quantizers: int = 8
    id_prefix: int = 1
    w_render: float = 1.0
    lr: float = 2e-4
    steps: int = 1000
    n_views: int = 150


def latent_recon_loss(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(recon, target)


def train_tokenizer(batches, cfg: TokenizerTrainConfig, renderer=None, lpips_fn=None, device="cpu",
                    on_step=None):
    """Train the reconstructive tokenizer. `batches` yields tensors [B, 8, 64, 64, 64].

    `renderer` (a Slang 3DGS adapter) is required for ablation="render_aux"; latent_only ignores it.
    `on_step(total_loss: float)` is called after each optimizer step, if given.
    Returns the trained tokenizer and the last step's loss breakdown.
    """
    if cfg.ablation not in ("render_aux", "latent_only"):
        raise ValueError(cfg.ablation)
    if cfg.ablation == "render_aux" and renderer is None:
        raise ValueError("ablation='render_aux' needs a renderer (Slang 3DGS adapter)")

    tok = SlatFsqReconstructiveTokenizer(cfg.levels, cfg.num_quantizers, cfg.id_prefix).to(device)
    opt = torch.optim.AdamW(tok.parameters(), lr=cfg.lr)
    render_loss = (MultiViewRenderLoss(renderer, lpips_fn, n_views=cfg.n_views).to(device)
                   if cfg.ablation == "render_aux" else None)

    last = {}
    step = 0
    for slat in batches:
        if step >= cfg.steps:
            break
        slat = slat.to(device)
        out = tok(slat)
        loss = latent_recon_loss(out["recon"], slat)
        breakdown = {"latent": loss.detach()}
        if render_loss is not None:
            r = render_loss(out["recon"], slat)
            loss = loss + cfg.w_render * r["render_total"]
            breakdown.update({k: r[k] for k in ("l1", "d_ssim", "lpips")})
        opt.zero_grad()
        loss.backward()
        opt.step()
        breakdown["total"] = loss.detach()
        last = breakdown
        if on_step is not None:
            on_step(float(loss.detach()))
        step += 1
    return tok, last
