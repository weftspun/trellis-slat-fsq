---
title: Generation channel = TRELLIS.2 SLAT → ~512 reconstructive FSQ tokens, trained with a Kyvo-style render aux-loss
date: 2026-07-13
status: accepted — render aux-loss ADOPTED (do what Kyvo does); training-time only; inference render-free; FSQ downstream
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

We generate avatars, worlds, and props from TRELLIS.2 SLAT. Generation needs **reconstructive** tokens —
the token stream must decode back to faithful geometry + PBR material, not merely discriminate assets.
This is the opposite pressure from the retrieval system (separate repo `slat-semantic-ids`), whose
compact 3-code semantic ID is deliberately lossy and render-free. Conflating the two led to an earlier
contradiction (betting a render-free, discriminative-only tokenizer would also serve generation). It does
not: this repo owns the reconstructive, render-supervised path.

## Evidence — Kyvo (arXiv:2506.08002v2), the direct precedent

- 3D VQ-VAE on TRELLIS SLAT: 64³×8 (L≈20k) → 8³×128 → quantize → ~512 tokens/object (~40×; beats SAR3D
  at 4× fewer tokens: 512 vs 2040).
- **Latent-space-only reconstruction was insufficient.** Kyvo added a pixel-space aux loss
  (L1 / D-SSIM / LPIPS) over 150 rendered viewpoints to train the tokenizer. Rendering is **training-time
  only**; inference just encodes SLAT → tokens.
- Backbone / data are Apache-2.0; HF `aadarsh99/kyvo-datasets-and-codebooks` ships the IMAGE VQGAN
  codebooks, NOT the 3D VQ-VAE codebook — the 3D tokenizer must be trained here.

## Decision

Generation channel = **encode TRELLIS.2 SLAT → ~512 reconstructive tokens/object**, training the
tokenizer with a **Kyvo-style multi-view render aux-loss** (do what Kyvo does). Rendering is used in
**training only**; inference encodes SLAT → tokens with **no rendering**. Downstream quantizer = **FSQ**,
not VQ (standing FSQ-over-VQ decision: fixed grid, no learned codebook, no collapse) — this is the one
deviation from Kyvo's VQ-VAE.

## Settled choices

- **Render aux-loss is ADOPTED**, not deferred. Because the assets are generated, the tokens must
  reconstruct; Kyvo's finding that latent-only reconstruction is insufficient applies directly. This is
  the accepted baseline, not a last resort.
- **FSQ over VQ** for the discrete bottleneck (prior decision: fsq-over-rqvae-for-semantic-ids).
- **~512 tokens/object** is the reconstructive budget. It lives here and is NEVER placed in the
  retrieval system's interaction sequence — the two token budgets do not share a context window.

## Scope boundary

This repo is the **generation** system only, and the **canonical** home for the render aux-loss and the
Kyvo evidence behind it. The **retrieval** system — pooled SLAT → compact 3-code FSQ semantic ID,
render-free — lives in the separate `slat-semantic-ids` repo (canonical for retrieval facts) and is a
distinct decision surface. If retrieval ever needs stronger inputs, it may *pool this repo's
render-trained tokens*; it does not add rendering of its own. Neither repo restates the other's facts.

## Open work (not implemented in this scaffold)

- Differentiable multi-view renderer (nvdiffrast / gsplat) + TRELLIS.2 decoder wiring for the pixel
  aux-loss.
- The SLAT → FSQ encoder/decoder and training loop (L1 / D-SSIM / LPIPS over rendered views + latent
  reconstruction).
- Corpus + view-sampling recipe for the render aux-loss.

## Consequences

- Generation gets reconstructive, render-supervised tokens; inference stays render-free (encode only).
- Heavy training-time dependency: a differentiable renderer + TRELLIS.2 decoder. Marked as the `render`
  optional extra in `pyproject.toml` so the core package installs without it.
- FOSS: TRELLIS.2 (MIT), vector-quantize-pytorch FSQ (MIT), Kyvo (Apache-2.0).

## Verification

Once implemented: reconstruct held-out SLAT through the FSQ tokenizer and measure multi-view render
error (L1 / SSIM / LPIPS) vs a latent-only-trained ablation, confirming the render aux-loss improves
reconstruction. Confirm inference path performs no rendering.
