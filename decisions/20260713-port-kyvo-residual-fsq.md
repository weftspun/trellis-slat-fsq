---
title: Port Kyvo (arXiv:2506.08002) to this repo with Residual FSQ — build the missing 3D tokenizer here, port only the LM layer
date: 2026-07-13
status: REACTIVATED (accepted) under 20260713-reproduce-kyvo-full-method-residual-fsq.md — Residual FSQ ADOPTED as the SLAT quantizer; tokenizer built here (Kyvo ships none); LM layer ported on top; coarse residual prefix IS the retrieval ID; per-stage [8,8,8,16]=8192. [Was briefly superseded by the no-budget descope, now itself superseded.]
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

We want to reproduce Kyvo's unified decoder-only model over `[text | image | 3D]` (arXiv:2506.08002v2,
*"Aligning Text, Images, and 3D Structure Token-by-Token"*), but with **Residual FSQ** as the 3D
quantizer instead of Kyvo's learned VQ codebook. Before committing, we needed to know what "porting Kyvo"
actually entails — how much is reusable code versus how much must be built — and where Residual FSQ slots
in.

Inspection of `github.com/AadSah/kyvo` establishes the decisive structural fact: **Kyvo's repo is a
transformer-LM fine-tuning project** (torchtune + Llama-3.2-1B) that trains and evaluates on
**pre-tokenized data**. It ships the *image* VQGAN codebooks (via the `taming-transformers` submodule) and
pre-computed 3D tokens on Hugging Face, but **the 3D VQ-VAE / tokenizer training code is not in the repo**.
The quantizer — the exact component we want to make Residual FSQ — lives upstream of everything Kyvo
published. So "swap VQ → Residual FSQ" is not a line change inside Kyvo; it is building the missing 3D
tokenizer, which is already this repo's open work (see
`20260713-generation-slat-fsq-render-auxloss.md`).

## Evidence — Kyvo paper (arXiv:2506.08002v2), figures confirmed from the full text

- **3D VQ-VAE:** TRELLIS SLAT `64³×8` (sparse voxel features) → quantized `8³×128` → **512 tokens/object
  (~40× reduction)**.
- **Token efficiency:** **512 vs 2040 tokens** against SAR3D — **4× fewer** at comparable-or-better
  reconstruction.
- **Render aux-loss:** latent-space reconstruction alone was **insufficient**; they add pixel-space
  **ℒ₁ / D-SSIM / LPIPS** on decoded renders sampled from **150 viewpoints**. Multi-view substantially
  beat single fixed-view. **Rendering is training-time only**; inference encodes SLAT → tokens.
- **Quantizer:** Kyvo uses **VQ with an 8,192-entry learned codebook**. This is the single component we
  deviate on.
- **LM layer:** decoder-only Llama-3.2-1B via torchtune; released tasks — 3D rendering from text/images,
  3D recognition, instruction-following scene edits, VQA, image-to-3D. Training/eval/metrics scripts
  (Jaccard / SSIM / L2 / text-accuracy) are the genuinely portable part.

## Decision

Adopt **Residual FSQ** (`vector_quantize_pytorch.ResidualFSQ`) as the SLAT quantizer for a Kyvo-style
unified model, and execute the port in dependency order:

1. **Build the SLAT → Residual FSQ tokenizer here first** (encoder/decoder + Residual FSQ + Kyvo-style
   multi-view render aux-loss). This is the foundation Kyvo never released and the thing that produces
   tokens.
2. **Port only Kyvo's LM layer** (torchtune decoder-only finetune/eval/metrics) on top of *our* tokens,
   over a unified sequence `[text | image | SLAT-residual-FSQ]`.

Rendering stays **training-time only**; inference is **encode-only, render-free**. The `~512`-token budget
remains the reconstructive budget owned by this repo.

## Settled choices

- **Residual FSQ over VQ (and over plain FSQ).** Keeps the standing FSQ-over-VQ property (fixed grid, no
  learned codebook, no collapse — `fsq-over-rqvae-for-semantic-ids`) while adding coarse-to-fine residual
  structure. This is the one deliberate deviation from Kyvo's 8192-entry VQ.
- **Per-code level set = `[8, 8, 8, 16]` → exactly 8192 codes**, giving per-code parity with Kyvo's
  8192-entry VQ (one FSQ code = 13 bits = one Kyvo token). This exact match is possible because 8192 = 2¹³
  is a pure power of two, so a level set of powers-of-2 whose exponents sum to 13 hits it dead-on. Among the
  exact matches (`[8,8,8,16]`, `[8,8,8,8,2]`, `[8,8,8,4,4]`, `[16,16,32]`) we pick **`[8,8,8,16]`** because
  it is the only one whose every level stays **≥ 8** — the regime where FSQ reconstructs best (the paper's
  ≥5 guidance penalizes levels of 2–4). This is the parity/refinement level set, NOT the coarse-ID prefix.
- **Coarse-ID prefix uses a SMALLER level set than 8192.** The retrieval/identification prefix is meant to
  be compact, so its coarse residual stage(s) use a smaller product (a few hundred to a few thousand codes,
  e.g. `[8,6,5]`≈240 or `[8,8,6,5]`≈1920), not the full 8192. Matching 8192 exactly is for per-code parity
  with Kyvo, not for the ~3-code ID.
- **512-token budget is set independently** of per-code capacity, as `spatial positions × residual depth`
  (Kyvo spends 512 as one VQ code per 8³ position; we spend ours as positions × R FSQ codes for the same
  budget). Per-code capacity (`∏levels`) and total token count are two separate knobs.
- **Coarse residual prefix IS the retrieval / identification ID.** Residual quantization is coarse-to-fine
  by construction, so the **first ~3 codes** serve as the retrieval ID and the **full residual stack ≈ 512
  tokens** serves generation — the unified dual-use codebook described in
  `20260713-pause-elixir-sequential-recommendation.md`. The full 512-token stream is never placed in a
  retrieval context window.
- **Build tokenizer before LM.** Porting the LM first leaves nothing real to feed it; the dependency runs
  one way only.
- **Kyvo assets are not directly reusable as inputs.** Their pre-tokenized HF data and image VQGAN
  codebooks are tied to CLEVR/ObjaWorld/Objectron and their VQ tokenizer, not TRELLIS.2 SLAT + Residual
  FSQ. We re-tokenize our own corpus. What ports is *code shape* (sequence assembly, training loop,
  metrics), not their tokens.

## Scope boundary

This repo (`trellis-slat-fsq`) owns **generation** and is the canonical home for the SLAT→FSQ line and the
render aux-loss, so this port decision is recorded here. The **retrieval** facts remain canonical in
`slat-semantic-ids`; this MADR only asserts that the *coarse prefix* of the generation codebook can serve
as the ID — it does not restate retrieval facts. The paused Elixir recommender
(`elixir-sequential-recommendation`) is unaffected and stays archived.

## Open work (not implemented in this scaffold)

- SLAT → Residual FSQ encoder/decoder module (replacing/extending the current `fsq_quantize/2` stub, which
  only wires the plain FSQ quantizer).
- Differentiable multi-view renderer (nvdiffrast / gsplat) + TRELLIS.2 decoder wiring for the pixel
  aux-loss (the empty `render` extra in `pyproject.toml`).
- Corpus + 150-view sampling recipe for the render aux-loss.
- Ported torchtune LM layer over `[text | image | SLAT-residual-FSQ]`, plus the tokenization pass that
  emits our tokens in Kyvo's expected sequence format.

## Consequences

- The "port" is honestly two efforts: a **from-scratch tokenizer** (most of the work; our open work) and a
  **genuine but smaller LM-layer port** on top.
- Residual FSQ gives generation + identification from one codebook, collapsing the earlier two-stack design
  further.
- Heavy training-time dependency remains the differentiable renderer + TRELLIS.2 decoder; inference stays
  render-free.
- FOSS: TRELLIS.2 (MIT), vector-quantize-pytorch ResidualFSQ (MIT), Kyvo (Apache-2.0), torchtune (BSD-3).

## Verification

- Reconstruct held-out SLAT through the Residual FSQ tokenizer; measure multi-view render error
  (ℒ₁ / SSIM / LPIPS) vs a latent-only-trained ablation, confirming the render aux-loss helps.
- Confirm the coarse ~3-code prefix alone gives usable retrieval separation while the full stack
  reconstructs.
- Confirm the inference path performs no rendering.
- Confirm the ported LM trains/evaluates on our emitted tokens with Kyvo's metric suite (Jaccard / SSIM /
  L2 / text-accuracy).
