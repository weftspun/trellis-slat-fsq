---
title: Reproduce Kyvo's full method (tokenizer + LM) with Residual FSQ, on Kyvo's released data AND our CC0 USD stages
date: 2026-07-13
status: accepted — REACTIVATES the training pipeline; full-method reproduction (residual-FSQ tokenizer + render aux-loss + torchtune LM + task suite); dual data (Kyvo HF + CC0 USD via TRELLIS.2)
tier: baseline
supersedes: 20260713-no-budget-descope-render-free-fsq-only.md
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

We now want to **reproduce Kyvo's method and results** (arXiv:2506.08002, Sahoo/Tibrewal/Gkioxari),
swapping their learned VQ for **Residual FSQ**. This reverses the no-budget descope
(`20260713-no-budget-descope-render-free-fsq-only.md`), which had obliterated the render aux-loss,
renderer, TRELLIS.2 decoder, learned encoder, and LM port. Reproduction requires all of them back.

Chosen scope (decision-maker): **full method** — the reconstructive tokenizer *and* the decoder-only LM and
task suite — evaluated on **both** data sources: **Kyvo's released pre-tokenized data** (CLEVR / ObjaWorld /
Objectron, for directly comparable numbers) **and our own CC0 USD stages** (`quaternius-stage`,
`kenney-stage`, `thebasemesh-stage`, via the TRELLIS.2 encoder).

## Decision

Reproduce Kyvo end-to-end with Residual FSQ, in dependency order:

1. **Reconstructive tokenizer** — SLAT `64³×8` → encoder → **Residual FSQ** → `~512` tokens → decoder,
   trained with the **Kyvo render aux-loss** (L1 / D-SSIM / LPIPS over 150 views). Headline result to
   reproduce: **render aux-loss beats a latent-only ablation** (Kyvo's central tokenizer finding).
   Renderer = **`google/slang-gaussian-rasterization`** (Slang.D 3D Gaussian Splatting, differentiable via
   Slang autodiff, PyTorch-integrated) fed by TRELLIS.2's SLAT→Gaussian decoder — Slang-native, no
   nvdiffrast/mesh path. (OpenUSD Hydra/Storm is not differentiable and is not used.)
2. **Unified decoder-only LM** — **`Qwen/Qwen3.5-0.8B`** (replacing Kyvo's Llama-3.2-1B) over
   `[text | image | SLAT-residual-FSQ]` with Kyvo's sequence layout (`BOS … OUTSEP … EOS`, per-modality
   boundary tokens, raw 3D order). The token embedding is extended with the SLAT residual-FSQ codes and the
   modality boundary tokens.
3. **Task suite + metrics** — 3D rendering, recognition, instruction edits, VQA, image-to-3D; metrics
   Jaccard / SSIM / L2 / text-accuracy.

## Evidence — full paper (PDF read 2026-07-13, `thirdparty/`), beyond the earlier HTML extraction

- **3D VQ-VAE recipe (Appendix D.2):** densify sparse SLAT → 64³×8 grid; 3D conv U-Net downsampling
  64³→32³→16³→8³ with channel widths (32, 128, 512, 1024); each 8³ cell → 128-dim vector → nearest of
  8192 EMA codes (τ=0.99); straight-through estimator. TRELLIS encoder/decoder stay **frozen**.
- **Loss:** `L = ‖x−x̂‖² + β·commit + λ_KL·D_KL + γ·L_render` with **β=0.25, λ_KL=1e-6, γ=0.1**, and
  `L_render = L1 + 0.2·(1−SSIM) + 0.2·LPIPS` over renders from Gaussian reconstructions (150 random
  views beats single fixed view). With **FSQ, the commit and KL terms vanish by construction** (no
  learned codebook) — our loss is `‖x−x̂‖² + γ·L_render`.
- **Optimization:** 200k steps, ~168k Objaverse-Sketchfab assets, batch 8, AdamW lr **3e-4** constant,
  no weight decay, mixed precision, adaptive grad clipping. Codebook usage is heavy-tailed but fully
  active (Fig. 23) — size is neither over- nor under-parameterized at 8192.
- **CRITICAL — Kyvo's training SLATs are synthetic (Appendix A.1):** extracting SLATs via the original
  TRELLIS pipeline (150 renders + DINOv2) was too expensive, so they render each asset **once** and
  feed it as image-conditioning to the **pre-trained TRELLIS slat generator**, using the sampled SLATs
  for training; evaluation uses true-pipeline SLATs and "performance transfers well". **Our
  `scripts/make_real_slats.py` / `make_slat_dataset.py` implement exactly this recipe** (single render
  → TRELLIS-image-large flow model → SLAT), so our dataset construction is method-faithful, not a
  shortcut.
- **Serialization (Appendix A.2/A.4):** scenes as marker-structured strings — `[SCENE-START]`,
  `[OBJECT-START]`, `[SIZE]`, `[COLOR]`, `[SHAPE]`, `[LOCATION]`, `[POSE]`, `[OUTPUT-SEP]`,
  `[IMAGE-START]`… registered as special tokens; **512 shape tokens follow `[SHAPE]`**; every location
  coordinate is a distinct numerical token (hybrid learned + sine-cosine embeddings); **bidirectional
  attention within shape-token spans** when training the LLM. Our `lm.py` boundary-token scheme is the
  same shape; adopting the full marker vocabulary + numeric tokens is open work for scene-level tasks.
- **Dataset scale (Appendix A.1):** CLEVR 120k scenes (rendering/recognition), 100k instruction pairs,
  20k QA; ObjaWorld 100k scenes ×2 setups; Objectron/ARKitScenes per Omni3D splits.

## Data

- **Kyvo released**: pre-tokenized HF sequences + SLAT for CLEVR / ObjaWorld / Objectron. We re-quantize
  SLAT with Residual FSQ (Kyvo's 3D VQ-VAE codebook is *not* released, only image VQGAN codebooks), so our
  tokens differ from theirs by construction; comparison is like-for-like on reconstruction/metrics.
- **CC0 USD stages**: `.usda` assets with geometry + materials → TRELLIS.2 encoder → SLAT → same tokenizer.

## Settled choices (carried forward + new)

- **Residual FSQ**, per-stage level set `[8, 8, 8, 16]` = exactly 8192 codes (per-code parity with Kyvo's
  8192-entry VQ); coarse prefix = retrieval ID (`20260713-port-kyvo-residual-fsq.md`).
- **Slang + PyTorch** substrate for the FSQ kernel; training in PyTorch. `torch` is required (now for
  training, not just inference).
- **LM backbone = `Qwen/Qwen3.5-0.8B`** (replaces Llama-3.2-1B; supersedes the interim Qwen3-0.6B pick).
  Rationale: a faithful, *minimal* reproduction needs a backbone near the paper's 1B. The **Qwen3.6 family
  was rejected** — no small model (27B / 35B-A3B); **Qwen3.5 Small has 0.8B**, right at scale. Gemma 4 was
  reranked and rejected (smallest E2B ≈ 2B effective; native multimodality is unused since images enter as
  VQGAN tokens).
- **Loading Qwen3.5-0.8B (Elixir/Bumblebee) — operational findings (2026-07-13):**
  - Neither Bumblebee 0.7.0 nor GitHub main maps `Qwen3_5ForConditionalGeneration`; the config loads
    under a **forced `module: Bumblebee.Text.Qwen3`** mapping (`TrellisSlatFsq.LM.load/2`). Bumblebee is
    pinned to **GitHub default branch** per directive.
  - HF serves the repo's weights via **xet-only storage**: plain HTTP `resolve` returns **403
    AccessDenied** (Bumblebee can't download it directly). Workaround: fetch once with
    `uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.5-0.8B`, then load via
    `source: {:local, snapshot}`.
  - Config: vocab 151,936; hidden 2560; 36 blocks; GQA 32/8.
  - `LM.extend_embeddings/2` rebuilds the graph from a re-configured spec and pads every param with a
    vocab-sized axis (random init, σ=0.02) — naming-robust.
- **Verification** includes the `plausible-witness-dag` Lean proof of the FSQ index map (planned) plus
  runtime metrics; render aux-loss tests on the Slang→PyTorch path.

## Consequences

- Reactivates a **heavy training-time dependency set**: differentiable renderer
  (`slang-gaussian-rasterization`, Slang.D 3DGS), TRELLIS.2 decoder + encoder weights, HF `transformers` +
  `peft` (Qwen3.5-0.8B LM, LoRA), LPIPS, USD tooling. Kept behind a `train` extra so render-free inference
  stays light.
- **Deployment** needs Slang **ports + adapters** (hexagonal): a kernel port with slangtorch/CUDA
  (training), torch-CPU (fallback), and compiled-Slang serving adapters — see `slang_ports.py`.
- Reproduced **results are not available from this scaffold alone** — they require GPU, external weights,
  and the datasets. This repo provides the method harness; results follow when those are supplied.
- The no-budget descope is superseded but retained on file; its render-free inference path (`encode`) is
  the inference half of this same tokenizer.

## Verification

- Tokenizer: reconstruct held-out SLAT through the Residual FSQ tokenizer; report multi-view render error
  (L1 / SSIM / LPIPS) for **render-aux vs latent-only** — reproducing Kyvo's finding that render-aux wins.
- Coarse prefix gives usable retrieval separation while the full stack reconstructs.
- LM: train/eval on both data sources with Kyvo's metric suite; inference path performs no rendering.
- FSQ index map certified via `plausible-witness-dag`.
