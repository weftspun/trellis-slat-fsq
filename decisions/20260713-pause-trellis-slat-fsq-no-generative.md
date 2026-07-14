---
title: Pause trellis-slat-fsq — generative is dropped; the line pivots to zero-shot sequential recommendation in multimodal-semantic-ids
date: 2026-07-13
status: accepted — repo PAUSED and archived (read-only); generation abandoned; multimodal-semantic-ids restored as the canonical successor
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

This repo owned the **generation** channel: reproduce Kyvo (arXiv:2506.08002) with Residual FSQ —
SLAT → ~512 reconstructive tokens → unified LM. That charter is the exact inverse of this morning's
`20260713-pause-elixir-sequential-recommendation.md`, which paused the recommender because
"identification folds into the generation codebook."

Decision-maker directive (2026-07-13 evening): **"let's forget about generative… only do sequential
recommendations… zero-shot sequential recs."** The two-repo bet flips back: generation is dropped
entirely; retrieval/recommendation is the goal. `weftspun/multimodal-semantic-ids` has been
**unarchived** and is the canonical home for the successor line — multimodal semantic IDs
(text / image / mesh / audio / body-phenotype) feeding zero-shot sequential recommendation.

## Decision

**Pause `trellis-slat-fsq`.** No further feature work; the repo is archived (read-only) on GitHub and
kept as reference. No generative training, no LM port, no render aux-loss will be pursued here or in
the successor.

## What carries over (referenced from multimodal-semantic-ids, not rewritten)

- **ResidualFSQ + the Lean-proven index map** (`verify/`: bijectivity onto [0, 8192), residual-stream
  theorems, plausible-witness-dag certification; slangtorch CUDA tests 4/4) — the successor's semantic-ID
  quantizer is the same `ResidualFSQ`, so the proofs underwrite ID uniqueness there too.
- **The SLAT extraction pipeline** (`scripts/make_real_slats.py`, `make_slat_dataset.py`, the pixi
  `trellis` env): single render → pre-trained TRELLIS slat generator → SLAT — which the Kyvo paper
  (Appendix A.1) itself used for training data. In the successor this is the **mesh-modality encoder
  feed**: pooled SLAT → mesh slot of the fused vector.
- **The Slang rasterizer** (`trellis_slat_fsq/raster.slang`, OpenUSD ingest, toon-ready) for dataset
  conditioning renders.
- **Toolchain knowledge**: slangtorch/CUDA/MSVC matrix (`scripts/run_slang_tests.ps1`,
  `scripts/_slang_toolchain.py`), xformers cu126-index pin, HF xet workaround, CC0 USD stages.

## What dies with this repo

The ~512-token reconstructive budget, the render aux-loss and Slang.D 3DGS renderer plan, the
Qwen3.5-0.8B unified LM, and the phase-5/6 training loops. Retrieval IDs are deliberately lossy and
discriminative (`fsq-over-rqvae-for-semantic-ids` in the successor); reconstruction pressure is the
generative channel's concern and no longer exists.

## Consequences

- One active repo again: `multimodal-semantic-ids` (restored), targeting zero-shot sequential
  recommendation — semantic IDs from content alone mean unseen items are recommendable without
  retraining (the TIGER-style property).
- Reversible via unarchive; this MADR is the pointer explaining why the repo is quiet.

## Verification

`weftspun/trellis-slat-fsq` shows **Archived** on GitHub with all work pushed;
`weftspun/multimodal-semantic-ids` is unarchived and carries the successor charter.
