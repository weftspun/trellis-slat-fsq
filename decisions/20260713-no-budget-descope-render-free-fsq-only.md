---
title: No-budget descope — obliterate all training/compute features; keep only render-free FSQ tokenization
date: 2026-07-13
status: accepted — ZERO-COMPUTE floor; render aux-loss + renderer + TRELLIS.2 decoder + corpus + LM port all OBLITERATED; reversible when budget exists
tier: baseline
supersedes: 20260713-generation-slat-fsq-render-auxloss.md, 20260713-port-kyvo-residual-fsq.md
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

There is **no compute budget** — no GPU training, no rendering, no LM fine-tuning. The prior accepted
decisions (`20260713-generation-slat-fsq-render-auxloss.md` adopting a Kyvo-style render aux-loss;
`20260713-port-kyvo-residual-fsq.md` porting Kyvo's LM on residual-FSQ tokens) both assume a training
budget we do not have. This decision descopes the repo to what costs **zero compute**.

The enabling fact: **FSQ (and Residual FSQ) have no learned codebook.** They are a *fixed* quantization
grid — a SLAT latent can be quantized to tokens deterministically, on CPU, with **no training**. That is
the only part of the pipeline that survives a zero-budget constraint.

## Decision

**Obliterate every feature that requires training or compute. Keep only render-free FSQ tokenization of
SLAT.**

Obliterated (all training/compute-bound):

- Kyvo-style **render aux-loss** (L1 / D-SSIM / LPIPS over 150 rendered views).
- **Differentiable multi-view renderer** (nvdiffrast / gsplat) and **TRELLIS.2 decoder** wiring.
- **Corpus assembly + 150-view sampling** recipe.
- **Kyvo LM port** (torchtune decoder-only Llama fine-tune + eval/metrics).
- **Learned SLAT encoder/decoder training.**
- The `render` optional dependency extra in `pyproject.toml` (was empty TODO anyway).

Kept (zero compute):

- **Render-free FSQ / Residual FSQ tokenization**: quantize SLAT → tokens with a fixed grid, CPU, no
  training. This is the surviving core, already scaffolded in `lib/trellis_slat_fsq.ex` (`fsq_quantize/2`).
- The `[8, 8, 8, 16]` → exactly-8192 level set stands as the parity choice from
  `20260713-port-kyvo-residual-fsq.md`; it needs no training to use.

## Accepted consequence (the cost of zero budget)

Kyvo's central finding is that **latent-only reconstruction is insufficient** — which is precisely why the
render aux-loss existed. Without it, tokens come from an **untrained fixed grid over raw SLAT**:
reconstruction quality is **unvalidated and likely degraded**. We accept this knowingly. With no budget
there is no alternative — nothing can be trained — so render-free FSQ tokenization is the honest floor, not
a claim of quality parity with Kyvo.

## Scope boundary

This repo still owns **generation** and remains the canonical home for the SLAT→FSQ line. It no longer
claims a render-supervised, reconstruction-validated tokenizer. The retrieval facts stay canonical in
`slat-semantic-ids`; the paused `elixir-sequential-recommendation` is unaffected.

## Reversibility

This is a **descope, not a deletion of intent.** The obliterated decisions remain on file, marked
superseded. If a compute budget appears, un-supersede
`20260713-generation-slat-fsq-render-auxloss.md` and `20260713-port-kyvo-residual-fsq.md` and resume the
render aux-loss + LM port from there.

## Verification

- `TrellisSlatFsq.init()` then `fsq_quantize/2` produces tokens from a SLAT latent on CPU with no training
  step and no rendering.
- `pyproject.toml` has no `render` extra and no renderer/decoder dependency.
- No training loop, renderer, corpus, or LM code exists in the repo.
