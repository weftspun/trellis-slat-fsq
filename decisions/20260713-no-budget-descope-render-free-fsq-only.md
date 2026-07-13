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

Kept (no *training*):

- **Render-free FSQ tokenization** (`trellis_slat_fsq/` Python package): quantize an 8³×128 SLAT grid →
  512 tokens with a fixed grid, no training. The surviving core.
- The `[8, 8, 8, 16]` → exactly-8192 level set stands as the parity choice from
  `20260713-port-kyvo-residual-fsq.md`; it needs no training to use.

## Implementation substrate — Slang + PyTorch (amendment)

The quantize+index math runs as a **Slang compute kernel** (`trellis_slat_fsq/fsq.slang`) invoked from
**PyTorch** via `slangtorch` on CUDA, with a numerically identical **pure-torch reference** for CPU /
no-Slang environments. This **reinstates `torch`** (and adds the optional `slangtorch`/CUDA toolchain),
reversing this MADR's earlier "no torch" minimization — but **only for the inference kernel; there is still
no training**, so the no-budget core holds. `slangtorch`/CUDA is an *optional* extra (`[slang]`); the torch
CPU path keeps the package usable without a GPU. (The earlier `lib/trellis_slat_fsq.ex` Elixir/Pythonx
binding remains a separate interface and is unaffected.)

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

- **Test only the Slang → PyTorch (CUDA) path** (`tests/test_tokenizer.py`): drive the Slang kernel through
  `encode()` on a CUDA tensor and assert shape (512 tokens), index range `[0, 8192)`, and saturated index
  extremes (0 and 8191). The pure-torch CPU reference is **not** tested. Suite skips without
  `slangtorch` + CUDA.
- **Proof step — `plausible-witness-dag`** (`github.com/fire/plausible-witness-dag`, a Lean/Lake
  iterative-deepening witness-search library) is the intended formal-verification harness: certify the FSQ
  index map (fixed grid → `[0, 8192)`, bijective on the code lattice) as a witness-DAG proof. *Not yet
  implemented* — it is Lean-side and is its own effort; recorded here as the committed verification method.
- **Real fixture — CC0 OpenUSD asset**: use a `.usda` file carrying **both geometry and materials** from
  the taskspun CC0 stages (`quaternius-stage`, `kenney-stage`, `thebasemesh-stage`). Note the gap: turning
  USD → SLAT needs the TRELLIS.2 encoder, which is obliterated here — so the USD fixture exercises the
  tokenizer only once a SLAT latent exists for it.
- `pyproject.toml` has no `render` extra and no renderer/decoder dependency; no training loop, renderer,
  corpus, or LM code exists in the repo.
