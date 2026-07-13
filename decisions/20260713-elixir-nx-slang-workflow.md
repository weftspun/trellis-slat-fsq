---
title: Move the entire reproduction workflow to Elixir/Nx — Slang kernels reach Nx via ports/adapters; Python/PyTorch implementation removed
date: 2026-07-13
status: accepted — workflow is Elixir-only (Nx/Axon/EXLA); fsq.slang is the shared kernel; torch stack deleted
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The Kyvo reproduction (`20260713-reproduce-kyvo-full-method-residual-fsq.md`) was scaffolded in
Python/PyTorch with Slang kernels via slangtorch. Decision-maker directive: the workflow must be
**entirely in Elixir**, as a **shader-Slang → Nx** pipeline. The repo already had an Elixir surface
(`lib/trellis_slat_fsq.ex`, previously a Pythonx binding), and the paused
`elixir-sequential-recommendation` MADR explicitly allowed its Elixir/Nx assets to be reused "if the
unified backbone chooses to" — it now does.

## Decision

Port the full workflow to **Elixir**: Nx (tensors/defn autodiff), **Axon** (encoder/decoder, training
loop), **Torchx** (LibTorch backend — **the backend on Windows**, where EXLA is unreliable; this reuses the
archived elixir repo's EXLA→Torchx swap), EXLA optional on Linux, **Bumblebee** (Qwen3.5-0.8B backbone).
The **Python/PyTorch implementation is removed**; `fsq.slang` stays as the language-neutral kernel (moved
to `priv/slang/`).

**Slang → Nx integration** is hexagonal (per the deployment ports/adapters directive):

- **Port**: `TrellisSlatFsq.SlangPort` — encode projected latent → token indices.
- **Adapter (deploy/parity)**: compiled Slang (`slangc` → PTX/CUDA or CPU) invoked through a NIF,
  exchanging Nx tensor binaries. Stubbed until the NIF is built.
- **Adapter (reference)**: pure `Nx.Defn` FSQ — same math, runs on EXLA CUDA or CPU host. This is also the
  differentiable path used in training (STE via `custom_grad`).
- Renderer: the Slang.D 3DGS rasterizer (`slang-gaussian-rasterization`) compiles independently of torch;
  it enters Elixir through the same NIF pattern (a `Renderer` behaviour).

Per the standing testing directive (adapted to the new substrate): **test only the Slang→Nx path**, not the
Nx-reference/CPU path.

## Settled choices

- Pythonx binding replaced: `lib/` is now the implementation, not a bridge. `pythonx` dep dropped.
- **pixi stays** as the toolchain manager (`pixi.toml`: erlang/elixir from conda-forge; tasks wrap mix);
  `pyproject.toml` is deleted with the Python package.
- LM: Bumblebee loads `Qwen/Qwen3.5-0.8B`; LoRA via Lorax. Caveat recorded: Bumblebee must support the
  Qwen3.5 architecture — if not, porting the architecture to Axon is open work.
- LPIPS has no Elixir implementation — open work (port or NIF); D-SSIM + L1 are native Nx.
- **Parquet via Explorer** (Polars, precompiled Rust NIF) for tokenized sequences / SLAT records / eval
  results. `duckdbex` was rejected: it compiles a C++ NIF and needs nmake/MSVC (failed on Windows);
  decision-maker corrected to Explorer.
- LoRA: `lorax` pins nx ~> 0.7 (conflicts with nx 0.9) — open work (vendor/port, or full fine-tune 0.8B).

## Consequences

- One language, one runtime (BEAM) for the whole method; GPU via EXLA/CUDA and Slang kernels.
- Ecosystem gaps become explicit open work: LPIPS, Bumblebee-Qwen3.5 coverage, the Slang NIF, TRELLIS.2
  encoder/decoder bindings.
- The Python scaffold is deleted, not archived — it remains in git history (`c1cd664`..) if needed.

## Verification

- `mix test` runs only Slang→Nx-path tests (tagged; excluded without the NIF + CUDA).
- The Lean `verify/` package (plausible-witness-dag) is unaffected — the index-map proof is
  language-neutral.
- Reproduction verification criteria unchanged from `20260713-reproduce-kyvo-full-method-residual-fsq.md`.
