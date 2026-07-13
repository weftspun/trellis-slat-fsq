---
title: Drop Elixir — return to Python + Slang + PyTorch (slangtorch)
date: 2026-07-13
status: accepted — Elixir/Nx stack removed; Python reproduction scaffold restored from 3d300f1; verify/ (Lean) unchanged
tier: baseline
supersedes: 20260713-elixir-nx-slang-workflow.md
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

The workflow was ported entirely to Elixir/Nx (`20260713-elixir-nx-slang-workflow.md`, commit `42821b3`),
and Phases 1–2 of the reproduction roadmap were executed on it. Decision-maker directive: **"go back to
python + shader slang pytorch. drop elixir."** The Python scaffold was checkpointed at `3d300f1` before
the port precisely to keep this reversible.

## Decision

Remove the Elixir stack (`lib/`, `test/`, `config/`, `mix.exs`, `pixi.toml` BEAM toolchain, the Erlang
NIF in `native/`); restore the **Python + slangtorch + PyTorch** reproduction scaffold from `3d300f1`
(`trellis_slat_fsq/` package, `tests/`, `pyproject.toml` with its `[tool.pixi]` tables). The Elixir line
lives on only in git history (`42821b3`, `73211d7`).

## What carries over from the Elixir phase (not lost)

- **`verify/` Lean package — unchanged and improved**: kernel-checked proofs (no `sorry`) that the FSQ
  index map is bijective onto [0, 8192) (`index_lt`, `index_decode`, `decode_index`, `index_inj`), plus
  the plausible-witness-dag build-time certification run. Language-neutral.
- **`trellis_slat_fsq/fsq_nif.slang`** — the plain-buffer deploy variant of the kernel (no slangtorch
  prelude), proven to compile via `slangc -target cpp` and produce correct indices (512 tokens in range,
  saturated extremes 0/8191 — validated through the Erlang NIF before removal). The Python deploy
  adapter (`slang_ports.py` `CompiledSlangServingAdapter`) should wrap the same compiled artifact via
  ctypes instead of a BEAM NIF.
- **Qwen3.5-0.8B operational findings**: HF weights are **xet-only** (plain HTTP 403) — fetch with
  `uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3.5-0.8B` and load locally. In Python,
  `transformers` supports `Qwen3_5ForConditionalGeneration` natively (config declares it), so no forced
  module mapping is needed — an advantage over Bumblebee.
- **Parquet stays the data format.** In Python: `duckdb` (pip wheel, no compiler needed — the Elixir-era
  rejection of duckdbex was about MSVC NIF builds, which doesn't apply) or `pyarrow`.

## Consequences

- The LM stack is `transformers` + `peft` again (LoRA unblocked — the Elixir blocker was lorax/nx).
- Torchx/Nx/Axon/Bumblebee findings are archived in history; the Bumblebee forced-mapping and
  BinaryBackend-slowness lessons don't apply to Python.
- pixi remains the toolchain/env manager via `pyproject.toml`'s `[tool.pixi]` tables.

## Verification

- `pytest` (Slang→PyTorch path only, per standing directive) once `slangtorch` + CUDA are present.
- `verify/`: `lake build` green (proofs + witness-DAG run).
- Reproduction phases resume per the roadmap MADR on the Python substrate.
