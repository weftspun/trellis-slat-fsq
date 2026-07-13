defmodule TrellisSlatFsq do
  @moduledoc """
  TRELLIS.2 SLAT -> ~512 reconstructive Residual-FSQ tokens per object — entirely in Elixir/Nx.

  Kyvo-method reproduction (arXiv:2506.08002) with Residual FSQ instead of VQ; per-stage levels
  `[8, 8, 8, 16]` = exactly 8192 codes (per-code parity with Kyvo's VQ). The FSQ kernel is shared with
  `priv/slang/fsq.slang` (Slang), reached through `TrellisSlatFsq.SlangPort` adapters; the Nx math runs
  on **Torchx (LibTorch) on Windows** — EXLA optional on Linux.

  Modules: `FSQ` / `ResidualFSQ` (quantizers), `Tokenizer` (Axon autoencoder), `RenderLoss` (Kyvo
  aux-loss), `Data` (Kyvo SLAT + CC0 USD stages), `LM` (Qwen3.5-0.8B via Bumblebee), `Train` / `Eval`
  (ablation + metrics), `SlangPort` (deploy adapters). Decisions in `decisions/`.
  """

  alias TrellisSlatFsq.{ResidualFSQ, SlangPort, Tokenizer}

  defdelegate token_budget, to: Tokenizer
  defdelegate quantized_grid, to: Tokenizer

  # Note: the Nx default backend (Torchx) is set in config/config.exs — no runtime init is needed.

  @doc """
  Render-free encode: projected latent (last axis = 4) or raw FSQ-dim tensor -> token indices.

  Routes through the active `SlangPort` adapter (Slang NIF when built, Nx reference otherwise).
  """
  def fsq_encode(latent), do: SlangPort.select().encode(latent)

  @doc "Residual encode: latent (last axis = 4) -> `{quantized, indices [..., stages]}`."
  defdelegate residual_encode(z, opts \\ []), to: ResidualFSQ, as: :forward
end
