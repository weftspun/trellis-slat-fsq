defmodule TrellisSlatFsq do
  @moduledoc """
  TRELLIS.2 SLAT -> reconstructive FSQ tokens (Elixir bindings via Pythonx).

  Encodes a Structured LATent (SLAT) into ~512 Finite-Scalar-Quantization tokens per object,
  for generating avatars / worlds / props.

  See `decisions/20260713-generation-slat-fsq-render-auxloss.md`.

  Target dimensions (from the decision):

    * SLAT latent: 64^3 x 8 (~20k latents)
    * Quantized: 8^3 x 128
    * ~512 tokens/object (~40x compression)

  FSQ (fixed grid, no learned codebook) is used over VQ. Because the grid is fixed, tokenization
  needs **no training and no GPU** — it runs on CPU. This module embeds CPython via
  [Pythonx](https://hex.pm/packages/pythonx) and calls the real `torch` +
  `vector-quantize-pytorch` FSQ implementation.

  Zero-compute descope (see `decisions/20260713-no-budget-descope-render-free-fsq-only.md`): the render
  aux-loss, differentiable renderer, TRELLIS.2 decoder, and learned encoder training are OBLITERATED (no
  budget). What remains is render-free FSQ tokenization of SLAT — reconstruction quality is unvalidated,
  which is the accepted cost of zero budget.

  ## Usage

      TrellisSlatFsq.init()
      {codes, indices} = TrellisSlatFsq.fsq_quantize(latent, [8, 8, 8, 6, 5])
  """

  @token_budget 512
  @quantized_edge 8
  @quantized_channels 128

  # Python environment for the embedded interpreter (uv-managed). Mirrors the repo's pyproject.toml.
  @pyproject """
  [project]
  name = "trellis_slat_fsq_runtime"
  version = "0.0.0"
  requires-python = ">=3.10"
  dependencies = [
    "torch>=2.5",
    "vector-quantize-pytorch",
    "numpy",
  ]
  """

  @doc "Approximate reconstructive token budget per object (~512)."
  @spec token_budget() :: pos_integer()
  def token_budget, do: @token_budget

  @doc "Quantized latent grid as `{spatial_edge, channels}` -> `{8, 128}`."
  @spec quantized_grid() :: {pos_integer(), pos_integer()}
  def quantized_grid, do: {@quantized_edge, @quantized_channels}

  @doc """
  Initializes the embedded Python interpreter with `torch` + `vector-quantize-pytorch`.

  Call once at startup before `fsq_quantize/2`. First run downloads the Python packages
  (torch is large); subsequent runs reuse the uv-managed environment.
  """
  @spec init() :: :ok
  def init do
    Pythonx.uv_init(@pyproject)
    :ok
  end

  @doc """
  Quantizes a latent tensor into FSQ codes using `vector_quantize_pytorch.FSQ`.

  * `latent` - a nested list shaped `{..., dim}` where `dim == length(levels)`
  * `levels` - the FSQ levels list, e.g. `[8, 8, 8, 6, 5]`

  Returns `{codes, indices}` decoded to Elixir terms. Requires `init/0` first.
  """
  @spec fsq_quantize(list(), [pos_integer()]) :: {list(), list()}
  def fsq_quantize(latent, levels) when is_list(latent) and is_list(levels) do
    {result, _globals} =
      Pythonx.eval(
        """
        import torch
        from vector_quantize_pytorch import FSQ

        fsq = FSQ(levels=list(levels))
        x = torch.tensor(latent, dtype=torch.float32)
        xhat, indices = fsq(x)
        (xhat.detach().tolist(), indices.detach().tolist())
        """,
        %{"latent" => latent, "levels" => levels}
      )

    Pythonx.decode(result)
  end
end
