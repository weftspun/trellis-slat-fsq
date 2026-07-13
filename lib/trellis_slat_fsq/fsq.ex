defmodule TrellisSlatFsq.FSQ do
  @moduledoc """
  Finite Scalar Quantization (Mentzer et al.) in pure `Nx.Defn` — fixed grid, no learned codebook.

  Levels `[8, 8, 8, 16]` -> exactly 8192 codes (2^13), per-code parity with Kyvo's 8192-entry VQ.
  Differentiable via a straight-through `custom_grad` on the rounding, so it sits inside an Axon
  training graph. Same math as `priv/slang/fsq.slang`; the Slang kernel is the deploy/parity adapter
  (see `TrellisSlatFsq.SlangPort`). Backend: Torchx on Windows (EXLA optional on Linux).
  """

  import Nx.Defn

  @levels [8, 8, 8, 16]
  @basis @levels |> Enum.scan(1, fn l, acc -> acc * l end) |> then(&[1 | Enum.drop(&1, -1)])
  @codebook_size Enum.product(@levels)

  def levels, do: @levels
  def basis, do: @basis
  def codebook_size, do: @codebook_size

  @doc "Round with a straight-through gradient (identity on the backward pass)."
  defn round_ste(z) do
    custom_grad(Nx.round(z), [z], fn g -> [g] end)
  end

  @doc "Bound the latent into the representable range per level (tanh with even-level offset)."
  defn bound(z) do
    levels = Nx.tensor(@levels, type: :f32)
    eps = 1.0e-3
    half_l = (levels - 1) * (1 + eps) / 2
    offset = Nx.select(Nx.remainder(Nx.tensor(@levels), 2) == 0, 0.5, 0.0)
    shift = Nx.atanh(offset / half_l)
    Nx.tanh(z + shift) * half_l - offset
  end

  @doc "z (last axis = #{length(@levels)}) -> dequantized codes in ~[-1, 1], differentiable (STE)."
  defn quantize(z) do
    half_width = Nx.floor(Nx.tensor(@levels, type: :f32) / 2)
    round_ste(bound(z)) / half_width
  end

  @doc "Normalized codes -> integer indices in [0, #{@codebook_size})."
  defn codes_to_indices(codes) do
    half_width = Nx.floor(Nx.tensor(@levels, type: :f32) / 2)
    shifted = codes * half_width + half_width
    (shifted * Nx.tensor(@basis, type: :f32))
    |> Nx.sum(axes: [-1])
    |> Nx.round()
    |> Nx.as_type({:s, 64})
  end

  @doc "One forward pass: `{codes, indices}`."
  defn forward(z) do
    codes = quantize(z)
    {codes, codes_to_indices(codes)}
  end
end
