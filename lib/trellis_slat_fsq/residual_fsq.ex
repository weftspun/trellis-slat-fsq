defmodule TrellisSlatFsq.ResidualFSQ do
  @moduledoc """
  Residual FSQ: `num_quantizers` FSQ stages, each quantizing the running residual.

  Coarse-to-fine by construction: the first `id_prefix` stage indices form the compact retrieval ID;
  the full stack is the ~512-token reconstruction budget (8^3 positions x stages). Stages are unrolled
  at defn build time (fixed small count), so the whole thing stays one differentiable graph.
  """

  import Nx.Defn

  alias TrellisSlatFsq.FSQ

  @default_num_quantizers 8
  @default_id_prefix 1

  def default_num_quantizers, do: @default_num_quantizers

  @doc """
  Quantize `z` through `num_quantizers` residual stages.

  Returns `{quantized_sum, indices}` where `quantized_sum` reconstructs the input (differentiable)
  and `indices` has a trailing stage axis `[..., num_quantizers]`.
  """
  deftransform forward(z, opts \\ []) do
    num_quantizers = Keyword.get(opts, :num_quantizers, @default_num_quantizers)

    {sum, residual, indices} =
      Enum.reduce(1..num_quantizers, {Nx.multiply(z, 0.0), z, []}, fn _stage, {sum, residual, acc} ->
        {codes, idx} = FSQ.forward(residual)
        {Nx.add(sum, codes), Nx.subtract(residual, codes), [idx | acc]}
      end)

    _ = residual
    {sum, indices |> Enum.reverse() |> Nx.stack(axis: -1)}
  end

  @doc "The coarse retrieval-ID prefix: first `id_prefix` stage indices."
  def id_codes(indices, id_prefix \\ @default_id_prefix) do
    Nx.slice_along_axis(indices, 0, id_prefix, axis: -1)
  end
end
