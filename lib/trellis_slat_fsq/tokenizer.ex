defmodule TrellisSlatFsq.Tokenizer do
  @moduledoc """
  Reconstructive SLAT tokenizer in Axon: encoder -> Residual FSQ -> decoder.

  SLAT `{batch, 8, 64, 64, 64}` (channels-first) -> Conv3d encoder -> `8^3 x 128` latent -> dense to the
  FSQ dim -> `TrellisSlatFsq.ResidualFSQ` (STE, differentiable) -> dense back -> ConvTranspose3d decoder
  -> reconstructed SLAT. ~512 tokens/object (8^3 positions; stage axis on top). Trained with latent recon
  plus the optional multi-view render aux-loss (`TrellisSlatFsq.RenderLoss`).
  """

  alias TrellisSlatFsq.ResidualFSQ

  @slat_channels 8
  @latent_channels 128
  @fsq_dim 4

  def token_budget, do: 512
  def quantized_grid, do: {8, @latent_channels}

  @doc "Full autoencoder Axon model: input \"slat\" -> reconstruction (FSQ bottleneck inside)."
  def model(opts \\ []) do
    num_quantizers = Keyword.get(opts, :num_quantizers, ResidualFSQ.default_num_quantizers())

    Axon.input("slat", shape: {nil, @slat_channels, 64, 64, 64})
    |> encoder()
    |> Axon.nx(&to_positions_last/1)
    |> Axon.dense(@fsq_dim)
    |> Axon.nx(fn z ->
      {codes, _indices} = ResidualFSQ.forward(z, num_quantizers: num_quantizers)
      codes
    end)
    |> Axon.dense(@latent_channels)
    |> Axon.nx(&to_channels_first/1)
    |> decoder()
  end

  @doc "Encoder half only (for `encode/3` at inference)."
  def encoder(input \\ Axon.input("slat", shape: {nil, @slat_channels, 64, 64, 64})) do
    input
    |> conv3(32)
    |> conv3(64)
    |> conv3(@latent_channels, activation: false)
  end

  defp conv3(x, out_channels, opts \\ []) do
    x =
      Axon.conv(x, out_channels,
        kernel_size: 3,
        strides: 2,
        padding: [{1, 1}, {1, 1}, {1, 1}],
        channels: :first
      )

    if Keyword.get(opts, :activation, true), do: Axon.silu(x), else: x
  end

  defp decoder(x) do
    x
    |> Axon.conv_transpose(64, kernel_size: 4, strides: 2, padding: [{1, 1}, {1, 1}, {1, 1}], channels: :first)
    |> Axon.silu()
    |> Axon.conv_transpose(32, kernel_size: 4, strides: 2, padding: [{1, 1}, {1, 1}, {1, 1}], channels: :first)
    |> Axon.silu()
    |> Axon.conv_transpose(@slat_channels, kernel_size: 4, strides: 2, padding: [{1, 1}, {1, 1}, {1, 1}], channels: :first)
  end

  # {b, c, 8, 8, 8} -> {b, 8, 8, 8, c}
  defp to_positions_last(t), do: Nx.transpose(t, axes: [0, 2, 3, 4, 1])
  # {b, 8, 8, 8, c} -> {b, c, 8, 8, 8}
  defp to_channels_first(t), do: Nx.transpose(t, axes: [0, 4, 1, 2, 3])
end
