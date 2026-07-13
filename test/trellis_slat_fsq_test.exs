defmodule TrellisSlatFsqTest do
  use ExUnit.Case, async: true

  # Per directive these tests cover ONLY the Slang -> Nx path (the compiled-Slang NIF adapter;
  # CPU target today, PTX/CUDA later). The Nx reference adapter is fallback/training only and is
  # deliberately NOT tested.
  @moduletag :slang_nif

  alias TrellisSlatFsq.SlangPort.Nif

  test "codebook is exactly 8192 ([8,8,8,16], per-code parity with Kyvo's VQ)" do
    assert TrellisSlatFsq.FSQ.codebook_size() == 8192
  end

  test "8^3 x 128 SLAT grid yields 512 tokens in [0, 8192)" do
    latent = Nx.broadcast(0.0, {512, 4})
    indices = Nif.encode(latent)
    assert Nx.shape(indices) == {512}
    assert Nx.to_number(Nx.reduce_min(indices)) >= 0
    assert Nx.to_number(Nx.reduce_max(indices)) < 8192
  end

  test "saturated inputs hit the index extremes (ground truth independent of any reference impl)" do
    lo = Nif.encode(Nx.broadcast(-100.0, {10, 4}))
    hi = Nif.encode(Nx.broadcast(100.0, {10, 4}))
    assert Nx.to_number(Nx.reduce_max(lo)) == 0
    assert Nx.to_number(Nx.reduce_min(hi)) == 8191
  end
end
