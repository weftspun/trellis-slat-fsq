defmodule TrellisSlatFsq.MixProject do
  use Mix.Project

  @version "0.2.0"
  @source_url "https://github.com/weftspun/trellis-slat-fsq"

  def project do
    [
      app: :trellis_slat_fsq,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "TRELLIS.2 SLAT -> ~512 reconstructive Residual-FSQ tokens per object (avatars / worlds / props). " <>
          "Entirely Elixir/Nx; Slang kernel via ports/adapters; Torchx backend on Windows.",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      # LibTorch backend — the Nx backend on Windows (EXLA ships no Windows XLA archive).
      {:torchx, "~> 0.9"},
      # Encoder/decoder networks + training loop.
      {:axon, "~> 0.7"},
      {:polaris, "~> 0.1"},
      # Qwen3.5-0.8B backbone (train-time only). LoRA: lorax pins nx ~> 0.7 (conflicts with nx 0.9),
      # so LoRA is open work — vendor/port lorax or full fine-tune the 0.8B.
      {:bumblebee, "~> 0.6", optional: true},
      # Parquet is the data format, read/written through Explorer (Polars; precompiled NIF, no MSVC).
      {:explorer, "~> 0.10"}
    ] ++ exla_dep()
  end

  # EXLA only off-Windows: there is no precompiled XLA for x86_64-windows (Torchx is the Windows backend).
  defp exla_dep do
    case :os.type() do
      {:win32, _} -> []
      _ -> [{:exla, "~> 0.9", optional: true}]
    end
  end

  defp package do
    [
      licenses: ["MIT", "Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
