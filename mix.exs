defmodule TrellisSlatFsq.MixProject do
  use Mix.Project

  @version "0.1.0"
  @source_url "https://github.com/weftspun/trellis-slat-fsq"

  def project do
    [
      app: :trellis_slat_fsq,
      version: @version,
      elixir: "~> 1.18",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description:
        "TRELLIS.2 SLAT -> ~512 reconstructive FSQ tokens per object (avatars / worlds / props). " <>
          "Elixir bindings over the Python implementation via Pythonx.",
      package: package(),
      source_url: @source_url
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      # Embeds CPython in the BEAM to call the Python torch + vector-quantize-pytorch (FSQ) code.
      {:pythonx, "~> 0.4"}
    ]
  end

  defp package do
    [
      licenses: ["MIT", "Apache-2.0"],
      links: %{"GitHub" => @source_url}
    ]
  end
end
