defmodule TrellisSlatFsq.SlangPort do
  @moduledoc """
  Hexagonal port for the Slang FSQ kernel (`priv/slang/fsq.slang`) — training, parity, and deployment.

  Adapters:

    * `TrellisSlatFsq.SlangPort.Nif` — compiled Slang (`slangc` -> PTX/CUDA or CPU object) invoked
      through a NIF, exchanging Nx tensor binaries. The deploy/parity path; the ONLY path covered by
      tests (per directive). Stub until the NIF is built.
    * `TrellisSlatFsq.SlangPort.NxReference` — pure `Nx.Defn` math (`TrellisSlatFsq.FSQ`), runs on
      Torchx (Windows) or EXLA. Fallback and the differentiable training path; NOT tested.
  """

  @doc "Encode a projected latent `[n, d]` into token indices `[n]` in `[0, 8192)`."
  @callback encode(latent :: Nx.Tensor.t()) :: Nx.Tensor.t()

  @doc "Pick the adapter: the Slang NIF when loaded, else the Nx reference."
  def select do
    if TrellisSlatFsq.SlangPort.Nif.loaded?(),
      do: TrellisSlatFsq.SlangPort.Nif,
      else: TrellisSlatFsq.SlangPort.NxReference
  end

  defmodule Nif do
    @moduledoc """
    Deployment adapter: compiled `fsq.slang` behind a NIF.

    Build (open work): `slangc priv/slang/fsq.slang -target ptx` (CUDA) or `-target cpp` (CPU),
    wrap with a small C NIF that takes the tensor binary + shape and returns the index binary.
    """
    @behaviour TrellisSlatFsq.SlangPort

    # Flipped to true by the NIF's @on_load once it exists; config until then.
    def loaded?, do: Application.get_env(:trellis_slat_fsq, :slang_nif_loaded, false)

    @impl true
    def encode(_latent) do
      raise "Slang NIF not built: compile priv/slang/fsq.slang (slangc -> ptx/cpp) and wire the NIF"
    end
  end

  defmodule NxReference do
    @moduledoc "Reference adapter: same math in Nx.Defn (Torchx/EXLA). Fallback + training; not tested."
    @behaviour TrellisSlatFsq.SlangPort

    @impl true
    def encode(latent) do
      {_codes, indices} = TrellisSlatFsq.FSQ.forward(latent)
      indices
    end
  end
end
