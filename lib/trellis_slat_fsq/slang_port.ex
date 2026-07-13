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
    # apply/3 keeps the check dynamic: the NIF replaces loaded?/0 at load time, so the compiler's
    # static "always false" view of the stub doesn't apply.
    if apply(TrellisSlatFsq.SlangPort.Nif, :loaded?, []),
      do: TrellisSlatFsq.SlangPort.Nif,
      else: TrellisSlatFsq.SlangPort.NxReference
  end

  defmodule Nif do
    @moduledoc """
    Deployment adapter: compiled Slang kernel (`priv/slang/fsq_nif.slang` -> `priv/fsq_nif.dll`)
    behind a NIF. Build with `native/build_windows.ps1` (slangc CPU target + MSVC). The NIF exchanges
    raw tensor binaries: f32 latent in, s32 indices out.
    """
    @behaviour TrellisSlatFsq.SlangPort

    @on_load :load_nif

    def load_nif do
      path = :filename.join(:code.priv_dir(:trellis_slat_fsq), ~c"fsq_nif")
      # Missing DLL is fine — loaded?/0 stays false and select/0 falls back to the Nx reference.
      case :erlang.load_nif(path, 0) do
        :ok -> :ok
        {:error, _} -> :ok
      end
    end

    @doc "True once the compiled-Slang NIF is loaded (the NIF overrides this stub)."
    def loaded?, do: false

    @impl true
    def encode(latent) do
      {n, d} = Nx.shape(latent)
      levels = TrellisSlatFsq.FSQ.levels()
      basis = TrellisSlatFsq.FSQ.basis()

      if d != length(levels),
        do: raise(ArgumentError, "latent last axis #{d} != #{length(levels)} FSQ levels")

      bin = latent |> Nx.as_type(:f32) |> Nx.backend_copy(Nx.BinaryBackend) |> Nx.to_binary()

      encode_raw(bin, n, d, levels, basis)
      |> Nx.from_binary(:s32)
      |> Nx.as_type(:s64)
    end

    @doc false
    def encode_raw(_bin, _n, _d, _levels, _basis), do: :erlang.nif_error(:slang_nif_not_loaded)
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
