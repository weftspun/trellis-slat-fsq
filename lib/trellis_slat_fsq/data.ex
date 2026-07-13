defmodule TrellisSlatFsq.Data do
  @moduledoc """
  Data sources: Kyvo's released SLAT (CLEVR / ObjaWorld / Objectron) AND CC0 USD stages.

  Both yield SLAT tensors `{8, 64, 64, 64}`. Kyvo does not release the 3D VQ-VAE codebook, so its SLATs
  are re-quantized with Residual FSQ here. The USD path (`.usda` with geometry + materials from the
  taskspun `quaternius-stage` / `kenney-stage` / `thebasemesh-stage` repos) needs the external TRELLIS.2
  encoder, injected as a `UsdEncoder` behaviour.

  Persistence: **Parquet via Explorer** (Polars-backed) — tokenized sequences, SLAT records, and eval
  results are read with `from_parquet/2` and written with `to_parquet/2`. `*.parquet` stays gitignored.
  """

  @slat_shape {8, 64, 64, 64}

  def slat_shape, do: @slat_shape

  defmodule UsdEncoder do
    @moduledoc "External TRELLIS.2 encoder: `.usda` path -> SLAT `{8, 64, 64, 64}`."
    @callback encode(usd_path :: String.t()) :: Nx.Tensor.t()
  end

  @doc "Stream Kyvo SLAT records from local `.nx` binary files (pre-downloaded from the Kyvo HF repo)."
  def kyvo_stream(paths) do
    Stream.map(paths, fn path ->
      slat = path |> File.read!() |> Nx.deserialize()

      if Nx.shape(slat) != @slat_shape,
        do: raise(ArgumentError, "expected #{inspect(@slat_shape)}, got #{inspect(Nx.shape(slat))}")

      %{slat: slat, source: :kyvo, id: path}
    end)
  end

  @doc "Stream SLAT from CC0 `.usda` assets through the injected TRELLIS.2 encoder."
  def usd_stream(usd_paths, encoder) do
    Stream.map(usd_paths, fn path ->
      %{slat: encoder.encode(path), source: :usd, id: path}
    end)
  end

  @doc """
  Read a Parquet file into row maps via Explorer (Polars).

  `columns` optionally projects a subset. Example — tokenized sequences back into Nx:

      TrellisSlatFsq.Data.from_parquet("tokens.parquet", ["id", "tokens"])
      |> Enum.map(fn %{"id" => id, "tokens" => tokens} ->
        %{id: id, tokens: Nx.tensor(tokens, type: :s64)}
      end)
  """
  def from_parquet(path, columns \\ nil) do
    opts = if columns, do: [columns: columns], else: []
    path |> Explorer.DataFrame.from_parquet!(opts) |> Explorer.DataFrame.to_rows()
  end

  @doc """
  Write `rows` (list of maps, e.g. `[%{"id" => "a", "tokens" => [1, 2]}]`) to Parquet via Explorer.

  Existing files are replaced.
  """
  def to_parquet(path, rows) do
    rows |> Explorer.DataFrame.new() |> Explorer.DataFrame.to_parquet!(path)
    :ok
  end

  @doc "Toy SLATs for exercising the harness with no external data (not a scientific result)."
  def synthetic(n \\ 8, key \\ Nx.Random.key(0)) do
    {tensors, _key} =
      Enum.map_reduce(1..n, key, fn i, key ->
        {t, key} = Nx.Random.normal(key, shape: @slat_shape)
        {%{slat: t, source: :synthetic, id: "syn-#{i}"}, key}
      end)

    tensors
  end
end
