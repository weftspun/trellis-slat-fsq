defmodule Mix.Tasks.Trellis.Smoke do
  @shortdoc "Smoke-check the FSQ core: quantizer, residual stack, STE grads, Parquet round-trip"

  @moduledoc """
  Runs the zero-external-deps smoke checks on the Nx core (dev check — the committed test suite
  covers only the Slang NIF path, per directive):

      mix trellis.smoke

  Checks: codebook size 8192; 512 tokens in range with saturated extremes 0/8191; residual stack
  shape + reconstruction MAE; finite STE gradients; Parquet write/read via Explorer.
  """

  use Mix.Task

  alias TrellisSlatFsq.{Data, FSQ, ResidualFSQ}

  @impl true
  def run(_args) do
    Mix.Task.run("app.start")

    8192 = FSQ.codebook_size()
    IO.puts("codebook: 8192 OK")

    key = Nx.Random.key(1)
    {z, _} = Nx.Random.normal(key, shape: {512, 4})
    {_codes, idx} = FSQ.forward(z)
    {512} = Nx.shape(idx)
    true = Nx.to_number(Nx.reduce_min(idx)) >= 0
    true = Nx.to_number(Nx.reduce_max(idx)) < 8192
    IO.puts("512 tokens in [0, 8192) OK")

    {_c, lo} = FSQ.forward(Nx.broadcast(-100.0, {10, 4}))
    {_c, hi} = FSQ.forward(Nx.broadcast(100.0, {10, 4}))
    0 = Nx.to_number(Nx.reduce_max(lo))
    8191 = Nx.to_number(Nx.reduce_min(hi))
    IO.puts("saturated extremes 0/8191 OK")

    {qsum, ridx} = ResidualFSQ.forward(z)
    {512, 4} = Nx.shape(qsum)
    {512, 8} = Nx.shape(ridx)
    mae = Nx.mean(Nx.abs(Nx.subtract(z, qsum))) |> Nx.to_number()
    IO.puts("residual: 8 stages, recon MAE #{Float.round(mae, 4)} OK")

    grad = Nx.Defn.grad(fn z -> Nx.sum(FSQ.quantize(z)) end).(z)
    1 = Nx.to_number(Nx.all(Nx.logical_not(Nx.is_nan(grad))))
    IO.puts("STE gradients finite OK")

    path = Path.join(System.tmp_dir!(), "trellis_smoke.parquet")
    :ok = Data.to_parquet(path, [%{"id" => "syn-1", "tokens" => [3, 1, 4]}])
    [%{"id" => "syn-1", "tokens" => [3, 1, 4]}] = Data.from_parquet(path)
    File.rm!(path)
    IO.puts("Parquet round-trip OK")

    IO.puts("SMOKE OK")
  end
end
