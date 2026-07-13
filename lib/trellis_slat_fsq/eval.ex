defmodule TrellisSlatFsq.Eval do
  @moduledoc """
  Metrics + the render-aux vs latent-only comparison (headline result), in Nx.

  Render-space L1 / SSIM / L2 over held-out SLAT reconstructions, token-set Jaccard, text accuracy —
  mirroring Kyvo's reporting.
  """

  alias TrellisSlatFsq.RenderLoss

  def l1(x, y), do: Nx.mean(Nx.abs(Nx.subtract(x, y))) |> Nx.to_number()
  def l2(x, y), do: Nx.mean(Nx.pow(Nx.subtract(x, y), 2)) |> Nx.to_number()

  def ssim(x, y) do
    dssim = RenderLoss.d_ssim(Nx.clip(x, 0, 1), Nx.clip(y, 0, 1)) |> Nx.to_number()
    1.0 - 2.0 * dssim
  end

  @doc "Token-set IoU — proxy for scene-structure agreement on discrete tokens."
  def jaccard(pred_idx, target_idx) do
    p = pred_idx |> Nx.to_flat_list() |> MapSet.new()
    t = target_idx |> Nx.to_flat_list() |> MapSet.new()
    union = MapSet.union(p, t)
    if MapSet.size(union) == 0, do: 1.0, else: MapSet.size(MapSet.intersection(p, t)) / MapSet.size(union)
  end

  def text_accuracy(pred_ids, target_ids) do
    Nx.mean(Nx.equal(pred_ids, target_ids)) |> Nx.to_number()
  end

  @doc "Mean render-space L1/SSIM/L2 of `predict_fn` reconstructions over held-out SLATs."
  def reconstruction_report(predict_fn, slats, renderer, opts \\ []) do
    n_views = Keyword.get(opts, :n_views, RenderLoss.n_views_default())

    reports =
      Enum.map(slats, fn slat ->
        slat = Nx.new_axis(slat, 0)
        recon = predict_fn.(slat)
        img_r = renderer.render(recon, n_views) |> flatten()
        img_t = renderer.render(slat, n_views) |> flatten()
        {l1(img_r, img_t), ssim(img_r, img_t), l2(img_r, img_t)}
      end)

    n = length(reports)

    %{
      render_l1: (reports |> Enum.map(&elem(&1, 0)) |> Enum.sum()) / n,
      render_ssim: (reports |> Enum.map(&elem(&1, 1)) |> Enum.sum()) / n,
      render_l2: (reports |> Enum.map(&elem(&1, 2)) |> Enum.sum()) / n,
      n: n
    }
  end

  @doc "Headline: does render-aux beat latent-only on render-space reconstruction?"
  def compare_ablations(render_aux, latent_only) do
    %{
      render_aux: render_aux,
      latent_only: latent_only,
      render_aux_wins:
        render_aux.render_l1 < latent_only.render_l1 and
          render_aux.render_ssim > latent_only.render_ssim
    }
  end

  defp flatten(t) do
    {b, v, c, h, w} = Nx.shape(t)
    Nx.reshape(t, {b * v, c, h, w})
  end
end
