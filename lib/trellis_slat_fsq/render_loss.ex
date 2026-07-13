defmodule TrellisSlatFsq.RenderLoss do
  @moduledoc """
  Kyvo-style multi-view render auxiliary loss (L1 + D-SSIM [+ LPIPS]) in Nx. TRAINING-TIME ONLY.

  The differentiable renderer is a behaviour: the intended adapter is the Slang.D 3D Gaussian Splatting
  rasterizer (`google/slang-gaussian-rasterization`, compiled + NIF-wrapped like the FSQ kernel) fed by
  TRELLIS.2's SLAT->Gaussian decoder. LPIPS has no Elixir implementation — open work (port or NIF); until
  then the loss is L1 + D-SSIM.
  """

  import Nx.Defn

  @n_views_default 150

  defmodule Renderer do
    @moduledoc "Differentiable SLAT -> multi-view images `{b, v, 3, h, w}` behaviour."
    @callback render(slat :: Nx.Tensor.t(), n_views :: pos_integer()) :: Nx.Tensor.t()
  end

  def n_views_default, do: @n_views_default

  @doc "Structural dissimilarity (1 - SSIM)/2 over `{n, c, h, w}` images in [0, 1]."
  defn d_ssim(x, y) do
    # 11x11 Gaussian window, sigma 1.5, depthwise via feature_group_size.
    win = gaussian_window()
    c = Nx.axis_size(x, 1)
    kernel = Nx.broadcast(win, {c, 1, 11, 11})
    opts = [padding: [{5, 5}, {5, 5}], feature_group_size: c]

    mu_x = Nx.conv(x, kernel, opts)
    mu_y = Nx.conv(y, kernel, opts)
    sig_x = Nx.conv(x * x, kernel, opts) - mu_x * mu_x
    sig_y = Nx.conv(y * y, kernel, opts) - mu_y * mu_y
    sig_xy = Nx.conv(x * y, kernel, opts) - mu_x * mu_y

    c1 = 0.01 * 0.01
    c2 = 0.03 * 0.03

    ssim =
      ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) /
        ((mu_x * mu_x + mu_y * mu_y + c1) * (sig_x + sig_y + c2))

    (1 - Nx.mean(ssim)) / 2
  end

  defnp gaussian_window do
    coords = Nx.iota({11}) - 5
    g = Nx.exp(-(coords * coords) / (2 * 1.5 * 1.5))
    g = g / Nx.sum(g)
    Nx.outer(g, g) |> Nx.reshape({1, 1, 11, 11})
  end

  @doc """
  L1 + D-SSIM over `renderer` views of reconstructed vs target SLAT.

  Returns `%{render_total: ..., l1: ..., d_ssim: ...}`. `renderer` implements `Renderer`.
  """
  def multi_view(renderer, recon_slat, target_slat, opts \\ []) do
    n_views = Keyword.get(opts, :n_views, @n_views_default)
    img_r = renderer.render(recon_slat, n_views) |> flatten_views()
    img_t = renderer.render(target_slat, n_views) |> flatten_views() |> Nx.Defn.Kernel.stop_grad()

    l1 = Nx.mean(Nx.abs(img_r - img_t))
    dssim = d_ssim(Nx.clip(img_r, 0, 1), Nx.clip(img_t, 0, 1))
    %{render_total: Nx.add(l1, dssim), l1: l1, d_ssim: dssim}
  end

  # {b, v, 3, h, w} -> {b*v, 3, h, w}
  defp flatten_views(t) do
    {b, v, c, h, w} = Nx.shape(t)
    Nx.reshape(t, {b * v, c, h, w})
  end
end
