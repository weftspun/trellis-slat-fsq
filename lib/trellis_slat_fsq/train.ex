defmodule TrellisSlatFsq.Train do
  @moduledoc """
  Tokenizer training with the render-aux vs latent-only ABLATION (the headline reproduction), in Axon.

  Kyvo's central tokenizer finding: latent reconstruction alone is insufficient; the multi-view render
  aux-loss (150 views) improves reconstruction. Two configs:

      ablation: :latent_only  -> loss = MSE(recon, slat)
      ablation: :render_aux   -> loss = MSE + w_render * (L1 + D-SSIM) over rendered views

  Rendering is TRAINING-TIME ONLY. Backend: Torchx on Windows (EXLA optional on Linux). The renderer is
  the Slang.D 3DGS adapter (`TrellisSlatFsq.RenderLoss.Renderer` behaviour).
  """

  alias TrellisSlatFsq.{RenderLoss, Tokenizer}

  @doc """
  Train the reconstructive tokenizer over a stream of `%{slat: tensor}` batches.

  Options: `:ablation` (`:render_aux` | `:latent_only`), `:renderer` (required for `:render_aux`),
  `:w_render`, `:n_views`, `:steps`, `:lr`, `:num_quantizers`.
  """
  def tokenizer(batches, opts \\ []) do
    ablation = Keyword.get(opts, :ablation, :render_aux)
    renderer = Keyword.get(opts, :renderer)
    w_render = Keyword.get(opts, :w_render, 1.0)
    n_views = Keyword.get(opts, :n_views, RenderLoss.n_views_default())
    steps = Keyword.get(opts, :steps, 1000)
    lr = Keyword.get(opts, :lr, 2.0e-4)

    if ablation == :render_aux and renderer == nil,
      do: raise(ArgumentError, "ablation: :render_aux needs :renderer (Slang 3DGS adapter)")

    model = Tokenizer.model(Keyword.take(opts, [:num_quantizers]))

    loss_fn = fn slat, recon ->
      latent = Nx.mean(Nx.pow(recon - slat, 2))

      case ablation do
        :latent_only ->
          latent

        :render_aux ->
          %{render_total: render} = RenderLoss.multi_view(renderer, recon, slat, n_views: n_views)
          Nx.add(latent, Nx.multiply(w_render, render))
      end
    end

    data = Stream.map(batches, fn %{slat: slat} -> {%{"slat" => batch_dim(slat)}, batch_dim(slat)} end)

    model
    |> Axon.Loop.trainer(loss_fn, Polaris.Optimizers.adamw(learning_rate: lr))
    |> Axon.Loop.run(data, Axon.ModelState.empty(), iterations: steps)
  end

  defp batch_dim(t), do: if(Nx.rank(t) == 4, do: Nx.new_axis(t, 0), else: t)
end
