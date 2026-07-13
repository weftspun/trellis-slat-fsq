defmodule TrellisSlatFsq.LM do
  @moduledoc """
  Unified decoder-only LM over `[text | image | SLAT-residual-FSQ]`, backbone `Qwen/Qwen3.5-0.8B`.

  Loaded via Bumblebee; LoRA via Lorax. Vocabulary is extended with the image VQGAN codes, the SLAT
  residual-FSQ codes, and the Kyvo-style boundary tokens; sequence layout follows Kyvo
  (`BOS ... OUTSEP <target> EOS`, 3D tokens in raw grid order).

  Caveats (recorded in the MADR): Bumblebee must support the Qwen3.5 architecture — if it does not,
  porting the architecture to Axon is open work. Embedding resize is Axon parameter surgery
  (`extend_embeddings/2` sketches it).
  """

  @backbone "Qwen/Qwen3.5-0.8B"
  @image_codebook 8192
  @slat_codebook 8192
  @specials ~w(bos eos outsep boimg eoimg bo3d eo3d)a

  def backbone, do: @backbone

  @doc "Vocab layout: `[ base text | image | slat | specials ]` with region offsets."
  def vocab(text_vocab_size) do
    image_offset = text_vocab_size
    slat_offset = image_offset + @image_codebook
    special_offset = slat_offset + @slat_codebook

    specials =
      @specials |> Enum.with_index(fn name, i -> {name, special_offset + i} end) |> Map.new()

    %{
      text_vocab_size: text_vocab_size,
      image_offset: image_offset,
      slat_offset: slat_offset,
      specials: specials,
      total: special_offset + length(@specials)
    }
  end

  @doc "Kyvo-layout sequence: `BOS + [img] + [3d] + [text] + OUTSEP + <target 3d> + EOS` (raw 3D order)."
  def assemble_sequence(vocab, parts) do
    sp = vocab.specials

    Enum.concat([
      [sp.bos],
      wrap(parts[:image_tokens], sp.boimg, sp.eoimg, vocab.image_offset),
      wrap(parts[:slat_tokens], sp.bo3d, sp.eo3d, vocab.slat_offset),
      List.wrap(parts[:text]),
      case parts[:target_slat_tokens] do
        nil -> []
        target -> [sp.outsep | wrap(target, sp.bo3d, sp.eo3d, vocab.slat_offset)]
      end,
      [sp.eos]
    ])
  end

  defp wrap(nil, _b, _e, _offset), do: []
  defp wrap(tokens, b, e, offset), do: [b] ++ Enum.map(tokens, &(&1 + offset)) ++ [e]

  @doc """
  Load the Qwen3.5-0.8B backbone via Bumblebee and extend its embeddings to `vocab.total`.

  Requires the `:bumblebee` optional dep; returns `{model_info, tokenizer}`.
  """
  def load(vocab, opts \\ []) do
    repo = Keyword.get(opts, :backbone, @backbone)
    source = Keyword.get(opts, :source, {:hf, repo})

    # Bumblebee 0.7 has no mapping for "Qwen3_5ForConditionalGeneration"; the Qwen3.5 dense config
    # parses under the Qwen3 module (verified: spec + weights load, forward pass sane).
    {:ok, model_info} =
      Bumblebee.load_model(source,
        module: Bumblebee.Text.Qwen3,
        architecture: :for_causal_language_modeling
      )

    {:ok, tokenizer} = Bumblebee.load_tokenizer(source)
    {extend_embeddings(model_info, vocab.total), tokenizer}
  end

  @doc """
  Extend the vocabulary dimension to `new_vocab` rows: rebuild the Axon graph from a re-configured
  spec, and pad every parameter that carries a vocab-sized axis (token embedding; untied LM head)
  with small random init. Robust to param naming: any axis equal to the old `vocab_size` is padded.
  """
  def extend_embeddings(%{params: params, spec: spec} = model_info, new_vocab) do
    old_vocab = spec.vocab_size

    if new_vocab == old_vocab do
      model_info
    else
      new_spec = Bumblebee.configure(spec, vocab_size: new_vocab)
      new_model = Bumblebee.build_model(new_spec)

      data =
        Map.new(params.data, fn {layer, layer_params} ->
          {layer,
           Map.new(layer_params, fn {name, tensor} ->
             {name, pad_vocab_axis(tensor, old_vocab, new_vocab)}
           end)}
        end)

      %{model_info | model: new_model, params: %{params | data: data}, spec: new_spec}
    end
  end

  defp pad_vocab_axis(tensor, old_vocab, new_vocab) do
    shape = tensor |> Nx.shape() |> Tuple.to_list()

    case Enum.find_index(shape, &(&1 == old_vocab)) do
      nil ->
        tensor

      axis ->
        pad_shape = shape |> List.replace_at(axis, new_vocab - old_vocab) |> List.to_tuple()
        {noise, _} = Nx.Random.normal(Nx.Random.key(0), 0.0, 0.02, shape: pad_shape, type: Nx.type(tensor))
        Nx.concatenate([tensor, noise], axis: axis)
    end
  end
end
