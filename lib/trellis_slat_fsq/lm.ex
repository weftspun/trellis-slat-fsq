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
    {:ok, model_info} = Bumblebee.load_model({:hf, repo})
    {:ok, tokenizer} = Bumblebee.load_tokenizer({:hf, repo})
    {extend_embeddings(model_info, vocab.total), tokenizer}
  end

  @doc """
  Extend token embedding + output head to `new_vocab` rows (new rows randomly initialized).

  Axon parameter surgery — open work to make robust across Bumblebee versions; sketched here so the
  obligation is explicit rather than hidden.
  """
  def extend_embeddings(model_info, _new_vocab) do
    # TODO: locate the embedding + lm_head params in model_info.params, pad with new rows, and
    # rebuild the Axon graph with the widened dims. Depends on Bumblebee's Qwen3.5 param naming.
    model_info
  end
end
