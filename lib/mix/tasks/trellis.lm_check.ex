defmodule Mix.Tasks.Trellis.LmCheck do
  @shortdoc "Load Qwen3.5-0.8B, extend the unified vocab, and run a timed forward pass"

  @moduledoc """
  End-to-end LM readiness check (Phase 2 of the reproduction plan):

      mix trellis.lm_check [--snapshot PATH] [--prompt TEXT]

  Loads the backbone (local HF snapshot if given — required while the repo is xet-only, see the
  hf-xet workaround in decisions/), extends embeddings to the unified `[text|image|slat|specials]`
  vocab, runs a timed forward pass, and prints the next-token sanity plus a sample Kyvo-layout
  sequence. The Nx backend comes from config/config.exs (Torchx).
  """

  use Mix.Task

  alias TrellisSlatFsq.LM

  @impl true
  def run(args) do
    Mix.Task.run("app.start")

    {opts, _, _} = OptionParser.parse(args, strict: [snapshot: :string, prompt: :string])
    prompt = opts[:prompt] || "The capital of France is"

    source =
      case opts[:snapshot] do
        nil -> {:hf, LM.backbone()}
        path -> {:local, path}
      end

    vocab = LM.vocab(151_936)
    IO.puts("unified vocab: total=#{vocab.total} slat_offset=#{vocab.slat_offset}")

    t0 = System.monotonic_time(:millisecond)
    {%{model: model, params: params, spec: spec}, tokenizer} = LM.load(vocab, source: source)
    IO.puts("loaded + extended in #{System.monotonic_time(:millisecond) - t0}ms: vocab #{spec.vocab_size}")

    t1 = System.monotonic_time(:millisecond)
    inputs = Bumblebee.apply_tokenizer(tokenizer, prompt)
    out = Axon.predict(model, params, inputs)
    next_id = out.logits[[0, -1]] |> Nx.argmax() |> Nx.to_number()

    IO.puts(
      "forward in #{System.monotonic_time(:millisecond) - t1}ms; logits #{inspect(Nx.shape(out.logits))}; " <>
        "next=#{inspect(Bumblebee.Tokenizer.decode(tokenizer, [next_id]))}"
    )

    seq = LM.assemble_sequence(vocab, %{text: [1, 2], slat_tokens: [7, 8], target_slat_tokens: [9]})
    IO.puts("sequence sample: #{inspect(seq)}")
    IO.puts("LM CHECK OK")
  end
end
