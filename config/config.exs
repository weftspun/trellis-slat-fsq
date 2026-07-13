import Config

# Torchx (LibTorch) is the Nx default backend — the backend on Windows, where EXLA ships no XLA
# archive (decisions/20260713-elixir-nx-slang-workflow.md). Applies to every entry point (mix run,
# mix test, iex) without runtime Nx.global_default_backend calls.
config :nx, default_backend: Torchx.Backend
