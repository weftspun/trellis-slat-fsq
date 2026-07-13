# Per directive: only the Slang -> Nx path is tested, and it needs the compiled Slang NIF + CUDA.
# Excluded by default; run with `mix test --include slang_cuda` (pixi task: test-slang).
ExUnit.start(exclude: [:slang_cuda])
