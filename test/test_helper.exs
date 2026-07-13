# Per directive: only the Slang -> Nx path is tested (the compiled-Slang NIF; CPU target today,
# PTX/CUDA later). Excluded unless the NIF is present; run with `mix test --include slang_nif`
# (pixi task: test-slang) after building it via native/build_windows.ps1.
ExUnit.start(exclude: [:slang_nif])
