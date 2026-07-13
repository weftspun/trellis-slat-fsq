# trellis-slat-fsq

TRELLIS.2 SLAT → ~512 reconstructive **Residual-FSQ** tokens per object, for generating avatars /
worlds / props — a reproduction of Kyvo (arXiv:2506.08002) with Residual FSQ instead of VQ, **entirely
in Elixir/Nx**.

- **Quantizer**: Residual FSQ, per-stage levels `[8,8,8,16]` = exactly **8192 codes** (per-code parity
  with Kyvo's VQ); the coarse stage prefix doubles as the retrieval ID.
- **Kernel**: shared Slang shader (`priv/slang/fsq.slang`) reached through ports/adapters
  (`TrellisSlatFsq.SlangPort`); Nx math runs on **Torchx (LibTorch) on Windows**, EXLA optional on Linux.
- **Method**: Axon encoder → Residual FSQ → decoder, trained with Kyvo's multi-view render aux-loss
  (Slang.D 3DGS rasterizer adapter) vs a latent-only ablation; unified LM = **Qwen3.5-0.8B** (Bumblebee).
- **Data**: Kyvo's released SLAT + CC0 `.usda` stages (quaternius / kenney / thebasemesh) via TRELLIS.2.
- **Verification**: `verify/` Lean package certifies the FSQ index map via
  [plausible-witness-dag](https://github.com/fire/plausible-witness-dag); tests cover only the Slang→Nx
  path (`pixi run test-slang`).

Toolchain via [pixi](https://pixi.sh): `pixi run setup && pixi run test`. Decisions in `decisions/`.
