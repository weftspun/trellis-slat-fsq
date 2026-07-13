# trellis-slat-fsq

TRELLIS.2 SLAT → ~512 reconstructive **Residual-FSQ** tokens per object, for generating avatars /
worlds / props — a reproduction of Kyvo (arXiv:2506.08002) with Residual FSQ instead of VQ, in
**Python + PyTorch** with **Slang** compute kernels.

- **Quantizer**: Residual FSQ, per-stage levels `[8,8,8,16]` = exactly **8192 codes** (per-code parity
  with Kyvo's 8192-entry VQ); the coarse stage prefix doubles as the retrieval ID.
- **Kernels**: `trellis_slat_fsq/fsq.slang` (slangtorch/CUDA, training) and `fsq_nif.slang`
  (plain-buffer deploy variant, `slangc -target cpp`); hexagonal ports/adapters in `slang_ports.py`.
- **Method**: SLAT encoder → Residual FSQ → decoder, trained with Kyvo's multi-view render aux-loss
  (Slang.D 3DGS rasterizer) vs a latent-only ablation; unified LM = **Qwen3.5-0.8B**
  (`transformers` + `peft`). Inference is render-free.
- **Data**: Kyvo's released SLAT + CC0 `.usda` stages (quaternius / kenney / thebasemesh) via TRELLIS.2;
  Parquet as the storage format.
- **Verification**: `verify/` proves the FSQ index map bijective onto [0, 8192) in Lean (no `sorry`),
  with a build-time [plausible-witness-dag](https://github.com/fire/plausible-witness-dag)
  certification run: `cd verify && lake build`. Tests cover only the Slang kernel path.

Toolchain via [pixi](https://pixi.sh) (`[tool.pixi]` in `pyproject.toml`). Decisions in `decisions/`.
