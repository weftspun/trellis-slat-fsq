# trellis-slat-fsq

TRELLIS.2 SLAT → ~512 **render-free** FSQ tokens per object (fixed grid, no learned codebook, no
training), for generating avatars / worlds / props.

No-training scope: FSQ is a fixed grid, so tokenization needs no training. The quantize+index math runs as
a **Slang** compute kernel (`trellis_slat_fsq/fsq.slang`) invoked from **PyTorch** via `slangtorch` on CUDA
(`pip install .[slang]`), with a pure-torch CPU reference for GPU-less environments. `[8,8,8,16]` = exactly
8192 codes → per-code parity with Kyvo's VQ.

The render aux-loss, renderer, TRELLIS.2 decoder, and LM port are obliterated (no budget) — reconstruction
quality is unvalidated, the accepted cost. Formal verification is via a
[`plausible-witness-dag`](https://github.com/fire/plausible-witness-dag) Lean proof (planned). See
[`decisions/20260713-no-budget-descope-render-free-fsq-only.md`](decisions/20260713-no-budget-descope-render-free-fsq-only.md).
