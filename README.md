# trellis-slat-fsq

TRELLIS.2 SLAT → ~512 **render-free** FSQ tokens per object (fixed grid, no learned codebook, no
training), for generating avatars / worlds / props.

Zero-compute scope: FSQ needs no training, so tokenization runs on CPU. The render aux-loss, renderer,
TRELLIS.2 decoder, and LM port are obliterated (no budget) — reconstruction quality is unvalidated, the
accepted cost of zero budget. See
[`decisions/20260713-no-budget-descope-render-free-fsq-only.md`](decisions/20260713-no-budget-descope-render-free-fsq-only.md).
