"""Render-free SLAT -> FSQ tokenizer (fixed grid, no training) on Slang + PyTorch.

TRELLIS.2 SLAT quantized grid (8^3 x 128) -> 512 tokens/object over an 8192-code fixed grid.

FSQ has no learned codebook: quantization is a deterministic grid, so tokenization needs no training.
The quantize+index math runs as a Slang compute kernel (``fsq.slang``) invoked from PyTorch via
``slangtorch`` on CUDA; a numerically identical pure-torch reference (``_fsq_torch``) serves CPU and any
environment without the Slang toolchain.

Scope: see ``decisions/20260713-no-budget-descope-render-free-fsq-only.md``. The render aux-loss,
differentiable renderer, TRELLIS.2 decoder, and learned encoder remain obliterated (no *training* budget);
reconstruction quality is unvalidated. This substrate reinstates ``torch`` (inference kernel only, still no
training). The per-code level set ``[8, 8, 8, 16]`` gives exactly 8192 codes -- per-code parity with Kyvo's
8192-entry VQ (see ``decisions/20260713-port-kyvo-residual-fsq.md``).
"""

from __future__ import annotations

import os

import torch

__all__ = ["SlatFsqTokenizer", "DEFAULT_LEVELS", "SLAT_CHANNELS", "SLAT_EDGE"]

# Kyvo's quantized SLAT grid: 8^3 spatial positions, 128 channels.
SLAT_EDGE = 8
SLAT_CHANNELS = 128

# Exact 8192-code grid (= 2^13), all levels >= 8 (best FSQ reconstruction). Per-code parity with Kyvo VQ.
DEFAULT_LEVELS = (8, 8, 8, 16)

# Slang kernel is loaded lazily on first CUDA encode so importing the package never requires the toolchain.
_SLANG_MODULE = None
_SLANG_PATH = os.path.join(os.path.dirname(__file__), "fsq.slang")


def _load_slang():
    global _SLANG_MODULE
    if _SLANG_MODULE is None:
        import slangtorch  # deferred: only needed for the CUDA fast path

        _SLANG_MODULE = slangtorch.loadModule(_SLANG_PATH)
    return _SLANG_MODULE


class SlatFsqTokenizer:
    """Deterministic, training-free SLAT -> FSQ token sequence.

    Parameters
    ----------
    levels:
        FSQ levels; ``prod(levels)`` is the codebook size. Default ``(8, 8, 8, 16)`` -> 8192.
    input_dim:
        SLAT channel dim to project down from (default 128). A latent whose last axis already equals
        ``len(levels)`` is quantized directly with no projection.
    seed:
        Seed for the fixed orthonormal projection. Fixed, not learned -- determinism only.
    device:
        Torch device for the projection/levels buffers (default CPU). Move a latent to CUDA to hit the
        Slang kernel; CPU latents use the pure-torch reference.
    """

    def __init__(self, levels=DEFAULT_LEVELS, input_dim: int = SLAT_CHANNELS, seed: int = 0, device=None):
        device = torch.device(device) if device is not None else torch.device("cpu")
        self.levels = torch.tensor(levels, dtype=torch.int32, device=device)
        if self.levels.ndim != 1 or self.levels.numel() == 0:
            raise ValueError("levels must be a non-empty 1-D sequence")
        self.input_dim = int(input_dim)
        self.dim = int(self.levels.numel())

        # Mixed-radix basis for folding per-dim codes -> a single index in [0, codebook_size).
        cumprod = torch.cumprod(self.levels[:-1].long(), dim=0)
        self.basis = torch.cat([torch.ones(1, dtype=torch.long, device=device), cumprod]).to(torch.int32)

        # Fixed (untrained) orthonormal projection SLAT-channels -> FSQ dim. Deterministic via seed.
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        q, _ = torch.linalg.qr(torch.randn(self.input_dim, self.dim, generator=gen))
        self.projection = q[:, : self.dim].to(device)

    @property
    def device(self):
        return self.levels.device

    @property
    def codebook_size(self) -> int:
        """Number of distinct codes (= product of levels)."""
        return int(torch.prod(self.levels).item())

    # --- FSQ core --------------------------------------------------------------------------------------

    def _fsq_torch(self, z: torch.Tensor) -> torch.Tensor:
        """Pure-torch reference for the Slang kernel (Mentzer et al. FSQ). Used on CPU / no-Slang."""
        L = self.levels.to(z.dtype)
        eps = 1e-3
        half_l = (L - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0).to(z.dtype)
        shift = torch.atanh(offset / half_l)
        bounded = torch.tanh(z + shift) * half_l - offset
        code = torch.round(bounded) + (self.levels // 2).to(z.dtype)  # -> [0, L-1]
        return (code * self.basis.to(z.dtype)).sum(dim=-1).round().to(torch.int64)

    def _fsq_slang(self, z: torch.Tensor) -> torch.Tensor:
        """CUDA fast path: run fsq.slang via slangtorch. Requires a CUDA tensor + the Slang toolchain."""
        module = _load_slang()
        n = z.shape[0]
        out = torch.empty(n, dtype=torch.int32, device=z.device)
        block = 256
        grid = (n + block - 1) // block
        module.fsq_encode(
            latent=z.contiguous(),
            levels=self.levels.to(z.device),
            basis=self.basis.to(z.device),
            out=out,
        ).launchRaw(blockSize=(block, 1, 1), gridSize=(grid, 1, 1))
        return out.to(torch.int64)

    # --- public API ------------------------------------------------------------------------------------

    def encode(self, latent) -> torch.Tensor:
        """Encode a SLAT latent into a flat sequence of token indices.

        ``latent`` last axis must be either ``len(levels)`` (quantized directly) or ``input_dim``
        (projected first). Returns a 1-D ``int64`` tensor, one token per spatial position -- e.g. an
        ``(8, 8, 8, 128)`` grid yields ``512`` tokens in raw grid order (matching Kyvo's 3D layout).
        Latents on CUDA use the Slang kernel; CPU latents use the pure-torch reference.
        """
        latent = torch.as_tensor(latent, dtype=torch.float32, device=self.device)
        last = latent.shape[-1]
        flat = latent.reshape(-1, last)
        if last == self.dim:
            proj = flat
        elif last == self.input_dim:
            proj = flat @ self.projection.to(flat.dtype)
        else:
            raise ValueError(
                f"latent last axis {last} must equal len(levels)={self.dim} or input_dim={self.input_dim}"
            )
        use_slang = proj.is_cuda and os.environ.get("TRELLIS_SLAT_FSQ_FORCE_TORCH") != "1"
        return (self._fsq_slang(proj) if use_slang else self._fsq_torch(proj)).reshape(-1)

    def token_budget(self, latent) -> int:
        """Number of tokens ``encode`` would emit for ``latent`` (product of its spatial axes)."""
        latent = torch.as_tensor(latent)
        return int(torch.prod(torch.tensor(latent.shape[:-1])).item()) if latent.ndim > 1 else 1

    def serialize(self, indices, bos: int, eos: int, sep: int | None = None) -> list[int]:
        """Wrap a token sequence Kyvo-style: ``BOS + tokens [+ SEP] + EOS`` (raw 3D order, no reorder)."""
        flat = torch.as_tensor(indices).reshape(-1).tolist()
        seq = [int(bos), *(int(i) for i in flat)]
        if sep is not None:
            seq.append(int(sep))
        seq.append(int(eos))
        return seq
