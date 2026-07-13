"""Differentiable FSQ + Residual FSQ layers in PyTorch (for training the reconstructive tokenizer).

The inference-time, index-only path lives in ``__init__.py`` (``SlatFsqTokenizer`` / the Slang kernel).
These layers additionally return the *continuous* dequantized codes with a straight-through estimator, so
they can sit inside an autograd graph for the render-aux-loss training of the reproduction
(``decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md``).
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = ["FSQ", "ResidualFSQ"]


def round_ste(z: torch.Tensor) -> torch.Tensor:
    """Round with a straight-through gradient (identity on the backward pass)."""
    return z + (z.round() - z).detach()


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al.).

    Quantizes the last dim (size ``len(levels)``) to a fixed grid. Returns dequantized codes in the input
    space (grad via STE) and integer indices in ``[0, prod(levels))``.
    """

    def __init__(self, levels=(8, 8, 8, 16)):
        super().__init__()
        levels_t = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("levels", levels_t, persistent=False)
        basis = torch.cat([torch.ones(1, dtype=torch.int64), torch.cumprod(levels_t[:-1], dim=0)])
        self.register_buffer("basis", basis, persistent=False)
        self.dim = len(levels)

    @property
    def codebook_size(self) -> int:
        return int(torch.prod(self.levels).item())

    def _bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        L = self.levels.to(z.dtype)
        half_l = (L - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0).to(z.dtype)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """z -> dequantized codes in ~[-1, 1], differentiable via STE."""
        quantized = round_ste(self._bound(z))
        half_width = (self.levels // 2).to(z.dtype)
        return quantized / half_width

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        half_width = (self.levels // 2).to(codes.dtype)
        shifted = codes * half_width + half_width  # -> per-dim integers in [0, level-1]
        return (shifted * self.basis.to(codes.dtype)).sum(dim=-1).round().to(torch.int64)

    def forward(self, z: torch.Tensor):
        codes = self.quantize(z)
        indices = self.codes_to_indices(codes)
        return codes, indices


class ResidualFSQ(nn.Module):
    """Residual FSQ: ``num_quantizers`` FSQ stages, each quantizing the running residual.

    Coarse-to-fine by construction: the first ``id_prefix`` stages form the compact retrieval ID; the full
    stack is the ~512-token reconstruction budget (see the port MADR). Returns the summed dequantized code
    (for reconstruction) and per-stage indices ``[..., num_quantizers]``.
    """

    def __init__(self, levels=(8, 8, 8, 16), num_quantizers: int = 8, id_prefix: int = 1):
        super().__init__()
        self.stages = nn.ModuleList([FSQ(levels) for _ in range(num_quantizers)])
        self.num_quantizers = num_quantizers
        self.id_prefix = id_prefix

    @property
    def codebook_size(self) -> int:
        return self.stages[0].codebook_size

    def forward(self, z: torch.Tensor):
        residual = z
        quantized_sum = torch.zeros_like(z)
        indices = []
        for stage in self.stages:
            codes, idx = stage(residual)
            residual = residual - codes
            quantized_sum = quantized_sum + codes
            indices.append(idx)
        indices = torch.stack(indices, dim=-1)  # [..., num_quantizers]
        return quantized_sum, indices

    def id_codes(self, indices: torch.Tensor) -> torch.Tensor:
        """The coarse retrieval-ID prefix: first ``id_prefix`` stage indices."""
        return indices[..., : self.id_prefix]
