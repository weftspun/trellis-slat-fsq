"""Hexagonal ports + adapters for the Slang FSQ kernel — training AND deployment.

The kernel must run across runtimes: training (slangtorch/CUDA), fallback/verification (torch CPU), and
deployment/serving (compiled Slang -> CUDA/Vulkan/CPU, no PyTorch). This module defines the port and the
adapters so callers depend on the interface, not a specific runtime.

Note: per the testing directive, only the Slang->PyTorch (CUDA) adapter is covered by tests; the torch-CPU
adapter is a fallback/oracle and is not tested. See decisions/ and the deployment note.
"""

from __future__ import annotations

from typing import Protocol

import torch


class FsqKernelPort(Protocol):
    """Encode a projected latent [N, D] -> token indices [N] in [0, prod(levels))."""

    def encode(self, projected: torch.Tensor) -> torch.Tensor:
        ...


class SlangCudaAdapter:
    """Training/CUDA adapter: runs fsq.slang via slangtorch. Requires a CUDA tensor + toolchain."""

    def __init__(self, tokenizer):
        self._tok = tokenizer  # a SlatFsqTokenizer

    def encode(self, projected: torch.Tensor) -> torch.Tensor:
        if not projected.is_cuda:
            raise RuntimeError("SlangCudaAdapter requires a CUDA tensor")
        return self._tok._fsq_slang(projected)


class TorchCpuAdapter:
    """Fallback/verification adapter: pure-torch reference. Not exercised by tests (by directive)."""

    def __init__(self, tokenizer):
        self._tok = tokenizer

    def encode(self, projected: torch.Tensor) -> torch.Tensor:
        return self._tok._fsq_torch(projected)


class CompiledSlangServingAdapter:
    """Deployment adapter: a compiled Slang kernel (CUDA/Vulkan/CPU) for serving without PyTorch.

    Placeholder: wire to a compiled `fsq.slang` artifact (e.g. via slangc -> target backend) at deploy time.
    Kept as an explicit port so the serving path is a first-class target, not an afterthought.
    """

    def __init__(self, compiled_module=None):
        self._module = compiled_module

    def encode(self, projected):  # projected: backend-native buffer
        if self._module is None:
            raise NotImplementedError(
                "Compiled-Slang serving adapter not wired. Compile fsq.slang for the target backend "
                "(slangc -> CUDA/Vulkan/CPU) and inject the module here."
            )
        return self._module.fsq_encode(projected)


def select_adapter(tokenizer, projected: torch.Tensor) -> FsqKernelPort:
    """Pick the training/fallback adapter by tensor placement (CUDA -> Slang, else torch CPU)."""
    return SlangCudaAdapter(tokenizer) if projected.is_cuda else TorchCpuAdapter(tokenizer)
