"""Data sources for the reproduction: Kyvo's released data AND our CC0 USD stages.

Both yield SLAT tensors (64^3 x 8) for the tokenizer, plus (for the LM) the Kyvo-style multimodal
sequences. External ingestion (HF download; USD -> mesh -> TRELLIS.2 SLAT) is injected/stubbed so the
package imports without the heavy deps.

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

from typing import Protocol

import torch
from torch.utils.data import Dataset

SLAT_SHAPE = (8, 64, 64, 64)  # channels, D, H, W


class TrellisEncoder(Protocol):
    """USD/mesh -> TRELLIS.2 SLAT. External (TRELLIS.2 weights); injected for the CC0 USD path."""

    def __call__(self, usd_path: str) -> torch.Tensor:
        """Return a SLAT tensor of shape (8, 64, 64, 64)."""
        ...


class KyvoSlatDataset(Dataset):
    """Kyvo's released SLAT for CLEVR / ObjaWorld / Objectron.

    `records` is a list of local paths to pre-extracted SLAT tensors (downloaded from the Kyvo HF repo).
    Kyvo does not release the 3D VQ-VAE codebook, only the image VQGAN codebooks, so we re-quantize these
    SLATs with Residual FSQ here.
    """

    def __init__(self, records: list[str]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        slat = torch.load(self.records[i])
        assert tuple(slat.shape) == SLAT_SHAPE, f"expected {SLAT_SHAPE}, got {tuple(slat.shape)}"
        return {"slat": slat.float(), "source": "kyvo", "id": self.records[i]}


class UsdStageSlatDataset(Dataset):
    """CC0 .usda assets (geometry + materials) -> TRELLIS.2 SLAT.

    `usd_paths` point at .usda files from the taskspun CC0 stages (quaternius / kenney / thebasemesh).
    `encoder` turns each into SLAT; it is external (TRELLIS.2) and must be supplied.
    """

    def __init__(self, usd_paths: list[str], encoder: TrellisEncoder):
        self.usd_paths = usd_paths
        self.encoder = encoder

    def __len__(self) -> int:
        return len(self.usd_paths)

    def __getitem__(self, i: int) -> dict:
        slat = self.encoder(self.usd_paths[i])
        assert tuple(slat.shape) == SLAT_SHAPE, f"expected {SLAT_SHAPE}, got {tuple(slat.shape)}"
        return {"slat": slat.float(), "source": "usd", "id": self.usd_paths[i]}


def synthetic_slat_dataset(n: int = 8, seed: int = 0) -> list[dict]:
    """Toy SLATs for exercising the harness on CPU with no external data (not a scientific result)."""
    gen = torch.Generator().manual_seed(seed)
    return [{"slat": torch.randn(*SLAT_SHAPE, generator=gen), "source": "synthetic", "id": f"syn-{i}"}
            for i in range(n)]
