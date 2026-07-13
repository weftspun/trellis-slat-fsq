"""Reconstructive SLAT tokenizer: encoder -> Residual FSQ -> decoder (trainable).

Reproduces Kyvo's 3D VQ-VAE with Residual FSQ instead of VQ. SLAT 64^3 x 8 -> encoder -> 8^3 x 128 latent
-> Residual FSQ (~512 tokens) -> decoder -> reconstructed SLAT. Trained with a latent recon loss plus the
optional multi-view render aux-loss (see render_loss.py). Encoder/decoder are dense Conv3d stacks here; a
sparse/TRELLIS-faithful backbone can be swapped in without changing the FSQ interface.

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .fsq_torch import ResidualFSQ

SLAT_IN_EDGE = 64
SLAT_IN_CHANNELS = 8
LATENT_EDGE = 8
LATENT_CHANNELS = 128


class SlatEncoder(nn.Module):
    """Dense Conv3d downsampler 64^3 x 8 -> 8^3 x C (three /2 stages)."""

    def __init__(self, out_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(SLAT_IN_CHANNELS, 32, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(32, 64, 3, stride=2, padding=1), nn.SiLU(),
            nn.Conv3d(64, out_channels, 3, stride=2, padding=1),
        )

    def forward(self, x):  # x: [B, 8, 64, 64, 64]
        return self.net(x)  # [B, C, 8, 8, 8]


class SlatDecoder(nn.Module):
    """Dense ConvTranspose3d upsampler 8^3 x C -> 64^3 x 8."""

    def __init__(self, in_channels: int = LATENT_CHANNELS):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose3d(in_channels, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose3d(32, SLAT_IN_CHANNELS, 4, stride=2, padding=1),
        )

    def forward(self, z):  # z: [B, C, 8, 8, 8]
        return self.net(z)  # [B, 8, 64, 64, 64]


class SlatFsqReconstructiveTokenizer(nn.Module):
    """Encoder -> Residual FSQ -> decoder. Emits ~512 tokens/object (8^3 positions x stages prefix)."""

    def __init__(self, levels=(8, 8, 8, 16), num_quantizers: int = 8, id_prefix: int = 1):
        super().__init__()
        self.encoder = SlatEncoder(LATENT_CHANNELS)
        self.decoder = SlatDecoder(LATENT_CHANNELS)
        # Project the 128-channel latent down to the FSQ dim, quantize, and project back for the decoder.
        self.to_fsq = nn.Linear(LATENT_CHANNELS, len(levels))
        self.from_fsq = nn.Linear(len(levels), LATENT_CHANNELS)
        self.quantizer = ResidualFSQ(levels, num_quantizers=num_quantizers, id_prefix=id_prefix)

    def encode(self, slat):
        """SLAT -> (indices [B, 8,8,8, Q], quantized latent grid for the decoder)."""
        z = self.encoder(slat)                       # [B, C, 8, 8, 8]
        z = z.permute(0, 2, 3, 4, 1)                 # [B, 8, 8, 8, C]
        codes, indices = self.quantizer(self.to_fsq(z))
        return indices, codes

    def decode(self, codes):
        z = self.from_fsq(codes).permute(0, 4, 1, 2, 3)  # [B, C, 8, 8, 8]
        return self.decoder(z)

    def forward(self, slat):
        indices, codes = self.encode(slat)
        recon = self.decode(codes)
        return {"recon": recon, "indices": indices, "codes": codes}
