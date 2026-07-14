import pytest
import torch

from trellis_slat_fsq import SLAT_CHANNELS, SLAT_EDGE, SlatFsqTokenizer

# Per directive: test ONLY the Slang -> PyTorch (CUDA) path. Do NOT test the pure-torch CPU reference.
# Every check below drives the Slang kernel through encode() on a CUDA tensor and asserts against
# independently-known values (shape, range, saturated index extremes). Skips without slangtorch + CUDA.
pytest.importorskip("slangtorch")
if not torch.cuda.is_available():
    pytest.skip("Slang -> PyTorch path requires CUDA", allow_module_level=True)

DEVICE = "cuda"


def test_slang_codebook_size_is_exactly_8192():
    # [8, 8, 8, 16] = 8192, per-code parity with Kyvo's 8192-entry VQ.
    assert SlatFsqTokenizer(device=DEVICE).codebook_size == 8192


def test_slang_slat_grid_yields_512_tokens_in_range():
    tok = SlatFsqTokenizer(device=DEVICE)
    grid = torch.zeros(SLAT_EDGE, SLAT_EDGE, SLAT_EDGE, SLAT_CHANNELS, device=DEVICE)
    tokens = tok.encode(grid)  # CUDA latent -> Slang kernel
    assert tokens.is_cuda
    assert tokens.shape == (512,)
    assert tokens.dtype == torch.int64
    assert int(tokens.min()) >= 0 and int(tokens.max()) < tok.codebook_size


def test_slang_saturated_inputs_hit_index_extremes():
    # Ground truth independent of any reference impl: all dims saturated low -> index 0; high -> 8191.
    tok = SlatFsqTokenizer(device=DEVICE)
    lo = tok.encode(torch.full((10, tok.dim), -100.0, device=DEVICE))
    hi = tok.encode(torch.full((10, tok.dim), 100.0, device=DEVICE))
    assert int(lo.max()) == 0
    assert int(hi.min()) == tok.codebook_size - 1


def test_slang_full_codebook_bijective_on_gpu():
    # Exhaustive-on-hardware twin of the Lean witness-DAG certification (verify/): drive ALL 8192
    # code tuples through the Slang CUDA kernel and demand a bijection onto [0, 8192) — every index
    # hit exactly once. Inputs are constructed ANALYTICALLY by inverting the kernel's bound function
    # (z = atanh((q + offset)/half_l) - shift lands bounded(z) exactly on round-target q); the
    # torch/CPU reference path is never consulted (per directive).
    tok = SlatFsqTokenizer(device=DEVICE)
    levels = torch.tensor([8, 8, 8, 16], dtype=torch.float64, device=DEVICE)
    eps = 1e-3
    half_l = (levels - 1) * (1 + eps) / 2
    offset = torch.full_like(levels, 0.5)  # all levels even
    shift = torch.atanh(offset / half_l)

    # All 8192 per-dim integer codes [c0, c1, c2, c3], c_i in [0, L_i).
    grids = torch.meshgrid(*(torch.arange(int(l), device=DEVICE) for l in levels), indexing="ij")
    codes = torch.stack([g.reshape(-1) for g in grids], dim=-1).double()  # [8192, 4]
    round_target = codes - torch.floor(levels / 2)  # -> [-L/2, L/2 - 1]
    z = torch.atanh((round_target + offset) / half_l) - shift

    indices = tok.encode(z.float())
    assert indices.shape == (8192,)
    assert torch.equal(
        torch.sort(indices).values, torch.arange(8192, dtype=torch.int64, device=indices.device)
    )
