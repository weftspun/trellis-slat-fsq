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
