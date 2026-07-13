// Finite Scalar Quantization (Mentzer et al.) encode kernel — fixed grid, no learned params, no training.
//
// One thread per spatial position: bounds each projected latent dim, rounds it to the FSQ grid, then
// folds the per-dim codes into a single mixed-radix index in `[0, prod(levels))`. With levels [8,8,8,16]
// that index range is exactly `[0, 8192)` — per-code parity with Kyvo's 8192-entry VQ.
//
// Invoked from PyTorch via slangtorch (see __init__.py). atanh is expanded to 0.5*log((1+a)/(1-a))
// so the kernel needs no atanh intrinsic.

[AutoPyBindCUDA]
[CUDAKernel]
void fsq_encode(
    TensorView<float> latent,   // [N, D] projected latent (float32)
    TensorView<int>   levels,   // [D]   FSQ levels per dim
    TensorView<int>   basis,    // [D]   mixed-radix weights (cumprod of levels, exclusive)
    TensorView<int>   out)      // [N]   output token indices
{
    uint i = (cudaBlockIdx().x * cudaBlockDim().x) + cudaThreadIdx().x;
    uint n = latent.size(0);
    uint d = latent.size(1);
    if (i >= n)
        return;

    int index = 0;
    for (uint j = 0; j < d; ++j)
    {
        int   L      = levels[j];
        float eps    = 1e-3f;
        float half_l = (float(L) - 1.0f) * (1.0f + eps) * 0.5f;
        float offset = (L % 2 == 0) ? 0.5f : 0.0f;
        float a      = offset / half_l;
        float shift  = 0.5f * log((1.0f + a) / (1.0f - a));   // atanh(a)

        float z       = latent[uint2(i, j)];
        float bounded = tanh(z + shift) * half_l - offset;
        int   q       = int(round(bounded));
        int   code    = q + (L / 2);                          // map to [0, L-1]
        index        += code * basis[j];
    }
    out[i] = index;
}
