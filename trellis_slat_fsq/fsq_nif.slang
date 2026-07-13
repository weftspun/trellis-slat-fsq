// Deploy/NIF variant of fsq.slang: plain structured buffers instead of slangtorch TensorView, so it
// compiles for CPU (`slangc -target cpp`) and Vulkan/CUDA without the slangtorch prelude. Same math as
// fsq.slang — keep the two in sync (the Lean proof in ../../verify covers the shared index map).

[numthreads(256, 1, 1)]
[shader("compute")]
void fsq_encode(
    uint3 tid : SV_DispatchThreadID,
    uniform StructuredBuffer<float> latent,   // [n * d] row-major projected latent
    uniform StructuredBuffer<int>   levels,   // [d] FSQ levels per dim
    uniform StructuredBuffer<int>   basis,    // [d] mixed-radix weights
    uniform RWStructuredBuffer<int> outIndex, // [n] token indices
    uniform uint n,
    uniform uint d)
{
    uint i = tid.x;
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

        float z       = latent[i * d + j];
        float bounded = tanh(z + shift) * half_l - offset;
        int   q       = int(round(bounded));
        int   code    = q + (L / 2);                          // -> [0, L-1]
        index        += code * basis[j];
    }
    outIndex[i] = index;
}
