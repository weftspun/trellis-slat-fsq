// Minimal mesh rasterizer in Slang (slangtorch/CUDA) for dataset conditioning renders.
//
// Ingests pre-projected triangles (screen-space xy + view z, per-vertex albedo, face normal) and
// produces an RGBA image: alpha = coverage (background stays 0 — the object mask TRELLIS wants,
// no background-removal step). Shading is done HERE in Slang so styles are parameters:
//   toonLevels == 0 -> smooth lambert + ambient
//   toonLevels >= 2 -> quantized diffuse bands (toon/cel)
//
// Two passes over a per-pixel depth buffer (uint32, InterlockedMin):
//   raster_depth: thread per triangle, scan its bbox, write quantized depth
//   raster_shade: same scan; the triangle owning the min depth writes shaded color.

static const float AMBIENT = 0.35;
static const float DIFFUSE = 0.65;

float3 shade(float3 albedo, float3 normal, float3 lightDir, int toonLevels)
{
    float ndl = max(dot(normalize(normal), normalize(lightDir)), 0.0);
    if (toonLevels >= 2)
        ndl = floor(ndl * float(toonLevels)) / float(toonLevels - 1); // banded (cel) response
    return albedo * (AMBIENT + DIFFUSE * ndl);
}

// Signed area / edge function for barycentric rasterization.
float edgeFn(float2 a, float2 b, float2 c)
{
    return (c.x - a.x) * (b.y - a.y) - (c.y - a.y) * (b.x - a.x);
}

[AutoPyBindCUDA]
[CUDAKernel]
void raster_depth(
    TensorView<float> verts,   // [T, 3, 3] screen x, y, view z per corner
    TensorView<int>   depth,   // [H, W] initialized to 0x7FFFFFFF
    uniform uint nTris, uniform uint height, uniform uint width)
{
    uint t = (cudaBlockIdx().x * cudaBlockDim().x) + cudaThreadIdx().x;
    if (t >= nTris)
        return;

    float2 p0 = float2(verts[uint3(t, 0, 0)], verts[uint3(t, 0, 1)]);
    float2 p1 = float2(verts[uint3(t, 1, 0)], verts[uint3(t, 1, 1)]);
    float2 p2 = float2(verts[uint3(t, 2, 0)], verts[uint3(t, 2, 1)]);
    float3 z = float3(verts[uint3(t, 0, 2)], verts[uint3(t, 1, 2)], verts[uint3(t, 2, 2)]);

    float area = edgeFn(p0, p1, p2);
    if (abs(area) < 1e-9)
        return;

    int x0 = max(int(floor(min(p0.x, min(p1.x, p2.x)))), 0);
    int x1 = min(int(ceil(max(p0.x, max(p1.x, p2.x)))), int(width) - 1);
    int y0 = max(int(floor(min(p0.y, min(p1.y, p2.y)))), 0);
    int y1 = min(int(ceil(max(p0.y, max(p1.y, p2.y)))), int(height) - 1);

    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
    {
        float2 p = float2(float(x) + 0.5, float(y) + 0.5);
        float w0 = edgeFn(p1, p2, p) / area;
        float w1 = edgeFn(p2, p0, p) / area;
        float w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)
            continue;
        float zi = w0 * z.x + w1 * z.y + w2 * z.z;
        int zq = int(clamp(zi, 0.0, 1.0) * 2147000000.0);
        int prev;
        depth.InterlockedMin(uint2(uint(y), uint(x)), zq, prev);
    }
}

[AutoPyBindCUDA]
[CUDAKernel]
void raster_shade(
    TensorView<float> verts,    // [T, 3, 3]
    TensorView<float> colors,   // [T, 3] albedo per triangle
    TensorView<float> normals,  // [T, 3] face normal
    TensorView<int>   depth,    // [H, W] from raster_depth
    TensorView<float> image,    // [H, W, 4] rgba, zero-initialized
    uniform float3 lightDir, uniform int toonLevels,
    uniform uint nTris, uniform uint height, uniform uint width)
{
    uint t = (cudaBlockIdx().x * cudaBlockDim().x) + cudaThreadIdx().x;
    if (t >= nTris)
        return;

    float2 p0 = float2(verts[uint3(t, 0, 0)], verts[uint3(t, 0, 1)]);
    float2 p1 = float2(verts[uint3(t, 1, 0)], verts[uint3(t, 1, 1)]);
    float2 p2 = float2(verts[uint3(t, 2, 0)], verts[uint3(t, 2, 1)]);
    float3 z = float3(verts[uint3(t, 0, 2)], verts[uint3(t, 1, 2)], verts[uint3(t, 2, 2)]);

    float area = edgeFn(p0, p1, p2);
    if (abs(area) < 1e-9)
        return;

    float3 albedo = float3(colors[uint2(t, 0)], colors[uint2(t, 1)], colors[uint2(t, 2)]);
    float3 normal = float3(normals[uint2(t, 0)], normals[uint2(t, 1)], normals[uint2(t, 2)]);
    float3 rgb = shade(albedo, normal, lightDir, toonLevels);

    int x0 = max(int(floor(min(p0.x, min(p1.x, p2.x)))), 0);
    int x1 = min(int(ceil(max(p0.x, max(p1.x, p2.x)))), int(width) - 1);
    int y0 = max(int(floor(min(p0.y, min(p1.y, p2.y)))), 0);
    int y1 = min(int(ceil(max(p0.y, max(p1.y, p2.y)))), int(height) - 1);

    for (int y = y0; y <= y1; ++y)
    for (int x = x0; x <= x1; ++x)
    {
        float2 p = float2(float(x) + 0.5, float(y) + 0.5);
        float w0 = edgeFn(p1, p2, p) / area;
        float w1 = edgeFn(p2, p0, p) / area;
        float w2 = 1.0 - w0 - w1;
        if (w0 < 0.0 || w1 < 0.0 || w2 < 0.0)
            continue;
        float zi = w0 * z.x + w1 * z.y + w2 * z.z;
        int zq = int(clamp(zi, 0.0, 1.0) * 2147000000.0);
        if (depth[uint2(uint(y), uint(x))] != zq)
            continue;
        image[uint3(uint(y), uint(x), 0)] = rgb.x;
        image[uint3(uint(y), uint(x), 1)] = rgb.y;
        image[uint3(uint(y), uint(x), 2)] = rgb.z;
        image[uint3(uint(y), uint(x), 3)] = 1.0;
    }
}
