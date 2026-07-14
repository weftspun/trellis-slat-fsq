"""Generate a SLAT training dataset the way Kyvo did (arXiv:2506.08002, Appendix A.1, "Objaverse"):

    "we extract a single render of each asset only, and pass it as image-conditioning to the
     pre-trained TRELLIS slat generator. We use the resultant slats as the inputs to the 3D VQ-VAE
     encoder during training [...] the synthetic slats are effective substitutes"

Sources:
  --usd DIR     CC0 .usda assets — geometry + materials ingested via OpenUSD (usd-core), rendered
                ONCE by OUR Slang rasterizer (trellis_slat_fsq/raster.slang via slangtorch on CUDA;
                lambert now, --toon N for cel bands), then fed to TRELLIS. Alpha is native coverage.
  --images DIR  pre-rendered/real object images fed to TRELLIS directly

Runs in the pixi `trellis` env:

    pixi run -e trellis python scripts/make_slat_dataset.py --trellis-repo <clone> --usd <dir> --count 24

Outputs `data/slat_dataset/<name>.pt` + `manifest.json`; resumable (skips existing). *.pt gitignored.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.pop("BOOST_ROOT", None)                    # a global BOOST_ROOT breaks spconv import
os.environ.setdefault("ATTN_BACKEND", "sdpa")
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")


# --- USD -> single render: OUR Slang rasterizer (raster.slang via slangtorch) -----------------------------

_RASTER = None


def _load_raster():
    global _RASTER
    if _RASTER is None:
        import _slang_toolchain

        _slang_toolchain.pin()
        import slangtorch

        _RASTER = slangtorch.loadModule(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "trellis_slat_fsq", "raster.slang"))
    return _RASTER


def render_once_slang(tm, size: int = 518, toon_levels: int = 0):
    """Single 3/4-view render via the Slang rasterizer -> RGBA PIL image (alpha = coverage)."""
    import numpy as np
    import torch
    from PIL import Image

    module = _load_raster()
    device = "cuda"

    v = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=device)
    f = torch.tensor(np.asarray(tm.faces), dtype=torch.long, device=device)
    vc = torch.tensor(np.asarray(tm.visual.vertex_colors)[:, :3] / 255.0,
                      dtype=torch.float32, device=device)

    # normalize to unit box, 3/4 lookAt camera, perspective project to screen space
    center = (v.min(0).values + v.max(0).values) / 2
    v = (v - center) / (v.max(0).values - v.min(0).values).max().clamp(min=1e-6)
    eye = torch.tensor([1.1, 0.75, 1.1], device=device) * 1.45
    fwd = -eye / eye.norm()
    right = torch.linalg.cross(fwd, torch.tensor([0.0, 1.0, 0.0], device=device))
    right = right / right.norm()
    up = torch.linalg.cross(right, fwd)
    rel = v - eye
    cam = torch.stack([rel @ right, rel @ up, rel @ fwd], dim=-1)  # view space, +z forward
    focal = 1.0 / np.tan(np.radians(25.0))
    sx = (cam[:, 0] / cam[:, 2]) * focal
    sy = (cam[:, 1] / cam[:, 2]) * focal
    screen = torch.stack([
        (sx * 0.5 + 0.5) * size,
        (0.5 - sy * 0.5) * size,
        (cam[:, 2] - cam[:, 2].min()) / (cam[:, 2].max() - cam[:, 2].min() + 1e-6),
    ], dim=-1)

    tri = screen[f]                                  # [T, 3, 3]
    tri_color = vc[f].mean(dim=1)                    # [T, 3] flat albedo
    e1 = v[f[:, 1]] - v[f[:, 0]]
    e2 = v[f[:, 2]] - v[f[:, 0]]
    tri_normal = torch.linalg.cross(e1, e2)
    tri_normal = tri_normal / tri_normal.norm(dim=-1, keepdim=True).clamp(min=1e-9)

    n_tris = tri.shape[0]
    depth = torch.full((size, size), 2**31 - 1, dtype=torch.int32, device=device)
    image = torch.zeros(size, size, 4, dtype=torch.float32, device=device)
    block, grid = 256, ((n_tris + 255) // 256, 1, 1)

    module.raster_depth(verts=tri.contiguous(), depth=depth,
                        nTris=n_tris, height=size, width=size
                        ).launchRaw(blockSize=(block, 1, 1), gridSize=grid)
    module.raster_shade(verts=tri.contiguous(), colors=tri_color.contiguous(),
                        normals=tri_normal.contiguous(), depth=depth, image=image,
                        lightDir=(0.5, 0.8, 0.4), toonLevels=toon_levels,
                        nTris=n_tris, height=size, width=size
                        ).launchRaw(blockSize=(block, 1, 1), gridSize=grid)

    return Image.fromarray((image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8), "RGBA")


def usd_to_trimesh(usd_path: str):
    """Load a .usda into one combined trimesh with per-vertex colors (UsdPreviewSurface diffuse)."""
    import numpy as np
    import trimesh
    from pxr import Usd, UsdGeom, UsdShade

    stage = Usd.Stage.Open(usd_path)
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue
        mesh = UsdGeom.Mesh(prim)
        points = np.array(mesh.GetPointsAttr().Get() or [], dtype=np.float64)
        counts = np.array(mesh.GetFaceVertexCountsAttr().Get() or [], dtype=np.int64)
        indices = np.array(mesh.GetFaceVertexIndicesAttr().Get() or [], dtype=np.int64)
        if len(points) == 0 or len(counts) == 0:
            continue

        # world transform
        xform = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(0), dtype=np.float64)
        pts = np.c_[points, np.ones(len(points))] @ xform
        points = pts[:, :3]

        # triangulate fans
        tris, offset = [], 0
        for c in counts:
            for k in range(1, c - 1):
                tris.append((indices[offset], indices[offset + k], indices[offset + k + 1]))
            offset += c

        # diffuse color from the bound material (fallback mid-gray)
        color = [0.6, 0.6, 0.6]
        binding = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
        if binding:
            for shader_prim in Usd.PrimRange(binding.GetPrim()):
                shader = UsdShade.Shader(shader_prim)
                if shader and shader.GetShaderId() == "UsdPreviewSurface":
                    inp = shader.GetInput("diffuseColor")
                    if inp and inp.Get() is not None:
                        color = list(inp.Get())
                    break

        tm = trimesh.Trimesh(vertices=points, faces=np.array(tris), process=False)
        tm.visual.vertex_colors = np.tile(
            (np.array(color + [1.0]) * 255).astype(np.uint8), (len(points), 1)
        )
        meshes.append(tm)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


# --- TRELLIS image -> SLAT -------------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis-repo", required=True)
    ap.add_argument("--usd", default=None, help="dir tree of .usda assets (rendered once each)")
    ap.add_argument("--images", default=None, help="dir of object images (fed to TRELLIS directly)")
    ap.add_argument("--count", type=int, default=24)
    ap.add_argument("--toon", type=int, default=0, help="cel-shading bands (0 = smooth lambert)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "slat_dataset"))
    args = ap.parse_args()

    sys.path.insert(0, args.trellis_repo)
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        sys.exit("CUDA required — CPU is banned")
    os.makedirs(args.out, exist_ok=True)

    # Build the (name, PIL image) work list.
    jobs = []
    if args.usd:
        usds = sorted(glob.glob(os.path.join(args.usd, "**", "*.usda"), recursive=True))
        print(f"{len(usds)} .usda assets found; taking {args.count}")
        renders_dir = os.path.join(args.out, "renders")
        os.makedirs(renders_dir, exist_ok=True)
        step = max(1, len(usds) // args.count)
        for path in usds[::step][: args.count]:
            name = os.path.splitext(os.path.basename(path))[0]
            if os.path.exists(os.path.join(args.out, f"{name}.pt")):
                continue
            png = os.path.join(renders_dir, f"{name}.png")
            if not os.path.exists(png):
                tm = usd_to_trimesh(path)
                if tm is None or len(tm.faces) < 8:
                    print(f"  skip {name}: no usable mesh")
                    continue
                try:
                    render_once_slang(tm, toon_levels=args.toon).save(png)
                except Exception as e:  # noqa: BLE001 — corrupt assets skipped, not fatal
                    print(f"  skip {name}: {e}")
                    continue
            jobs.append((name, Image.open(png), os.path.relpath(path, args.usd)))
    if args.images:
        for path in sorted(glob.glob(os.path.join(args.images, "*.png")))[: args.count]:
            name = os.path.splitext(os.path.basename(path))[0]
            if not os.path.exists(os.path.join(args.out, f"{name}.pt")):
                jobs.append((name, Image.open(path), os.path.basename(path)))
    if not jobs:
        sys.exit("nothing to do (no sources, or all outputs already exist)")

    from trellis.pipelines import TrellisImageTo3DPipeline

    print(f"loading TRELLIS ({len(jobs)} objects queued) ...")
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()

    manifest_path = os.path.join(args.out, "manifest.json")
    manifest = json.load(open(manifest_path)) if os.path.exists(manifest_path) else []
    for i, (name, image, source) in enumerate(jobs):
        t1 = time.time()
        image = pipeline.preprocess_image(image)
        cond = pipeline.get_cond([image])
        torch.manual_seed(hash(name) % (2**31))
        coords = pipeline.sample_sparse_structure(cond, 1, {})
        slat = pipeline.sample_slat(cond, coords, {})

        feats, sc = slat.feats.detach(), slat.coords.detach()
        dense = torch.zeros(feats.shape[1], 64, 64, 64, device=feats.device)
        dense[:, sc[:, 1].long(), sc[:, 2].long(), sc[:, 3].long()] = feats.T.float()
        torch.save({"dense": dense.cpu(), "coords": sc.cpu(), "feats": feats.cpu()},
                   os.path.join(args.out, f"{name}.pt"))
        manifest.append({"id": name, "source": source, "active_voxels": int(sc.shape[0]),
                         "channels": int(feats.shape[1]), "feat_std": float(feats.float().std())})
        print(f"[{i + 1}/{len(jobs)}] {name}: {sc.shape[0]} voxels ({time.time() - t1:.1f}s)")

    json.dump(manifest, open(manifest_path, "w"), indent=2)
    print(f"dataset: {len(manifest)} SLATs -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
