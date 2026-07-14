"""Extract REAL TRELLIS SLATs: run microsoft/TRELLIS-image-large's SLAT flow model, capture latents.

Mirrors `TrellisImageTo3DPipeline.run` but stops BEFORE decode_slat — the structured latent itself is
the product (sparse feats [N, 8] over a 64^3 grid), densified to (8, 64, 64, 64) for our tokenizer.
Inputs are TRELLIS' bundled example images (real single-object images) by default.

Runs in the pixi `trellis` env (py3.11 + spconv-cu120 + torch 2.7.1+cu126; ATTN_BACKEND=sdpa):

    pixi run -e trellis python scripts/make_real_slats.py --trellis-repo <path> [--count 8]

Outputs: data/real_slats/<name>.pt  ({"dense": (8,64,64,64) f32, "coords": [N,4], "feats": [N,8]})
plus manifest.json. *.pt is gitignored — data stays out of the repo.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

os.environ.setdefault("ATTN_BACKEND", "sdpa")         # no xformers/flash-attn on this box
os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")  # sparse module only knows xformers|flash_attn
os.environ.setdefault("SPCONV_ALGO", "native")    # skip spconv autotuning


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trellis-repo", required=True, help="path to a microsoft/TRELLIS clone")
    ap.add_argument("--images", nargs="*", default=None, help="input images (default: repo examples)")
    ap.add_argument("--count", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "real_slats"))
    args = ap.parse_args()

    sys.path.insert(0, args.trellis_repo)

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        sys.exit("CUDA required — CPU is banned")

    from trellis.pipelines import TrellisImageTo3DPipeline

    images = args.images or sorted(glob.glob(os.path.join(args.trellis_repo, "assets", "example_image", "*.png")))
    images = images[: args.count]
    if not images:
        sys.exit("no input images found")

    os.makedirs(args.out, exist_ok=True)

    print("loading microsoft/TRELLIS-image-large ...")
    t0 = time.time()
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    print(f"pipeline ready in {time.time() - t0:.1f}s on {torch.cuda.get_device_name(0)}")

    manifest = []
    for i, path in enumerate(images):
        name = os.path.splitext(os.path.basename(path))[0]
        t1 = time.time()
        image = pipeline.preprocess_image(Image.open(path))
        cond = pipeline.get_cond([image])
        torch.manual_seed(i)
        coords = pipeline.sample_sparse_structure(cond, 1, {})
        slat = pipeline.sample_slat(cond, coords, {})  # sparse: feats [N, C], coords [N, 4]

        feats, sc = slat.feats.detach(), slat.coords.detach()
        channels = feats.shape[1]
        dense = torch.zeros(channels, 64, 64, 64, device=feats.device)
        dense[:, sc[:, 1].long(), sc[:, 2].long(), sc[:, 3].long()] = feats.T.float()

        out_path = os.path.join(args.out, f"{name}.pt")
        torch.save({"dense": dense.cpu(), "coords": sc.cpu(), "feats": feats.cpu()}, out_path)
        manifest.append({
            "id": name, "file": os.path.basename(out_path), "source_image": os.path.basename(path),
            "active_voxels": int(sc.shape[0]), "channels": int(channels),
            "feat_std": float(feats.float().std()),
        })
        print(f"[{i + 1}/{len(images)}] {name}: {sc.shape[0]} active voxels, C={channels} "
              f"({time.time() - t1:.1f}s)")

    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest)} real SLATs -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
