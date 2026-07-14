"""One training turn of the reconstructive SLAT tokenizer (roadmap Phase 5, latent_only arm).

Exercises the real optimization path — Conv3d encoder -> ResidualFSQ (STE) -> decoder, MSE latent
reconstruction — on GPU. The render_aux arm needs the Slang.D 3DGS renderer (Phase 3) and real SLAT
(Phase 4); until then this trains the latent-only ablation baseline on synthetic SLAT, which is the
comparison arm the headline result is measured against.

Usage: python scripts/phase5_tokenizer_train.py [--steps 200] [--batch 2] [--objects 16]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

from trellis_slat_fsq.train import TokenizerTrainConfig, train_tokenizer

CKPT = os.path.join(os.path.dirname(__file__), "..", "tokenizer_latent_only.pt")


def batches(objects: torch.Tensor, batch: int, steps: int):
    """Cycle a fixed synthetic corpus, `batch` objects per step."""
    n = objects.shape[0]
    for step in range(steps):
        idx = torch.arange(step * batch, (step + 1) * batch) % n
        yield objects[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--objects", type=int, default=16)
    ap.add_argument("--data", default=None, help="dir of real TRELLIS SLATs (scripts/make_real_slats.py)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required — CPU is banned")
    print(f"device: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(0)
    if args.data:
        from trellis_slat_fsq.data import load_real_slats

        records = load_real_slats(args.data)
        corpus = torch.stack([r["slat"] for r in records])
        print(f"corpus: {corpus.shape[0]} REAL TRELLIS SLATs from {args.data}")
    else:
        corpus = torch.randn(args.objects, 8, 64, 64, 64)
        print(f"corpus: {args.objects} synthetic (noise) SLATs — pass --data for real ones")

    losses: list[float] = []
    cfg = TokenizerTrainConfig(ablation="latent_only", steps=args.steps)

    t0 = time.time()
    tok, last = train_tokenizer(
        batches(corpus, args.batch, args.steps), cfg, device="cuda", on_step=losses.append
    )
    dt = time.time() - t0

    torch.save(tok.state_dict(), CKPT)
    summary = {
        "arm": "latent_only",
        "steps": args.steps,
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_min": min(losses),
        "sec_per_step": round(dt / args.steps, 3),
        "final_breakdown": {k: float(v) for k, v in last.items()},
        "checkpoint": os.path.basename(CKPT),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
