"""Phase 6 loop (LM results), end-to-end: tokenize corpus -> train Qwen3.5-0.8B -> task-suite eval.

Runs the COMPLETE pipeline at whatever scale the machine allows. On CPU with synthetic SLAT and an
untrained tokenizer this is a plumbing validation, NOT the scientific result — the same loop pointed at
real SLAT + a render-aux-trained tokenizer + GPU produces the reproduction numbers
(decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md, roadmap Phase 6).

Steps (matching the taskweft plan):
  13. tokenize_corpus_sequences  — SLAT -> ResidualFSQ tokens -> Kyvo-layout sequences -> Parquet (duckdb)
  14. train_qwen_lm              — Qwen3.5-0.8B (local xet snapshot) + extended vocab + LoRA, causal-LM
                                   loss on the target segment only
  15. eval_task_suite            — greedy generation of target 3D tokens; Jaccard + text-accuracy

Usage:  python scripts/phase6_loop.py [--objects 4] [--steps 2] [--gen-tokens 24] [--seq-3d 128]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import duckdb
import torch

from trellis_slat_fsq.eval import jaccard, text_accuracy
from trellis_slat_fsq.lm import UnifiedVocab, assemble_sequence
from trellis_slat_fsq.tokenizer import SlatFsqReconstructiveTokenizer

SNAPSHOT_GLOB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/*"
)


DEVICE = "cuda"  # CPU is banned (user directive); scripts hard-fail without CUDA.


def step13_tokenize(n_objects: int, seq_3d: int, out_parquet: str, data_dir: str | None = None) -> list[dict]:
    """SLAT -> ResidualFSQ tokens (stage-0 prefix of the first seq_3d positions) -> Parquet.

    With `data_dir`, uses REAL TRELLIS SLATs (scripts/make_real_slats.py); else synthetic noise."""
    torch.manual_seed(0)
    tok = SlatFsqReconstructiveTokenizer().to(DEVICE)  # untrained encoder; quantizer is the fixed FSQ grid

    if data_dir:
        from trellis_slat_fsq.data import load_real_slats

        records = load_real_slats(data_dir)[:n_objects]
        slats = [(r["id"], r["slat"].unsqueeze(0).to(DEVICE)) for r in records]
        print(f"[13] using {len(slats)} REAL TRELLIS SLATs from {data_dir}")
    else:
        slats = [(f"syn-{i}", torch.randn(1, 8, 64, 64, 64, device=DEVICE)) for i in range(n_objects)]

    rows = []
    for name, slat in slats:
        with torch.no_grad():
            indices, _codes = tok.encode(slat)  # [1, 8, 8, 8, Q]
        stage0 = indices[0, ..., 0].reshape(-1)[:seq_3d].tolist()  # raw grid order (Kyvo layout)
        rows.append({"id": name, "tokens": stage0})

    con = duckdb.connect()
    con.execute("CREATE TABLE seqs (id VARCHAR, tokens BIGINT[])")
    con.executemany("INSERT INTO seqs VALUES (?, ?)", [(r["id"], r["tokens"]) for r in rows])
    con.execute(f"COPY seqs TO '{out_parquet}' (FORMAT PARQUET)")
    back = con.execute(f"SELECT count(*) FROM read_parquet('{out_parquet}')").fetchone()[0]
    assert back == n_objects
    print(f"[13] tokenized {n_objects} objects -> {out_parquet} ({back} rows, {seq_3d} 3D tokens each)")
    return rows


def load_lm(vocab: UnifiedVocab, snapshot: str):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(snapshot, dtype=torch.bfloat16, device_map=DEVICE)
    hf_tok = AutoTokenizer.from_pretrained(snapshot)
    model.resize_token_embeddings(vocab.total)
    cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    return model.to(DEVICE), hf_tok


def build_example(vocab: UnifiedVocab, hf_tok, tokens: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """`BOS [3d] "reconstruct" OUTSEP [target 3d] EOS`; loss only on the target segment."""
    text_ids = hf_tok("reconstruct the object", add_special_tokens=False)["input_ids"]
    seq = assemble_sequence(vocab, slat_tokens=tokens, text=text_ids, target_slat_tokens=tokens)
    input_ids = torch.tensor([seq], device=DEVICE)
    labels = input_ids.clone()
    outsep_pos = seq.index(vocab.specials().outsep)
    labels[0, : outsep_pos + 1] = -100  # supervise only the target 3D block + EOS
    return input_ids, labels


def step14_train(model, vocab, hf_tok, rows, steps: int, lr: float = 2e-4) -> list[float]:
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        row = rows[step % len(rows)]
        input_ids, labels = build_example(vocab, hf_tok, row["tokens"])
        t0 = time.time()
        out = model(input_ids=input_ids, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad()
        losses.append(out.loss.item())
        print(f"[14] step {step + 1}/{steps} loss={out.loss.item():.4f} ({time.time() - t0:.1f}s, seq={input_ids.shape[1]})")
    return losses


@torch.no_grad()
def step15_eval(model, vocab, hf_tok, rows, gen_tokens: int) -> dict:
    model.eval()
    sp = vocab.specials()
    row = rows[-1]  # held-out-ish: least-trained example
    text_ids = hf_tok("reconstruct the object", add_special_tokens=False)["input_ids"]
    prompt = assemble_sequence(vocab, slat_tokens=row["tokens"], text=text_ids)
    prompt = prompt[:-1] + [sp.outsep, sp.bo3d]  # replace EOS with the answer prefix
    ids = torch.tensor([prompt], device=DEVICE)

    generated = []
    for _ in range(gen_tokens):
        logits = model(input_ids=ids).logits[0, -1]
        next_id = int(logits.argmax())
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]], device=DEVICE)], dim=1)

    target = [t + vocab.slat_offset for t in row["tokens"][:gen_tokens]]
    report = {
        "jaccard": jaccard(torch.tensor(generated), torch.tensor(target)),
        "text_accuracy": text_accuracy(torch.tensor(generated), torch.tensor(target)),
        "in_slat_vocab": sum(vocab.slat_offset <= g < vocab.slat_offset + 8192 for g in generated) / len(generated),
        "n_generated": len(generated),
    }
    print(f"[15] eval: {json.dumps(report, indent=2)}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--objects", type=int, default=4)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--gen-tokens", type=int, default=24)
    ap.add_argument("--seq-3d", type=int, default=128, help="3D tokens per object (512 = full budget)")
    ap.add_argument("--data", default=None, help="dir of real TRELLIS SLATs (scripts/make_real_slats.py)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required — CPU is banned (install torch with --torch-backend=auto)")
    print(f"device: {torch.cuda.get_device_name(0)}")

    snapshots = glob.glob(SNAPSHOT_GLOB)
    if not snapshots:
        sys.exit("no local Qwen3.5-0.8B snapshot; fetch with: uvx --from 'huggingface_hub[hf_xet]' hf download Qwen/Qwen3.5-0.8B")
    snapshot = snapshots[0]

    out_parquet = os.path.join(os.path.dirname(__file__), "..", "phase6_sequences.parquet")
    rows = step13_tokenize(args.objects, args.seq_3d, out_parquet, data_dir=args.data)

    vocab = UnifiedVocab(text_vocab_size=151_936)
    print(f"unified vocab: total={vocab.total} slat_offset={vocab.slat_offset}")
    t0 = time.time()
    model, hf_tok = load_lm(vocab, snapshot)
    print(f"[14] Qwen3.5-0.8B loaded + vocab-extended + LoRA in {time.time() - t0:.1f}s")

    losses = step14_train(model, vocab, hf_tok, rows, args.steps)
    report = step15_eval(model, vocab, hf_tok, rows, args.gen_tokens)

    print(json.dumps({"phase6": "complete", "losses": losses, "eval": report,
                      "scale": "toy corpus on GPU — pipeline validation; reproduction numbers need "
                               "real SLAT + the render-aux-trained tokenizer (Phases 3-5)"}))


if __name__ == "__main__":
    main()
