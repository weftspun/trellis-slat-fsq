"""Unified decoder-only LM over [text | image | SLAT-residual-FSQ], backbone = Qwen/Qwen3.5-0.8B.

Replaces Kyvo's Llama-3.2-1B (torchtune) with Qwen3.5-0.8B via HF transformers + peft/LoRA — the minimal
viable dense model in the Qwen3.5 family (0.8B), close to the paper's 1B. The Qwen3.6 family was rejected:
it has no small model (smallest 27B / 35B-A3B), which would far exceed Kyvo's backbone.

The vocabulary is extended with the SLAT residual-FSQ codes and the modality boundary tokens; images enter
as VQGAN tokens (Kyvo scheme), so no vision encoder is used. Sequence layout follows Kyvo:
`BOS ... [modality blocks] ... OUTSEP <target> EOS`, 3D tokens in raw grid order (no reorder).

See decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field

BACKBONE = "Qwen/Qwen3.5-0.8B"


@dataclass
class SpecialTokens:
    """Kyvo-style modality boundaries (ids assigned in the extended vocab region)."""
    bos: int
    eos: int
    outsep: int
    boimg: int
    eoimg: int
    bo3d: int
    eo3d: int


@dataclass
class UnifiedVocab:
    """Layout of the extended vocabulary: [ base text | image VQGAN | SLAT residual-FSQ | specials ]."""
    text_vocab_size: int
    image_codebook_size: int = 8192          # Kyvo image VQGAN
    slat_codebook_size: int = 8192           # residual-FSQ per-stage ([8,8,8,16])
    n_special: int = 7

    @property
    def image_offset(self) -> int:
        return self.text_vocab_size

    @property
    def slat_offset(self) -> int:
        return self.text_vocab_size + self.image_codebook_size

    @property
    def special_offset(self) -> int:
        return self.slat_offset + self.slat_codebook_size

    @property
    def total(self) -> int:
        return self.special_offset + self.n_special

    def specials(self) -> SpecialTokens:
        b = self.special_offset
        return SpecialTokens(bos=b, eos=b + 1, outsep=b + 2, boimg=b + 3, eoimg=b + 4, bo3d=b + 5, eo3d=b + 6)


def assemble_sequence(vocab: UnifiedVocab, *, text=None, image_tokens=None, slat_tokens=None,
                      target_slat_tokens=None) -> list[int]:
    """Build one training/inference sequence, offsetting each modality into its vocab region.

    Mirrors Kyvo's `BOS + <inputs> + OUTSEP + <target> + EOS` (raw 3D order). Any modality may be omitted.
    """
    sp = vocab.specials()
    seq = [sp.bos]
    if image_tokens is not None:
        seq += [sp.boimg, *(vocab.image_offset + int(t) for t in image_tokens), sp.eoimg]
    if slat_tokens is not None:
        seq += [sp.bo3d, *(vocab.slat_offset + int(t) for t in slat_tokens), sp.eo3d]
    if text is not None:
        seq += [int(t) for t in text]  # text ids are already in the base region
    if target_slat_tokens is not None:
        seq += [sp.outsep, sp.bo3d, *(vocab.slat_offset + int(t) for t in target_slat_tokens), sp.eo3d]
    seq.append(sp.eos)
    return seq


def build_lm(vocab: UnifiedVocab, lora: bool = True, backbone: str = BACKBONE):
    """Load Qwen3.5-0.8B, resize embeddings to `vocab.total`, and (optionally) wrap with LoRA.

    Requires the `train` extra (transformers + peft). Kept import-deferred so the package imports without it.
    """
    from transformers import AutoModelForCausalLM  # deferred heavy dep

    model = AutoModelForCausalLM.from_pretrained(backbone)
    model.resize_token_embeddings(vocab.total)
    if lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return model
