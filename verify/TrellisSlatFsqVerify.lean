/-
# Proof obligations for the SLAT -> Residual-FSQ tokenizer

## Original task

Reproduce Kyvo — Sahoo, Tibrewal, Gkioxari, *"Aligning Text, Images, and 3D Structure
Token-by-Token"* (arXiv:2506.08002) — replacing its 3D VQ-VAE's learned 8192-entry VQ codebook with
**Residual FSQ**: TRELLIS.2 SLAT (64^3 x 8) -> encoder -> quantize -> **~512 reconstructive tokens per
object** (8^3 positions; token generation for avatars / worlds / props), where the coarse residual
prefix doubles as the retrieval/identification ID. Decision records:

* ../decisions/20260713-generation-slat-fsq-render-auxloss.md   (generation channel, render aux-loss)
* ../decisions/20260713-port-kyvo-residual-fsq.md               (Residual FSQ; [8,8,8,16] = exactly 8192)
* ../decisions/20260713-reproduce-kyvo-full-method-residual-fsq.md (full-method reproduction)

These proofs underwrite the load-bearing discrete claim of that task: the FSQ level set [8, 8, 8, 16]
gives **per-code parity with Kyvo's 8192-entry VQ** — a codebook of exactly 8192 codes with no
collisions and no unreachable entries — and the residual stage streams inherit that bijectivity at
every depth, so the ~512-token stream is reconstruction-faithful at the discrete layer and the
coarse-prefix ID is well-defined. Kernel under proof: ../trellis_slat_fsq/fsq.slang (slangtorch/CUDA)
and its plain-buffer deploy variant fsq_nif.slang; the Lean `index` models their shared mixed-radix
fold.

## Assurance layers

1. **Kernel-checked proofs** — the FSQ index map (mixed-radix fold over levels [8, 8, 8, 16], basis
   [1, 8, 64, 512], basis derived not asserted) is a bijection onto [0, 8192), with an explicit
   inverse (`decode`) and `omega`; residual stage streams are componentwise bijective; no axioms,
   no `sorry`.
2. **Witness-DAG certification** — `PlausibleWitnessDag.resolve` (fire/plausible-witness-dag) searches
   for a round-trip violation witness (`decode (index c) ≠ c`) across an escalation ladder whose final
   rung sweeps the ENTIRE codebook; the build fails unless the outcome is `provablyNone`.
3. **Exhaustive-on-hardware twin** — ../tests/test_tokenizer.py drives all 8192 code tuples through
   the actual Slang CUDA kernel on the GPU and asserts the same bijection (each index hit exactly
   once). Float bound/round behavior between those points and stage-to-stage residual peeling remain
   empirical by nature.
-/

import PlausibleWitnessDag

namespace TrellisSlatFsqVerify

/-- FSQ levels [8, 8, 8, 16]; product is exactly 8192 = 2^13 (per-code parity with Kyvo's VQ). -/
def levels : List Nat := [8, 8, 8, 16]

/-- Mixed-radix basis: exclusive cumulative product of levels — [1, 8, 64, 512]. -/
def basis : List Nat := [1, 8, 64, 512]

/-- Codebook size: 8*8*8*16 = 8192. -/
def codebookSize : Nat := 8192

theorem levels_prod : levels.foldl (· * ·) 1 = codebookSize := by decide

/-- The mixed-radix basis is DERIVED from the levels (exclusive scan of products), not asserted. -/
theorem basis_is_levels_scan :
    basis = (levels.take 3).foldl (fun acc l => acc ++ [acc.getLast! * l]) [1] := by decide

/-- Token budget: 8^3 spatial positions = 512 tokens/object (one code per position per stage). -/
theorem positions_count : 8 * 8 * 8 = 512 := by decide

/-- Per-dim codes for one token. -/
structure Codes where
  c0 : Fin 8
  c1 : Fin 8
  c2 : Fin 8
  c3 : Fin 16
  deriving DecidableEq, Repr

/-- The FSQ index map (same fold as fsq.slang / fsq.ex). -/
def index (c : Codes) : Nat :=
  c.c0.val * 1 + c.c1.val * 8 + c.c2.val * 64 + c.c3.val * 512

/-- Range soundness: every code tuple maps into [0, 8192). -/
theorem index_lt (c : Codes) : index c < codebookSize := by
  obtain ⟨⟨v0, h0⟩, ⟨v1, h1⟩, ⟨v2, h2⟩, ⟨v3, h3⟩⟩ := c
  simp only [index, codebookSize]
  omega

/-- Explicit inverse: peel the mixed-radix digits back off. Total (top digit mod 16), so no
dependent proof argument gets in the way of rewriting. -/
def decode (i : Nat) : Codes where
  c0 := ⟨i % 8, Nat.mod_lt _ (by omega)⟩
  c1 := ⟨i / 8 % 8, Nat.mod_lt _ (by omega)⟩
  c2 := ⟨i / 64 % 8, Nat.mod_lt _ (by omega)⟩
  c3 := ⟨i / 512 % 16, Nat.mod_lt _ (by omega)⟩

/-- Right inverse: index (decode i) = i on [0, 8192) — the map is SURJECTIVE onto the codebook. -/
theorem index_decode (i : Nat) (h : i < codebookSize) : index (decode i) = i := by
  simp only [index, decode, codebookSize] at *
  omega

/-- Left inverse: decode (index c) = c — the map is INJECTIVE (no codebook collisions). -/
theorem decode_index (c : Codes) : decode (index c) = c := by
  obtain ⟨⟨v0, h0⟩, ⟨v1, h1⟩, ⟨v2, h2⟩, ⟨v3, h3⟩⟩ := c
  simp only [decode, index, Codes.mk.injEq, Fin.mk.injEq]
  refine ⟨?_, ?_, ?_, ?_⟩ <;> omega

/-- Bijectivity, stated as injectivity (immediate from the two-sided inverse). -/
theorem index_inj (a b : Codes) (h : index a = index b) : a = b := by
  rw [← decode_index a, ← decode_index b, h]

/-- Surjectivity onto the codebook, stated explicitly. -/
theorem index_surj (y : Nat) (h : y < codebookSize) : ∃ c : Codes, index c = y :=
  ⟨decode y, index_decode y h⟩

/-- Capstone: every index in [0, 8192) has EXACTLY ONE code tuple — the "8192-entry codebook" claim
in one statement (existence = surjectivity, uniqueness = injectivity / no collisions).
Stated without `∃!` (Mathlib notation; this package is core-only). -/
theorem index_exists_unique (y : Nat) (h : y < codebookSize) :
    ∃ c : Codes, index c = y ∧ ∀ c' : Codes, index c' = y → c' = c :=
  ⟨decode y, index_decode y h, fun c' hc' => by rw [← decode_index c', hc']⟩

/-! ## Residual FSQ stage streams

Residual FSQ emits one code tuple per stage; a position's token stream is a function
`Fin Q → Codes` and its index stream is the componentwise `index`. The float residual-peeling
between stages is empirical (GPU-tested); what is provable — and proven here — is that the
DISCRETE stream layer inherits bijectivity componentwise: distinct streams never collide, every
in-range index stream is realized, and the coarse ID prefix is a projection of an injective map. -/

/-- Componentwise index of a Q-stage code stream. -/
def stageIndices {Q : Nat} (f : Fin Q → Codes) : Fin Q → Nat :=
  fun k => index (f k)

/-- Range: every stage of a stream indexes into [0, 8192). -/
theorem stageIndices_lt {Q : Nat} (f : Fin Q → Codes) (k : Fin Q) :
    stageIndices f k < codebookSize :=
  index_lt (f k)

/-- Injectivity: two streams with the same index stream are equal (no collisions at any depth —
this is what makes the full ~512-token stream reconstruction-faithful at the discrete layer). -/
theorem stageIndices_inj {Q : Nat} (a b : Fin Q → Codes)
    (h : ∀ k, stageIndices a k = stageIndices b k) : a = b :=
  funext fun k => index_inj _ _ (h k)

/-- Surjectivity: every in-range index stream is realized by some code stream (capacity is exactly
8192^Q — nothing in the stream space is wasted). -/
theorem stageIndices_surj {Q : Nat} (g : Fin Q → Nat) (h : ∀ k, g k < codebookSize) :
    ∃ f : Fin Q → Codes, ∀ k, stageIndices f k = g k :=
  ⟨fun k => decode (g k), fun k => index_decode (g k) (h k)⟩

/-- The coarse retrieval-ID prefix (first `p` stages) is literally a restriction of the stream —
so equal streams have equal IDs by construction, and ID lossiness is exactly stream truncation. -/
theorem idPrefix_is_restriction {Q p : Nat} (hp : p ≤ Q) (f : Fin Q → Codes) (k : Fin p) :
    stageIndices (fun j : Fin p => f (j.castLE hp)) k = stageIndices f (k.castLE hp) :=
  rfl

/-! ## Witness-DAG certification (plausible-witness-dag)

Searches for a round-trip violation witness inside the driver's Fin windows. `candidateIsWitness w`
decodes `w`, re-encodes, and flags a witness iff the round trip breaks — which the proofs above show
is impossible, so the expected outcome is `provablyNone`. -/

/-- Boolean round-trip check used as the plausible candidate predicate. -/
def roundTripBroken (w : Nat) : Bool :=
  if w < codebookSize then index (decode w) != w else false

/-- Deterministic read-back: scan candidates up to the walk budget. -/
def readback (walkSteps : Nat) : PlausibleWitnessDag.Readback Nat :=
  let bound := min walkSteps codebookSize
  match (List.range bound).find? roundTripBroken with
  | some w => { value := w, found := true, witnessIdx := w, budgetHit := false }
  | none => { value := 0, found := false, budgetHit := bound < codebookSize }

/-- Escalation ladder: the default rungs plus a final rung whose walk budget covers the ENTIRE
8192-entry codebook, so the deterministic read-back is exhaustive and a missing witness resolves to
`provablyNone` rather than `budgetHit`. -/
def fullLadder : Array PlausibleWitnessDag.Level :=
  PlausibleWitnessDag.ladder.push
    { idx := 3, walkSteps := codebookSize, finBound := 4096, numInst := 2000 }

def runCertification : IO Unit := do
  let (_, lvl, trace) ← PlausibleWitnessDag.resolve
    "fsq-index-roundtrip-violation"
    (fun _ w => roundTripBroken w)
    readback
    fullLadder
  let ok := trace.outcome == .provablyNone
  IO.println s!"witness-dag: level={lvl} outcome={repr trace.outcome} exhaustive={ok}"
  unless ok do
    throw <| IO.userError "witness-dag certification did not resolve to provablyNone"

#eval runCertification

end TrellisSlatFsqVerify
