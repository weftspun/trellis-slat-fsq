---
title: Pause elixir-recgpt-fsq (elixir-sequential-recommendation) — recommendation folds into the dual-use SLAT→FSQ gen+ID line
date: 2026-07-13
status: accepted — repo PAUSED and archived (read-only); superseded by unified bidirectional generation+identification
tier: baseline
decision-makers: K. S. Ernest (iFire) Lee
---

## Context and Problem Statement

`elixir-recgpt-fsq` (org repo `weftspun/elixir-sequential-recommendation`) is a RecGPT-style sequential
recommender in Elixir/Nx: FSQ item tokens → FuXi-Linear sequence model → next-item generative retrieval.
It was the downstream **consumer** of semantic IDs — the ranking half of a two-stack design (standalone
retrieval repos `slat-semantic-ids` and `multimodal-semantic-ids` produced the IDs).

That two-stack split no longer holds. The identification role collapses into the generation codebook (see
`20260713-generation-slat-fsq-render-auxloss.md`): one hierarchical/residual FSQ over TRELLIS.2 SLAT yields
**~512 reconstructive tokens for generation** and a **coarse ~3-code prefix that IS the retrieval /
identification ID**. Bidirectional *generation + identification* is then one Kyvo-style decoder-only model
(arXiv:2506.08002) over a unified token space `[text | image | SLAT-FSQ]`, not a separate recommender stack.
A standalone Elixir sequential recommender is redundant under that architecture.

## Decision

**Pause `elixir-recgpt-fsq`.** No further feature work; the repo is archived (read-only) on GitHub and kept
as reference. Recommendation/identification is served by the coarse FSQ prefix of the dual-use SLAT→FSQ
codebook; ranking becomes a consumer of those codes inside the unified backbone rather than a distinct repo.

## Settled choices

- **~3 codes for retrieval / identification** = the coarse prefix of the same FSQ used for generation; the
  full ~512-token stream stays reconstruction-only and is never placed in a retrieval context window.
- **One model, both directions.** Generation and identification share a codebook and token space (Kyvo-style
  unified decoder-only LM), so a separate sequential-recommender repo is not carried forward.
- **Archived, not deleted.** Pausing is reversible: unarchive if a standalone recommender is ever needed
  again. Archived alongside the retrieval-only repos `slat-semantic-ids` and `multimodal-semantic-ids`.

## Scope boundary

This repo (`trellis-slat-fsq`) owns **generation** and is the surviving canonical home for the SLAT→FSQ line,
so this cross-repo pause is recorded here. The paused `elixir-recgpt-fsq` owned the Elixir sequential
recommender (RecGPT / FuXi-Linear). Its last work — a hexagonal core/ports/adapters restructure, an
EXLA→Torchx backend swap, and an MCP server — is preserved in the archived repo and is **not** carried
forward unless the unified backbone chooses to reuse it.

## Consequences

- One fewer active stack; retrieval logic folds into the unified generation+identification backbone.
- Elixir/Nx assets (FSQ, FuXi-Linear inference, MCP serving) remain available for reference in the archive.
- Reversible via unarchive; this MADR is the pointer explaining why the repo is quiet.

## Verification

`weftspun/elixir-sequential-recommendation` shows **Archived** on GitHub with no open PRs pending merge; the
identification path is exercised as the coarse FSQ prefix in the SLAT→FSQ line rather than in this repo.
