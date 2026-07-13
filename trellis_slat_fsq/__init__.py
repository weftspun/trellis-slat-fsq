# SPDX-License-Identifier: MIT OR Apache-2.0
# Copyright (c) 2026 K. S. Ernest (iFire) Lee and weftspun contributors
"""trellis-slat-fsq: TRELLIS.2 SLAT -> ~512 reconstructive FSQ tokens for generation.

The GENERATION system: a SLAT->FSQ tokenizer trained with a Kyvo-style multi-view render aux-loss
(training-time only; inference render-free). Its ~512 reconstructive tokens/object decode back to
geometry+PBR to generate avatars/worlds/props. Distinct from the render-free RETRIEVAL system in the
separate `slat-semantic-ids` repo (compact 3-code FSQ semantic ID).

Scaffold only: the differentiable multi-view renderer, TRELLIS.2 decoder wiring, and training loop are
not implemented yet. See decisions/20260713-generation-slat-fsq-render-auxloss.md.
"""

from __future__ import annotations

__all__: list[str] = []
