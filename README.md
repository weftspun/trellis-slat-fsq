# trellis-slat-fsq

TRELLIS.2 SLAT → ~512 **reconstructive** FSQ tokens per object, for **generating** avatars / worlds /
props. The tokenizer is trained with a Kyvo-style multi-view **render aux-loss** (training-time only);
inference is **render-free** (encode SLAT → tokens, no rendering). FSQ (not VQ), consistent with the
standing FSQ-over-VQ decision.

This is the **generation** system. The separate
[`slat-semantic-ids`](../slat-semantic-ids) repo is the **retrieval** system — it turns pooled SLAT into
a compact 3-code FSQ semantic ID and is render-free end to end. The two are decoupled: ~512
reconstructive tokens here never share a context budget with the 3-code retrieval ID there.

Status: **scaffold**. The differentiable multi-view render loss, TRELLIS.2 decoder wiring, and training
loop are not implemented yet — see [`decisions/20260713-generation-slat-fsq-render-auxloss.md`](decisions/20260713-generation-slat-fsq-render-auxloss.md).

Licensed under either of Apache-2.0 ([LICENSE-APACHE](LICENSE-APACHE)) or MIT ([LICENSE-MIT](LICENSE-MIT))
at your option; each source file carries an `SPDX-License-Identifier: MIT OR Apache-2.0` tag.
