# Smoke vs assertions

Would a contract that only checks "the call does not raise" have caught the same regressions as one that compares outputs with reviewed baselines? Every Transformers regression a contract has caught so far, sorted by how it showed up. Sources: [evidence](evidence.md), [improving contracts](improving-contracts.md), [`cases.json`](../experiments/cases.json).

## TL;DR

- **18 regressions caught, on a fixture or a real checkpoint: 6–7 raised, 11 returned a wrong value with no error.** A smoke test catches about a third.
- **Untargeted catches only** (no call written knowing the bug): 3 raised (#48981, #49456, #49403 in part), 2 were silent (#49135, #49385). Smoke-only loses close to half, including the five-month backfill's only frozen-contract catch, UMT5.
- **Baselines cost little noise:** one false alarm in five months of `main` (Wav2Vec2's RNG stream), plus environment drift that the lock's pins already absorb.
- **Middle tier:** baseline-free property checks (causality, round-trip, non-empty config, finite gradients, conditioning matters) reach about 11–12 of 18. The rest need a recorded value, several on the real checkpoint only.
- **Either way, a contract catches only what it calls:** all four backfill misses were missing calls, not missing assertions ([backfill](evidence.md#shadow-backfill)).

## Caught by a smoke test: the call raises

| Regression | Signal | Targeted? |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | crash | post-hoc |
| Whisper batch `decode_with_timestamps` ([#47997](https://github.com/huggingface/transformers/pull/47997)) | crash | post-hoc |
| Whisper `max_length=1` ([#46795](https://github.com/huggingface/transformers/pull/46795)) | raises | post-hoc |
| Whisper `WhisperForCausalLM` assistant ([#48000](https://github.com/huggingface/transformers/pull/48000)) | `UnboundLocalError` | post-hoc |
| Whisper assisted `generate` ([#48981](https://github.com/huggingface/transformers/pull/48981) → [#49075](https://github.com/huggingface/transformers/pull/49075)) | `RuntimeError`, 41 commits | **untargeted**, at its introducing commit |
| Whisper `generate()` on original checkpoints ([#49456](https://github.com/huggingface/transformers/issues/49456), from #48282) | raises | **untargeted**, the night it merged |
| Whisper batch decode of a tensor ([#49403](https://github.com/huggingface/transformers/issues/49403)) | merges rows **or** raises | **untargeted**; smoke catches only the raising shape |

## Needs an assertion: wrong value, no error

| Regression | Signal | Baseline-free check that would see it |
| --- | --- | --- |
| UMT5 decoder not causal under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135), from #47014) | logits off 1.22, loss 0.20; **untargeted**, at its introducing commit | causality: future-token leak 2.9–15 vs 2e-6 |
| BigBird/RemBERT/RoFormer causal LM ([#48785](https://github.com/huggingface/transformers/pull/48785)) | leak 16.8 vs 5e-5 | causality |
| T5 batch `decode` ([#47793](https://github.com/huggingface/transformers/pull/47793)) | spaces before punctuation | decode round-trip |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | return `{}` | non-empty config |
| `masked_spec_embed` uninitialized ([#49385](https://github.com/huggingface/transformers/issues/49385)); **untargeted** | loss moves; gradients non-finite on m8i, finite on M5 | finite gradients, machine-dependent |
| MusicGen Melody conditioning drop ([#48679](https://github.com/huggingface/transformers/pull/48679)) | generated codes | dropping conditioning must change output |
| `additional_special_tokens` lost ([#47848](https://github.com/huggingface/transformers/pull/47848)) | token ids; real checkpoint only | — |
| OLMo stray EOS ([#47988](https://github.com/huggingface/transformers/pull/47988)) | token ids; real checkpoint only | — |
| Whisper left-padded positions ([#48028](https://github.com/huggingface/transformers/pull/48028)) | generated tokens | — |
| Whisper assisted suppression ([#48108](https://github.com/huggingface/transformers/pull/48108)) | extra `Thank you.`; real checkpoint only | — (greedy vs assisted equality, on speech) |
| Whisper `max_length` miscount ([#47150](https://github.com/huggingface/transformers/pull/47150)) | token count | length check |

Outside Transformers: Transformers.js on WebGPU returned all-zero fp16/q4f16 logits with no error ([repro](../experiments/webgpu_browser/README.md)); only a golden output saw it.

## What baselines cost

- **False alarms:** one in 1,468 commits, Wav2Vec2's training loss following a load-time RNG shift ([#46259](https://github.com/huggingface/transformers/pull/46259)); one baseline update. A smoke test would not have fired.
- **Environment drift:** 42 numeric failures in the [GPU dependency study](evidence.md#gpu-dependency-upgrades) (NumPy 2 normalization, the CUDA 13 bundle); reproducible, not noise. The lock's pins keep them out of routine runs.
- **Upkeep:** baselines per platform and dtype, CPU references at four threads, a human review for every changed value ([design](design.md)).

## Limits

- Small sample; 11 of 18 calls were written knowing the bug, which inflates both columns equally but says nothing about the unknown-bug rate.
- "Baseline-free check" is an assessment of what would see each bug, not a measured run. #49135 and #48785 were measured with a leak probe; the others were not.
- Smoke tests still need fixtures and the same call coverage; they remove baseline upkeep, not contract writing.
