# Model integration contracts — evidence so far

Companion to the [proposal](../README.md). This document collects what the local experiments and the historical regression replay show. The synthetic-fault experiments behind the fixture design are detailed in [Small fixtures](fixtures.md); mechanics are in the [technical design](design.md).

## Summary

- **Small fixtures detect what matters, if the checks are numerical.** On synthetic faults across six model families, frozen random fixtures with recorded logits, loss, and gradient checks caught 120 of 126 fault cases; a passing forward pass caught none. The misses were one model (Whisper) whose random fixture ignores its inputs; overfitting that fixture fixed it.
- **The contract must call what users call.** Replaying eleven candidate Transformers regressions (eight reproducible), generic forward/generate checks missed every processor and tokenizer bug; the same small fixtures called through the affected public API caught them.
- **No false alarms so far.** Fixtures frozen on today's `main` stayed clean across ten benign changes from the past two months.

These are small local experiments, not fleet-level evidence: ten model families, synthetic faults, eleven candidate regressions (eight reproducible), ten benign changes, CPU only. The full case list, with commits and verdicts, is in [cases.json](../experiments/cases.json).

## Historical regression replay

To test real bugs rather than synthetic faults, I replayed documented Transformers regressions fixed since May 2026 ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)).

Method:

1. Freeze random and overfit slim fixtures, plus real-checkpoint baselines, on a known-good build (`main` at `4d536e1f`).
2. Extract the bad build (the fix commit's parent) and the fixed build.
3. Run the frozen checks on both builds without regenerating anything. A case counts as caught when the bad build is flagged and the fixed build is not.
4. Separately, run the affected public call on the real checkpoint and on the slim fixture under both builds.

Builds before 2026-08-17 need `tokenizers` 0.22.x (and builds before mid-June an older `kernels`), so those ran with a dependency overlay. Old builds cannot always run in today's environment; the fleet lock needs to pin the dependency environment, not just model revisions.

| Regression (fix PR) | Real checkpoint | Generic slim checks | Slim fixture, affected public call |
| --- | --- | --- | --- |
| CLIP processor drops `return_tensors` when `text_kwargs` is passed ([#46195](https://github.com/huggingface/transformers/pull/46195)) | crash | missed | crash, caught |
| Whisper batch `decode_with_timestamps` ([#47997](https://github.com/huggingface/transformers/pull/47997)) | crash | missed | crash, caught |
| T5 batch `decode` ignores `clean_up_tokenization_spaces` ([#47793](https://github.com/huggingface/transformers/pull/47793)) | caught | missed | caught only with punctuation in the inputs |
| `additional_special_tokens` lost with legacy tokenizer config ([#47848](https://github.com/huggingface/transformers/pull/47848)) | caught | missed | missed: re-saved tokenizer hides it |
| OLMo tokenizer appends a stray EOS ([#47988](https://github.com/huggingface/transformers/pull/47988)) | caught | missed: re-saved tokenizer hides it | — |
| UMT5 decoder self-attention not causal under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135)) | caught under SDPA only | missed under eager; caught under SDPA on all 6 fixtures (logit delta 1.1–15) | caught under SDPA: future-token leak 2.9–15 vs 2e-6 |
| Processor config helpers return `{}` for `preprocessor_config.json`-only repos ([#47628](https://github.com/huggingface/transformers/pull/47628)) | caught | — | caught with copied processor files; missed with re-saved files |
| BigBird/RemBERT/RoFormer causal LM attends to future tokens ([#48785](https://github.com/huggingface/transformers/pull/48785)) | caught: future-token leak 16.8 vs 5e-5 | — | caught: leak 0.0098 vs 2e-7, logits differ by ~2e-3 |
| T5 tied-weights order ([#48238](https://github.com/huggingface/transformers/pull/48238)) | not reproduced with `t5-small` | clean | — |
| MMS `set_target_lang` leaves a stale added-token entry ([#47440](https://github.com/huggingface/transformers/pull/47440)) | not reproduced: internal state only, identical ids and text | — | not reproduced |
| CLIP conversion mapping applied at the wrong nesting level ([#46406](https://github.com/huggingface/transformers/pull/46406)) | not reproduced with `CLIPTextModelWithProjection` | — | not reproduced |

The pattern is consistent:

- **Generic checks are not enough.** A forward pass plus greedy generate on frozen inputs caught only the model-level regression (UMT5), and only when run under the default attention implementation. The others live in processors, tokenizers, and decoding helpers.
- **The usage path catches them, and the small fixture is enough for it.** Whenever the contract made the affected call, the slim fixture caught the bug just as the real checkpoint did. None of these needed real weights.
- **Fixture packaging matters.** Three loading regressions were visible on the real checkpoint but not on a fixture whose processor files had been re-saved by a fixed build. For the processor-config case, a fixture with the original files copied unchanged caught it on all three families tested.
- **Runtime defaults matter.** The UMT5 bug only exists under SDPA, the default. Eager-pinned checks missed it on both the real checkpoint and the fixtures; SDPA checks caught it everywhere.
- **Switching attention after loading is not reliable.** On current `main`, `set_attn_implementation("sdpa")` after loading a T5-family model updates the top-level config but leaves the encoder and decoder on eager, silently ([#49372](https://github.com/huggingface/transformers/issues/49372)). My first SDPA pass used it and never exercised SDPA for T5/UMT5. Contracts should choose the attention implementation at load time and record the one actually in use.

To measure false alarms, I replayed 10 benign changes from the same window that touched the same code paths (RoPE, greedy decoding, image processing, masking, attention refactors, auto-generated inits). Across 20 builds and 42 frozen checks per build, there were **no false alarms** with eager attention. Rechecking all 20 builds under SDPA, chosen at load time, also produced no false alarms, including the change that moved Wav2Vec2 to SDPA dispatch and an attention refactor for continuous batching. (The May 2026 build does not support SDPA for T5, so that variant fails on both bad and fixed builds; it counts as unsupported, not as a detection.)

The candidate search also showed that documented regressions inside a given model family are relatively rare in a five-month window; most bugs that matter to these models are in shared processor, tokenizer, and attention code. That supports organizing contracts around public usage paths rather than per-model forward checks.

This is a small sample: eleven candidate regressions (three not reproducible on standard checkpoints), ten benign changes, CPU only, and no GPU-specific case.
