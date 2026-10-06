# Small fixtures for model integration contracts

Companion to the [model integration contracts proposal](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts-md). This document covers how small fixtures are built, what we know about frozen random versus overfit fixtures, and how this relates to the current tiny-model work in Transformers.

## What a small fixture is

The small fixture should preserve enough of the real model structure to exercise its integration path, without requiring the full checkpoint.

There are two useful starting points.

### 1. Frozen random fixture

Start from a reduced model configuration, initialize it deterministically, freeze the resulting weights and inputs, and record numerical outputs.

This is cheap and does not require training the fixture.

A random fixture is only useful if its outputs actually depend on its inputs. When the fixture is built, we should perturb or swap its inputs and require the recorded logits to move well beyond tolerance. Some tiny random models wash out their inputs (see the Whisper result below); such a fixture cannot catch preprocessing or label bugs, whatever we assert.

### 2. Overfit fixture

Optionally train the reduced model on a few examples so that it produces human-readable expected outputs.

This is useful when exact behavior adds coverage, but it should not be a requirement unless we can show that it catches bugs the numerical checks miss. So far that only happened for fixtures that failed the input-sensitivity check above, so overfitting is the fallback for those fixtures rather than the default.

The fixture still needs to preserve processor-coupled dimensions and model-specific configuration where those affect behavior: vocabulary, special tokens, image sizes, patch sizes, audio dimensions, RoPE settings, MoE structure, and so on.

Not every model will be shrinkable with a generic recipe. A failed fixture build should be visible as missing coverage, not silently skipped.

Only the weights and model config should be shrunk. Tokenizer, processor, and preprocessing files should be copied from the real model byte for byte, not loaded and re-saved. Re-saving with the fixture builder's framework version silently normalizes legacy configs, and in the [historical regression replay](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts-md) that hid two real tokenizer-loading regressions from the slim fixture while the real checkpoint still showed them.

Copying costs almost nothing. Across eight families, the original processor files were within about ±1.5 MB of the re-saved ones (UMT5: +4.5 MB on an 87 MB fixture), and loaded as fast or faster locally. The real cost is that fixtures keep the full vocabulary, so processor files are 20–50% of each fixture; that is needed to keep the real tokenizer and is independent of copying.

## Current fixture work

There is already related work in Transformers around generating small random models from test configurations and real processors.

That is a useful first stage because it gives us broad smoke coverage cheaply.

Today's stage 1 (`utils/create_dummy_models.py`, [#49103](https://github.com/huggingface/transformers/pull/49103)) gives random tiny models built from unit-test configs with the real vocabulary, used as smoke tests. To serve as the frozen fixtures described above, it would additionally need: a fixed seed and pinned fixture revisions, recorded baselines instead of shape-only assertions, the input-sensitivity check, and processor files copied unchanged rather than re-saved and resized.

A second stage would start from the actual model configuration and processor, apply architecture-specific shrink rules, freeze the resulting fixture, check that it is sensitive to its inputs, and overfit it only when it is not.

Starting from the real config should preserve more integration-relevant behavior, but it will not automatically work for every architecture. Nested configs, coupled dimensions, MoE models, and processor assumptions will need explicit handling.

## Local experiment

I ran a small experiment comparing frozen random fixtures with overfit fixtures ([script](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_experiment-py), [raw results](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_results-json)).

The setup used reduced SmolLM and CLIP models, three seeds, and 14 synthetic faults, giving 42 fault/seed cases per policy.

Detection was:

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/42 | 0/42 |
| Recorded discrete outputs | 19/42 | 18/42 |
| Recorded logits | 33/42 | 33/42 |
| Logits + targeted gradients/update | 39/42 | 39/42 |
| Above + recorded loss | **42/42** | **42/42** |

The important result is not that the fixture caught every synthetic fault. These were handpicked faults.

The useful result is that overfitting did not add detection beyond the numerical and training checks in this experiment.

In fact, exact overfit answers still passed for some faults that disabled RoPE, removed masks, or changed pixel scaling. The numerical outputs caught those cases.

Targeted gradient checks caught disconnected paths, and recording loss caught shuffled labels even when gradients remained finite and parameters still updated.

There were no false alarms in 60 control runs covering repetition, save/reload, and eager-to-SDPA execution.

The fixtures were also much smaller than the full checkpoints, but these local timings should not be treated as fleet-level speedup estimates. They exclude installation, loading, downloads, CI startup, fixture maintenance, and several other real costs.

### More model families

I then ran the same script, checks, and tolerances on four more families from the `hf-tiny-v2` fast-integration PR list ([#49103](https://github.com/huggingface/transformers/pull/49103)): BERT (masked LM), T5 (encoder-decoder), Whisper (speech seq2seq), and Wav2Vec2 (CTC). That gave 28 faults and 84 fault/seed cases per policy ([raw results](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_results_more-json)).

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/84 | 0/84 |
| Recorded discrete outputs | 30/84 | 24/84 |
| Recorded logits | 57/84 | 60/84 |
| Logits + targeted gradients/update | 69/84 | 72/84 |
| Above + recorded loss | **78/84** | **84/84** |

BERT, T5, and Wav2Vec2 behaved like SmolLM and CLIP: both policies caught everything.

Whisper is the exception. The tiny random Whisper decoder barely depends on its audio, so reversing the audio moved logits by about 5e-5 (below the 1e-4 tolerance), and shuffling labels across clips left the loss unchanged. It missed both faults on every seed. The overfit Whisper had to tell the clips apart, so the same faults moved logits by 1–24 and loss from about 0.01 to 4–7.

Exact answers were still weak on their own: the overfit fixtures' answers only caught 1–3 faults per fixture. Overfitting helped by making the numbers depend on the input, not by making answers checkable.

Overfitting cost under 4 s per fixture for BERT, T5, and Wav2Vec2, and 23–51 s for Whisper. One Wav2Vec2 seed did not converge in 500 steps, which is the kind of fixture maintenance the pilot needs to measure. There were no false alarms in 120 control runs.

So the current default should be:

**Use frozen random fixtures with recorded numerical outputs, loss, and targeted training checks, gated by an input-sensitivity check. Overfit only the fixtures that fail that check.**

This experiment shows local sensitivity. It does not establish historical bug coverage or maintenance savings.

