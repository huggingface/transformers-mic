# Small fixtures for model integration contracts

Companion to the [model integration contracts proposal](../README.md).

PR-level contract checks need a model that is small enough to run on CPU in seconds but still exercises the real integration path. This document describes the project to build those small fixtures, step by step: what already exists in Transformers (stage 1), frozen random fixtures (stage 2), and overfit fixtures as a fallback (stage 3), with the local experiments behind each decision.

## What a fixture must provide

A small fixture is useful for contracts only if it has all of the following:

- **Real structure, small weights.** The real model config, shrunk (layers, width, heads, experts), keeping processor-coupled dimensions and model-specific settings: vocabulary, special tokens, image and patch sizes, audio dimensions, RoPE settings, MoE structure.
- **Real processor files, unchanged.** Tokenizer, processor, and preprocessing files copied from the real model byte for byte.
- **Frozen and pinned.** Deterministic initialization, frozen weights and inputs, published at a pinned revision (for example `hf-internal-testing/<model>-slim@<sha>`).
- **Recorded baselines.** Outputs, selected logits, loss, targeted gradients, and an optimizer update, recorded on a known-good build and reviewed.
- **Sensitive to its inputs.** Perturbing or swapping inputs must move the recorded logits well beyond tolerance.
- **Run like users run it.** Baselines recorded under the default runtime configuration (for example SDPA attention), with eager as an extra variant.

A failed fixture build should be visible as missing coverage, not silently skipped.

## Steps

| Stage | Produces | Status |
| --- | --- | --- |
| 1. Random tiny models from test configs | `tiny-random-*` models for smoke tests | exists in Transformers, being extended |
| 2. Frozen random fixtures from real configs | pinned fixtures with recorded baselines | validated locally on 8 families; not built at scale |
| 3. Overfit fallback | fixtures trained on a few examples | only for fixtures that fail the input-sensitivity check |

## Stage 1: random tiny models from test configs

This is the existing work in Transformers: `utils/create_dummy_models.py`, a nightly `check_tiny_models` workflow, publication to `hf-tiny-v2`, and fast integration tests in [#49103](https://github.com/huggingface/transformers/pull/49103) ([#49331](https://github.com/huggingface/transformers/pull/49331) fixes tiny-model creation for newly added models).

What it does today:

- builds the config from each model's unit-test `ModelTester`, overriding `vocab_size` and special-token ids from the real tokenizer;
- finds a processor from a real checkpoint (best effort), keeps the full vocabulary (tokenizer shrinking is disabled), rebuilds image processors with `size`/`crop_size` set to the tiny config's `image_size`, and re-saves everything;
- initializes the model randomly without a fixed seed;
- in #49103, loads `hf-tiny-v2/tiny-random-*` without a pinned revision and checks that `forward()` and `generate()` run and return non-empty outputs, without asserting values.

This is a useful first stage: broad smoke coverage of processor → model → forward/generate for many architectures, cheaply. It is not yet a frozen fixture:

| Requirement | Stage 1 today |
| --- | --- |
| Real structure | unit-test config, so real settings (RoPE scaling, sliding windows, MoE layout) are not preserved |
| Real processor files unchanged | real vocabulary, but files are re-saved and image preprocessing is resized |
| Frozen and pinned | no seed; tests follow the latest Hub revision |
| Recorded baselines | shape-only assertions |
| Input sensitivity | not checked |

## Stage 2: frozen random fixtures

Recipe:

1. Start from the real model config and copy the real processor files unchanged.
2. Shrink the config with architecture-specific rules. In the experiments: hidden size 64, intermediate size 128, 2 layers, 4 heads; vocabulary, image and patch sizes, and audio dimensions preserved; dropout and SpecAugment disabled so training checks are deterministic.
3. Initialize from a fixed seed and save.
4. Run the input-sensitivity check: perturb or swap inputs and require logits to move well beyond tolerance. If it fails, go to stage 3.
5. Record baselines on a known-good build, under the default runtime configuration: discrete outputs, selected logits, loss, gradients on named modules, and whether an SGD step updates parameters.
6. Publish at a pinned revision and add it to the fleet lock.

Shrinking from the real config preserves more integration-relevant behavior than unit-test configs, but it will not work generically for every architecture. Nested configs, coupled fields (`layer_types`, `head_dim`), MoE models, and processor assumptions need explicit rules, with test configs as a fallback.

### Evidence: synthetic faults on SmolLM and CLIP

Reduced SmolLM-135M and CLIP ViT-B/32, three seeds, 14 synthetic faults (11 inference, 2 disconnected gradient paths, 1 shuffled-label fault): 42 fault/seed cases per policy. Random and overfit fixtures were compared on identical faults, inputs, and assertions ([script](../experiments/experiment.py), [raw results](../experiments/results/smollm_clip.json)).

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/42 | 0/42 |
| Recorded discrete outputs | 19/42 | 18/42 |
| Recorded logits | 33/42 | 33/42 |
| Logits + targeted gradients/update | 39/42 | 39/42 |
| Above + recorded loss | **42/42** | **42/42** |

- A successful forward pass caught nothing; numerical checks did the work.
- Exact overfit answers still passed with RoPE disabled, masks removed, or pixel scale changed. Logits caught those.
- Targeted gradient checks caught detached modules; recorded loss caught shuffled labels even though gradients stayed finite and parameters still updated.
- No false alarms in 60 controls (repeats, save/reload, eager vs SDPA).

| Cost | SmolLM | CLIP |
| --- | --- | --- |
| Parameters, fixture / full | 3.22M / 134.52M | 3.51M / 151.28M |
| Saved fixture with processor | 16.4 MB | 17.7 MB |
| Warm inference, four inputs | 4.0–6.8 ms | 1.8–1.9 ms |
| Full-checkpoint inference | 111 ms | 54 ms |

These timings exclude loading, downloads, CI startup, and upkeep; they are not fleet-level speedup estimates.

### Evidence: four more families

Same script, checks, and tolerances on BERT (masked LM), T5 (encoder-decoder), Whisper (speech seq2seq), and Wav2Vec2 (CTC), from the #49103 model list: 28 faults, 84 fault/seed cases per policy ([raw results](../experiments/results/more_families.json)).

| Checks | Random | Overfit |
| --- | ---: | ---: |
| Successful inference only | 0/84 | 0/84 |
| Recorded discrete outputs | 30/84 | 24/84 |
| Recorded logits | 57/84 | 60/84 |
| Logits + targeted gradients/update | 69/84 | 72/84 |
| Above + recorded loss | **78/84** | **84/84** |

BERT, T5, and Wav2Vec2 behaved like SmolLM and CLIP: random fixtures caught everything. All six random misses were Whisper; see stage 3. No false alarms in 120 controls.

### Evidence: historical regressions

The [historical regression replay](evidence.md) froze random and overfit fixtures for six families (plus UMT5 and OLMo) on a known-good build, then checked them on the bad and fixed builds of seven real regressions and ten benign changes. Three results shaped the stage 2 recipe:

- **Copy processor files; do not re-save them.** Three loading regressions ([#47628](https://github.com/huggingface/transformers/pull/47628) plus the two below) were hidden by re-saved processor files; with the original files copied, the slim fixture caught [#47628](https://github.com/huggingface/transformers/pull/47628) on CLIP, Whisper, and Wav2Vec2. Two tokenizer-loading regressions ([#47848](https://github.com/huggingface/transformers/pull/47848), [#47988](https://github.com/huggingface/transformers/pull/47988)) were visible on the real checkpoint but not on the fixtures, because the fixture builder had re-saved the tokenizer with a fixed build. Copying costs almost nothing: across eight families the original files were within about ±1.5 MB of re-saved ones (UMT5: +4.5 MB on an 87 MB fixture) and loaded as fast or faster. Processor files are 20–50% of each fixture because the full vocabulary is kept; that is inherent to using the real tokenizer.
- **Record baselines under the default runtime configuration.** The UMT5 causal-attention regression ([#49135](https://github.com/huggingface/transformers/pull/49135)) only exists under SDPA. Under SDPA, the frozen checks caught it on all six fixtures (logit delta 1.1–15) and on the real checkpoint; eager-pinned checks missed it everywhere. Choose the attention implementation at load time: on current `main`, `set_attn_implementation()` after loading silently leaves T5-family encoder/decoder stacks on eager ([#49372](https://github.com/huggingface/transformers/issues/49372)).
- **Frozen fixtures are stable.** Fixtures frozen on today's `main` produced no false alarms on 20 builds across ten benign changes from the previous two months, under both eager and SDPA.

## Stage 3: overfit fallback

Recipe: train the stage 2 model on a few contrasting examples (AdamW, learning rate 5e-3, gradient clipping, up to 500 steps, stopping once answers are correct and loss is below 0.05 on two consecutive checks), then record baselines as in stage 2.

Use it only when the random fixture fails the input-sensitivity check. The evidence:

- **Where overfitting helped: Whisper.** The tiny random Whisper decoder barely depends on its audio. Reversing the audio moved logits by about 5e-5 (below the 1e-4 tolerance), and shuffling labels across clips left the loss unchanged, so the random fixture missed both on every seed. The overfit Whisper had to tell clips apart: the same faults moved logits by 1–24 and loss from about 0.01 to 4–7.
- **Where it did not: everything else.** For SmolLM, CLIP, BERT, T5, and Wav2Vec2, random fixtures already depended on their inputs and caught the same faults.
- **Exact answers are weak on their own.** Overfit answers caught only 1–3 of 6–8 faults per fixture. Overfitting helps by making the numbers input-dependent, not by making answers checkable.
- **Cost and upkeep.** Training took 0.3–3.5 s per fixture for BERT, T5, and Wav2Vec2, and 23–51 s for Whisper (3000-frame inputs). One Wav2Vec2 seed did not converge in 500 steps; it still caught every fault, but non-convergence is fixture maintenance and should fail the build visibly.

| Overfit cost | BERT | T5 | Whisper | Wav2Vec2 |
| --- | --- | --- | --- | --- |
| Steps | 50 | 90–110 | 180–390 | 160–500 |
| Training | 0.3–0.4 s | 0.7–1.0 s | 23–51 s | 1.0–3.5 s |
| Fixture size | 9.1 MB | 11.3 MB | 18.5 MB | 0.4 MB |

## How the experiments were run

- ARM Mac, CPU, float32, two threads, PyTorch 2.11.0, Transformers `main` at `4d536e1f` (unmodified), `atol=rtol=1e-4`.
- Seeds 17, 29, and 43; random and overfit fixtures share the same initialization, config, processor, inputs, and assertions.
- Configs and processors come from pinned real checkpoints: `HuggingFaceTB/SmolLM-135M`, `openai/clip-vit-base-patch32`, `google-bert/bert-base-uncased`, `google-t5/t5-small`, `openai/whisper-tiny`, `facebook/wav2vec2-base-960h`, `google/umt5-small`, `allenai/OLMo-1B-hf`.
- Tasks: capital-city completion (SmolLM, OLMo), colored-square retrieval (CLIP), `[MASK]` capital fill (BERT), "capital of X:" → city (T5, UMT5), two-tone audio clips → color word (Whisper) or spelled word (Wav2Vec2).
- Faults are handpicked and synthetic. They establish local sensitivity, not a historical catch rate or maintenance savings; the replay is the first step toward that.
- The eager-vs-SDPA control reloads each fixture with SDPA at load time. An earlier version switched attention after loading, which does not reach T5's encoder/decoder ([#49372](https://github.com/huggingface/transformers/issues/49372)); results were re-run with the fix and are unchanged (largest eager/SDPA difference 1.3e-5, no false alarms).

Reproduce:

``` sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py \
  --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
```

## Open work

- Shrink rules for coupled configs, nested multimodal configs, and MoE, and a measured success rate across the fleet.
- Bring stage 1 closer to stage 2: fixed seeds, pinned revisions, recorded baselines, unchanged processor files, and the input-sensitivity check, ideally in the same `create_dummy_models.py` pipeline rather than a parallel one.
- Publish fixtures as `hf-internal-testing/<model>-slim` and record their revisions in the fleet lock.
- Calibrate tolerances on the CI CPU platform and on one GPU, and decide which baselines must be hardware-specific.
- Measure fixture upkeep in the pilot: shrink-rule fixes, non-converging overfits, and baseline reviews.
