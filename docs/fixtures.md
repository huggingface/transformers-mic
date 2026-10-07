# Small fixtures for MICs

Companion to the [MICs proposed standard](../README.md#the-proposed-standard).

PR-level contract checks need a model that is small enough to run on CPU in seconds but still exercises the real integration path. This document describes the project to build those small fixtures, step by step: what already exists in Transformers (stage 1), frozen random fixtures (stage 2), and overfit fixtures as a fallback (stage 3), with the local experiments behind each decision.

## What a fixture must provide

A small fixture is useful for contracts only if it has all of the following:

- **Real structure, small weights.** The real model config, shrunk (layers, width, heads, experts), keeping processor-coupled dimensions and model-specific settings: vocabulary, special tokens, image and patch sizes, audio dimensions, RoPE settings, MoE structure.
- **Real processor files, unchanged.** Tokenizer, processor, and preprocessing files copied from the real model byte for byte.
- **Frozen and pinned.** Deterministic initialization, frozen weights and inputs, published at a pinned revision (for example `hf-internal-testing/SmolLM-slim@<sha>`). One fixture serves every checkpoint with the same fixture key: byte-identical processor files and the same config fields wherever shrinking keeps them ([example](../example/README.md#the-parts)).
- **Recorded baselines.** Outputs, selected logits, loss, targeted gradients, and an optimizer update, recorded on a known-good build and reviewed, separately for each platform class and dtype the fixture runs on (see [Hardware and dtype](#hardware-and-dtype)).
- **Sensitive to its inputs.** Perturbing or swapping inputs must move the recorded logits well beyond tolerance.
- **Run like users run it.** Baselines recorded under the default runtime configuration (for example SDPA attention), with eager as an extra variant.

A failed fixture build should be visible as missing coverage, not silently skipped.

## Steps

| Stage | Produces | Status |
| --- | --- | --- |
| 1. Random tiny models from test configs | `tiny-random-*` models for smoke tests | exists in Transformers, being extended |
| 2. Frozen random fixtures from real configs | pinned fixtures with recorded baselines | validated locally on 8 families; not built at scale |
| 3. Overfit | fixtures trained on a few examples | proposed as the default after the [weights comparison](#choosing-the-weights); required where the random fixture fails the input-sensitivity check |

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

Stage 2 keeps stage 1's random weights but changes everything around them: the config is shrunk from the real model instead of taken from unit tests, processor files are copied unchanged instead of re-saved, and the fixture is seeded, pinned, and checked against recorded baselines instead of only being run.

Recipe:

1. Start from the real model config and copy the real processor files unchanged.
2. Shrink the config with architecture-specific rules. In the experiments: hidden size 64, intermediate size 128, 4 heads; vocabulary, image and patch sizes, and audio dimensions preserved; dropout and SpecAugment disabled so training checks are deterministic. Shrink depth by [layer pattern](#shrinking-depth-by-layer-pattern), not to a fixed count.
3. Initialize from a fixed seed and save.
4. Run the build gates: the input-sensitivity check (perturb or swap inputs and require logits to move well beyond tolerance; if it fails, go to stage 3) and the [numeric-health check](#numeric-health-gate).
5. Record baselines on a known-good build, under the default runtime configuration: discrete outputs, selected logits, loss, gradients on named modules, and whether an SGD step updates parameters.
6. Publish at a pinned revision, marked as a test artifact (tag and card) and kept out of default Hub search, then add it to the fleet lock.

Shrinking from the real config preserves more integration-relevant behavior than unit-test configs, but it will not work generically for every architecture. Nested configs, coupled fields (`layer_types`, `head_dim`), MoE models, and processor assumptions need explicit rules, with test configs as a fallback.

### Shrinking depth by layer pattern

A fixed `num_hidden_layers = 2` is wrong for any model whose layers differ: 170 Transformers configs define `layer_types` (sliding/full attention, mamba/attention, linear/full attention), and MoE configs choose dense or sparse blocks per layer index. Assigning `num_hidden_layers = 2` on a loaded config, as the experiments did, keeps only the first two entries of the pattern:

| Real config | Layers, kinds | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: sliding ×5, full ×1, repeated | 2 sliding; full attention is never built | layers 0 and 5: sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: mamba ×5, attention ×1, repeated; MoE everywhere | 2 mamba; forward crashes (the cache needs an attention layer) | layers 0 and 5: mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then 26 sparse (`first_k_dense_replace = 1`) | 1 dense + 1 sparse, by luck of `k = 1` | layers 0 and 1: dense + sparse |

The rule ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)): a layer's kind is its `layer_types` entry combined with whether its MLP is dense or sparse (`first_k_dense_replace` and `moe_layer_freq`, or `mlp_only_layers` and `decoder_sparse_step`). Keep the first layer of each distinct kind, in order, and re-index every per-layer list and index field (`layer_types`, `first_k_dense_replace`, `mlp_only_layers`) to the kept layers. This gives 2 layers for sliding/full, 3 for sliding/full/mamba, and 1 dense + 1 sparse for dense-then-sparse MoE. In all 15 modeling files that read `first_k_dense_replace`, it only selects dense or sparse per index (`layer_idx >= first_k_dense_replace`), so 1 + 1 covers both paths and the boundary between them. On all three configs, every kind was built, every layer ran, and logits, loss, and gradients were finite.

Not covered by the rule: configs that derive the pattern from periods and offsets instead of a list (Jamba's `attn_layer_period` and `expert_layer_period`) need per-architecture rules that rewrite the period, and logic that depends on absolute layer index beyond dense/sparse (KV-cache sharing across layers, per-layer RoPE) needs the same review.

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

## Stage 3: overfit

Recipe: train the stage 2 model on a few contrasting examples (AdamW, learning rate 5e-3, gradient clipping, up to 500 steps, stopping once answers are correct and loss is below 0.05 on two consecutive checks), then record baselines as in stage 2.

It is required when the random fixture fails the input-sensitivity check, and [proposed as the default](#choosing-the-weights) everywhere else. The evidence from the first experiments:

- **Where overfitting helped: Whisper.** The tiny random Whisper decoder barely depends on its audio. Reversing the audio moved logits by about 5e-5 (below the 1e-4 tolerance), and shuffling labels across clips left the loss unchanged, so the random fixture missed both on every seed. The overfit Whisper had to tell clips apart: the same faults moved logits by 1–24 and loss from about 0.01 to 4–7.
- **Where it did not: everything else.** For SmolLM, CLIP, BERT, T5, and Wav2Vec2, random fixtures already depended on their inputs and caught the same faults.
- **Exact answers are weak on their own.** Overfit answers caught only 1–3 of 6–8 faults per fixture. Overfitting helps by making the numbers input-dependent, not by making answers checkable.
- **Cost and upkeep.** Training took 0.3–3.5 s per fixture for BERT, T5, and Wav2Vec2, and 23–51 s for Whisper (3000-frame inputs) on the Mac; 0.5–6 s on the A10G. One Wav2Vec2 seed in three did not converge in 500 steps, and which seed depends on the hardware (17 on CPU, 43 on the A10G). It still caught every fault, but non-convergence is fixture maintenance and should fail the build visibly. Because the outcome depends on hardware, a fixture is trained once and frozen, never retrained in CI.

| Overfit cost | BERT | T5 | Whisper | Wav2Vec2 |
| --- | --- | --- | --- | --- |
| Steps | 50 | 90–110 | 180–390 | 160–500 |
| Training | 0.3–0.4 s | 0.7–1.0 s | 23–51 s | 1.0–3.5 s |
| Fixture size | 9.1 MB | 11.3 MB | 18.5 MB | 0.4 MB |

## Choosing the weights

Random weights from a fixed seed (stage 2) and overfit weights (stage 3) are two ways to fill the fixture. Two more come from the real checkpoint: keep its first layers at full width (**pruned**), or keep the fixture's architecture and fill each tensor with a leading slice of the real one (**sliced**: real weight statistics, not a functional model). The concern behind them is numeric: random initialization, init-scheme changes, hardware, or multimodal stacks could produce NaN or degenerate fixtures that real weights avoid.

All four were built for the six families and evaluated with the same faults, controls, and tolerances ([script](../experiments/fixture_init.py)). Each saved fixture was then loaded in float32, bfloat16, and float16 to run inference and a training step. Everything ran on the Mac CPU and on the A10G ([CPU results](../experiments/results/fixture_init.json), [GPU results](../experiments/results/fixture_init_gpu.json), [GPU run](https://github.com/huggingface/model-integration-contracts/actions/runs/37583722412)); detection counts were identical on both.

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors, fixture width) |
| --- | ---: | ---: | ---: | ---: |
| Fixtures (six families) | 18 (3 seeds) | 18 (3 seeds) | 6 | 6 |
| Faults caught, all checks | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Control false alarms | 0 | 0 | 0 | 0 |
| Smallest logit move from an input fault, Whisper | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients (3 dtypes × CPU and GPU) | 0 of 108 | 0 of 108 | 0 of 36 | 0 of 36 |
| Weights | 0.4–14.6 MB | same as random | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1 of 18 did not converge on each | 0.3–1.1 s after loading the real checkpoint | < 0.1 s after loading the real checkpoint |
| Largest bfloat16 logit drift from float32, CPU or GPU | 0.09 | 0.30 | 0.97 | 0.34 |

What this says:

- **No variant produced a NaN.** 288 load-and-train checks in three dtypes on two platforms, all finite, every training step passed. That is six small families at width 64, so it does not show NaN cannot happen; it shows the risk is not visible at this scale and should be caught by a gate rather than assumed away.
- **Real weights fix Whisper without training.** The pruned Whisper caught every fault, with input faults moving logits by 1.9–5.7, like the overfit one. But pruned fixtures are 8–235× larger, because the embeddings and every kept layer stay at full width, and they grow with the model: SmolLM-135M at 2 of 30 layers is 35M parameters (142 MB), Llama-3-8B at 2 layers would be about 1.5B (1.05B of it untied embeddings).
- **Slicing gives nothing over random.** Real weight statistics without the real function behave like random weights: the sliced Whisper still barely depends on its audio and missed the shuffled labels.
- **Overfit is the most sensitive and stays small.** It caught everything at the random fixture's size, and its build is itself a one-time test that gradients flow and the model can learn. Its cost is build time and non-convergence, which depends on hardware.
- **Larger logits drift more in lower precision.** Pruned fixtures drifted up to 0.97 in bfloat16 and changed outputs for CLIP and Wav2Vec2; random fixtures drifted up to 0.09. Either way, bfloat16 and float16 need their own baselines (see [Hardware and dtype](#hardware-and-dtype)).

Proposed rule, replacing "overfit only on failure":

1. Build the random fixture from a fixed seed, shrunk by layer pattern.
2. Overfit it by default. If it does not converge within budget, fail the build visibly, then fall back to pruned real layers where the size is acceptable, or to the random fixture if it passes the input-sensitivity check.
3. Run the gates on the final fixture: input sensitivity and numeric health.
4. Freeze it. Init, training, and hardware only matter again on a rebuild.

### Numeric-health gate

A fixture is accepted only if, after saving and reloading, it gives finite outputs, loss, and gradients, and a training step updates its parameters, in every dtype and on every platform class it will be checked on. A failure falls back as above (overfit, then pruned) and is reported as a fixture build failure, not as a contract failure. `dtype_health` in the [script](../experiments/fixture_init.py) is the check as run here.

## Hardware and dtype

A fixture is not tied to CPU. The same files (weights, shrunk config, processor files) ran unchanged on an NVIDIA A10G GPU in the [GPU run](evidence.md#on-gpu). What depends on hardware and dtype is the baseline, so the fixture is shared and its baselines are per platform class and dtype:

| Baseline recorded on → checked on | Logits at `1e-4` | Generated outputs |
| --- | --- | --- |
| Same GPU, float32 (three runs) | stable: max drift 6.6e-5 | unchanged |
| CPU → GPU, float32 | stable for SmolLM, BERT, T5 (≤ 5e-5); CLIP, Whisper, Wav2Vec2 drift 1e-3 to 3e-2 | unchanged |
| float32 → bfloat16 or float16 | over tolerance on every fixture | changed for some (T5 and Wav2Vec2 fixtures in bfloat16, real SmolLM in both) |

The CPU-to-GPU drift in the convolutional families is TF32: PyTorch lets cuDNN run convolutions in TF32 by default on Ampere and newer GPUs. With it turned off, CPU baselines checked on the GPU had no flags ([evidence](evidence.md#on-gpu)). Users run with the default, so a GPU contract keeps TF32 on and records its own GPU baseline.

For the fixture recipe this means:

- **Record the baselines where the check runs:** CPU baselines for CPU PR checks, GPU baselines for GPU jobs, in the dtype users get for that checkpoint (for example bfloat16 when the checkpoint is stored in bfloat16).
- **Discrete outputs are the portable signal.** In float32, generated tokens and labels stayed identical from CPU to GPU on every fixture; logits did not. Cross-platform comparisons should rely on outputs, with logits compared only against a baseline from the same platform class.
- **Keep the training step and freezing in float32.** Fixtures are trained and saved in float32 on any device; lower precision only applies when loading for a check.

## How the experiments were run

- ARM Mac, CPU, float32, two threads, PyTorch 2.11.0, Transformers `main` at `4d536e1f` (unmodified), `atol=rtol=1e-4`.
- GPU: one NVIDIA A10G on the Transformers `aws-g5-4xlarge-cache` runners, torch 2.14.0+cu130, Transformers `8625785e` (same tree as `4d536e1f`), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml). `CONTRACT_DEVICE` and `CONTRACT_DTYPE` select the device and dtype; fixtures are always trained and saved in float32.
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
python experiments/fixture_init.py   # random, overfit, pruned, sliced
python experiments/layer_patterns.py # needs network for the three configs
```

## Open work

- Shrink rules for coupled configs, nested multimodal configs, period-based layer patterns (Jamba), and MoE expert counts, and a measured success rate across the fleet. The [layer-pattern rule](#shrinking-depth-by-layer-pattern) is validated on three configs only.
- Bring stage 1 closer to stage 2: fixed seeds, pinned revisions, recorded baselines, unchanged processor files, and the input-sensitivity check, ideally in the same `create_dummy_models.py` pipeline rather than a parallel one.
- Publish fixtures as `hf-internal-testing/<name>-slim`, one per fixture key, and record their revisions in the fleet lock. Keep them out of model discovery: in the [agent walkthrough](agent-walkthrough.md), a size-filtered Hub search surfaced `trl-internal-testing/tiny-*` fixtures as candidate models, and a fleet of slim fixtures would add thousands more.
- Calibrate tolerances on the CPU-only CI runners and on the other GPU types in CI. On one A10G, baselines must be GPU-specific for convolutional families and dtype-specific everywhere.
- Measure fixture upkeep in the pilot: shrink-rule fixes, non-converging overfits, and baseline reviews.
