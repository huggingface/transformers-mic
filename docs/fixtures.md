# Small fixtures for MICs

Companion to the [MICs proposed standard](../README.md#the-proposed-standard).

PR-level contract checks need a model that runs on CPU in seconds and still exercises the real integration path. This page gives the recipe and the experiments behind it.

## TL;DR

**Small fixtures work** if they are built from the real config and processor files, frozen, and checked numerically. Tested on six families (SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2), on CPU and one A10G:

| Finding | Numbers |
| --- | --- |
| "Forward pass runs" catches nothing | 0 synthetic faults caught |
| Recorded logits, loss, and targeted gradients catch faults | **126/126** overfit, **120/126** random; all six misses are Whisper (its random fixture ignores audio); no false alarms |
| Slim fixtures catch real regressions | All 8 reproducible historical ones, with original processor files and the affected public call under the default runtime (SDPA) |
| No NaN, no need for real weights | 288 load-and-train checks in float32, bfloat16, float16. Pruned real layers also fix Whisper but are 8–235× larger; sliced real tensors behave like random |
| Fixtures carry across hardware; baselines do not | Record baselines per platform class (CPU, GPU) and per dtype |
| One generic shrink rule goes far | [56% of Transformers architectures](#does-it-scale-to-every-architecture) (314 of 559), including 15 of the 17 in #49103. Most of the rest need rules for coupled size fields, or real processor inputs |

**Proposed recipe:** shrink the real config by layer pattern, seed, keep random weights if they pass an input-sensitivity gate, overfit only if not, check numeric health, freeze, record baselines per platform and dtype. [`example/build_fixtures.py`](../example/build_fixtures.py) implements it generically. Gate perturbations are defined per input name; overfit data is the contract's own training step.

**Next moves:**

1. **Redirect Transformers' tiny-model work to this recipe** ([why](#relation-to-todays-tiny-models)). Pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103). Change `create_dummy_models.py` to build from real configs, copy processor files, seed, and run the gates (overfitting where random weights fail).
2. **Close the gap on the [architecture sweep](#does-it-scale-to-every-architecture):**
   - per-architecture rules for coupled size fields (the largest failure group);
   - real processor inputs instead of synthetic ones;
   - the image path of multimodal models.

   Retire the smoke-only tests once the rate is acceptable.
3. **Give every contract a training step.** The builder captures the contract's own training call (inputs and labels by name) and overfits on it when needed. No per-family training data. Inference-only contracts need one added.
4. **Publish one fixture per fixture key** as `hf-internal-testing/proto-<name>-slim`, pinned in the fleet lock and kept out of model discovery. Done for the eight pilot fixtures ([`publish_fixtures.py`](../example/publish_fixtures.py)), e.g. [`proto-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/proto-whisper-tiny-slim). Still to check: that the `test-artifact` tag keeps them out of search.
5. **Calibrate tolerances** on CPU-only CI runners and other GPU types; record bfloat16 baselines.
6. **Measure upkeep in the pilot:** shrink-rule fixes, non-converging overfits, baseline reviews.

## Recipe

1. **Real structure, small weights.**
   - Start from the real config. Shrink width (hidden 64, intermediate 128, 4 heads in the experiments) and depth [by layer pattern](#shrinking-depth-by-layer-pattern).
   - Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE settings, and MoE structure.
   - Change nothing else. Dropout, LayerDrop, and SpecAugment stay as configured; contracts seed them.
   - Keep the real checkpoint's tensor names. Parameters the checkpoint lacks are dropped, so loading initializes them as it does for users.

   Together these let the Wav2Vec2 fixture catch [#49385](https://github.com/huggingface/transformers/issues/49385) (SpecAugment's mask vector left uninitialized on load). An earlier recipe hid it by saving that tensor and disabling SpecAugment.
2. **Real processor files, byte for byte.** Copy, never re-save. One fixture serves every checkpoint with the same fixture key: identical processor files, and identical config fields wherever shrinking keeps them ([example](../example/README.md#the-parts)).
3. **Weights.** Initialize from a fixed seed and run the input-sensitivity gate.
   - Pass: keep the random weights.
   - Fail: overfit on the contract's training batch (AdamW, lr 5e-3, gradient clipping, ≤ 500 steps, until loss < 0.05 twice in a row). The gate must pass afterwards.
   - Neither works: fail the build visibly.

   On the six families, the gate keeps random weights for SmolLM, CLIP, BERT, T5, UMT5, and Wav2Vec2. It overfits Whisper, the one family where random weights missed faults.
4. **Gates, run on the saved fixture.** A failed gate is a fixture build failure (visible missing coverage), not a contract failure.
   - **Input sensitivity.** Perturbations are defined once per input name (`input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`, `input_features`, `input_values`, ...): swap rows within the batch, reverse, all-ones mask, channel swap, halve. Each must move the logits by at least 10× the strictest tolerance the fixture must hold on any platform (1e-3 for CPU float32).
   - **Known blind spots** can be accepted in the manifest with a reason. Example: Wav2Vec2-base's bias-free first convolution feeds a group norm, so at fixture size the output ignores input amplitude (5e-5), while the real checkpoint moves by 3.9.
   - **Numeric health.** Outputs, loss, and gradients must be finite, and a training step must update parameters. Checked in every dtype and on every platform class the fixture will run on.
5. **Freeze and publish** at a pinned revision, marked as a test artifact and kept out of Hub search. Training and hardware matter again only on a rebuild: overfit convergence depends on hardware (a different Wav2Vec2 seed failed on CPU and on the A10G).
6. **Record baselines** on a known-good build under the default runtime (e.g. SDPA, with eager as a variant). Record discrete outputs, selected logits, loss, gradients on named modules, and whether an SGD step updates parameters. One baseline per platform class and dtype.

## What a fixture result tells an agent

- A fixture pass checks the integration path that fixture exercises. Nothing more.
- Overfit answers are not reference outputs for the real checkpoint.
- Neither fixture nor checkpoint results prove correctness on new inputs.
- Keep fixture results separate from real-checkpoint baselines, and expose which target, inputs, and runtime were checked.

The [agent walkthroughs](evidence.md#how-agents-use-the-hub-today) support tested examples, explicit environments, and reference checks. They do not validate this shrinking or overfitting recipe; the experiments below do, and the pilot must extend them. Agents should find a contract through a link or lookup from the real model. The fixture itself stays a test artifact, excluded from model search.

## Choosing the weights

Four ways to fill the same fixture, compared on the six families with identical faults, controls, and tolerances. Each saved fixture was reloaded in float32, bfloat16, and float16 for inference and a training step, on the Mac CPU and on the A10G. Detection counts were identical on both ([script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)).

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors cut to fixture width) |
| --- | ---: | ---: | ---: | ---: |
| Faults caught | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Smallest logit move from an input fault, Whisper | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients | 0/108 | 0/108 | 0/36 | 0/36 |
| Weights | 0.4–15 MB | same | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1/18 did not converge | ~1 s plus loading the checkpoint | < 0.1 s plus loading the checkpoint |
| Largest bfloat16 logit drift from float32 | 0.09 | 0.30 | 0.97 | 0.34 |

- **Random:** enough everywhere except Whisper. Its tiny decoder barely depends on the audio: reversing the audio moved logits by 5e-5, below the 1e-4 tolerance.
- **Overfit:** most sensitive, same size as random. Its build doubles as a one-time check that gradients flow and the model learns.
- **Pruned:** fixes Whisper without training, but size grows with the model. SmolLM-135M at 2 of 30 layers is 142 MB; Llama-3-8B at 2 layers would be about 1.5B parameters.
- **Sliced:** real weight statistics without the real function behave like random weights.
- **No NaN anywhere.** At width 64 on six families, that is not proof. The numeric-health gate checks it at build time instead of assuming it.

**Overfit is more stable in lower precision, but the gate's choice still stands.**

- In bfloat16, random fixtures changed discrete outputs against float32 (Wav2Vec2 on 3 of 3 seeds, T5 on 2 of 3, CPU and A10G alike). Overfit fixtures never did.
- A random fixture's smallest input signal can fall below bfloat16 drift. BERT padding: 1.7e-3 signal vs 4e-3 drift for random; 1.6–2.2 signal vs 0.14 drift for overfit.
- But on the A10G, contract fixtures changed no discrete output in bfloat16, and GPU baselines held at `1e-4` per platform and dtype. No platform needed a looser tolerance.
- If one ever does, the gate's threshold rises with it. At the measured bfloat16-vs-float32 drift (BERT 2.6e-3, CLIP 0.05), random BERT and CLIP fixtures would fail the gate and be overfit.
- Not shown yet: a real historical regression that only an overfit fixture catches. Random fixtures caught all 8 reproducible ones too.

Two smaller findings from earlier runs ([SmolLM/CLIP](../experiments/results/smollm_clip.json), [four more families](../experiments/results/more_families.json)):

- Exact overfit answers alone caught only 1–3 of 6–8 faults per fixture.
- Loss was the only check that caught shuffled labels.

## Shrinking depth by layer pattern

A fixed `num_hidden_layers = 2` is wrong whenever layers differ. 170 Transformers configs define `layer_types`, and MoE configs choose dense or sparse blocks per layer index ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)):

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

**Rule.**

- A layer's kind is its `layer_types` entry plus whether its MLP is dense or sparse (`first_k_dense_replace`/`moe_layer_freq`, or `mlp_only_layers`/`decoder_sparse_step`).
- Keep the first layer of each kind, in order. Re-index the per-layer fields.
- Result: 2 layers for sliding/full, 3 for sliding/full/mamba, 1 dense + 1 sparse for first-k-dense MoE.
- All 15 modeling files that read `first_k_dense_replace` only compare it with the layer index, so this is safe.

**Not covered.** Period-based patterns (Jamba) and index-dependent logic (cross-layer KV sharing) need per-architecture rules.

## Does it scale to every architecture?

The recipe was tried on every model type in Transformers (`4d536e1f`, 725 types), configs only ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints used](../experiments/results/sweep_sources.json)).

For each type, a pinned checkpoint's config was shrunk with **one generic rule**: widths by field name in every sub-config, depth by layer pattern, MoE to 4 experts, backbone stages left at defaults. The primary head was then built with random weights and run forward and backward on synthetic inputs sized from the config.

| | Model types |
| --- | ---: |
| No usable checkpoint (no doc checkpoint and none on the Hub, or only gated) | 86 |
| Component of a composite model only (`*_vision_model`, `*_text_model`), not a usage path | 80 |
| **Usage paths tried** | **559** |
| Built and ran forward and backward, all finite | **314 (56%)** |
| Failed: coupled size fields and other per-architecture constraints | 167 |
| Failed: synthetic inputs the model does not accept (video, time series, extra required inputs) | 61 |
| Failed: config round-trip or optional dependency | 17 |

- **Where it works:** 131 causal LMs, 42 image-text-to-text, 28 image classifiers, 24 masked LMs, 20 seq2seq, 8 CTC models.
  - Includes 15 of the 17 #49103 models. MusicGen fails a config round-trip; Qwen2.5-VL fails on its M-RoPE sections.
  - Median fixture: 6.5M parameters. The 11 above 50M keep a large real vocabulary (RemBERT, Gemma 3n) or a component the generic rule does not shrink (PerceptionLM's vision tower, LUKE's entity embeddings).
- **Coupled fields are the main cost.** Each needs a small rule, written once per family and reused by models sharing its code. Typical cases:
  - M-RoPE sections that must sum to half the head dimension (GLM-4V, Qwen-VL);
  - convolution groups that must divide channels;
  - per-stage widths and heads in hierarchical vision models (Swin v2, PVT);
  - neck sizes tied to backbone stages;
  - cache layouts in mamba hybrids.
- **Input failures are a sweep limitation, not a recipe one.** Contracts feed real processor outputs; the sweep only made synthetic tensors. For the same reason, the 62 multimodal models that passed ran text only; their image path is untested.
- **Side finding:** 89 config classes document a checkpoint whose config is another model type (e.g. `segformer` points at Seed-OSS, `granitemoehybrid` at Granite Speech). The sweep used the most downloaded ungated Hub model of the right type instead.

**Bottom line:** per-architecture work is unavoidable but bounded. About half of all architectures need none; the rest need one rule per family, not per checkpoint. Contracts multiply by usage path and fixture key, not by checkpoint.

## Hardware and dtype

The same fixture files ran unchanged on the A10G. Only baselines are platform- and dtype-specific ([evidence](evidence.md#on-gpu)):

- **Same GPU, float32:** stable; max drift 6.6e-5 over three runs.
- **CPU baselines on GPU:** fine for SmolLM, BERT, T5. CLIP, Whisper, and Wav2Vec2 drift 1e-3 to 3e-2 because cuDNN runs convolutions in TF32 by default. Keep the default (users get it) and record GPU baselines.
- **bfloat16 or float16 vs float32:** over tolerance everywhere; sometimes generated outputs change. Record a baseline in the dtype users get.
- **Discrete outputs are the portable signal.** Compare logits only against a baseline from the same platform class. Train and save fixtures in float32.

## Relation to today's tiny models

Transformers already builds `tiny-random-*` models (`utils/create_dummy_models.py`, nightly `check_tiny_models`, `hf-tiny-v2`) and is adding fast integration tests on them in [#49103](https://github.com/huggingface/transformers/pull/49103). Those models:

- come from unit-test configs, so real RoPE, sliding-window, and MoE settings are lost;
- use re-saved and resized processors;
- have no seed and no pinned revision;
- only get checks that forward and generate run.

**Proposed new direction:** stop extending the smoke-only tests and redirect the pipeline to this recipe.

- A passing forward pass caught 0 of 126 faults.
- Re-saved processor files hid three real loading regressions.
- Recorded numerical checks on overfit fixtures from real configs caught all of them.

Keep the pipeline's machinery (config shrinking for hundreds of architectures, processor discovery, publishing) and change it, rather than replacing it. Its breadth is what the recipe still has to match: the recipe is validated here on six families and three layer-pattern configs.

## How the experiments were run

- **CPU:** ARM Mac, float32, two threads, torch 2.11.0, Transformers `4d536e1f`, `atol=rtol=1e-4`.
- **GPU:** one A10G on the Transformers `aws-g5-4xlarge-cache` runners, torch 2.14.0+cu130, Transformers `8625785e` (the same tree), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml).
- **Fixtures:** seeds 17, 29, and 43. Configs and processors come from pinned real checkpoints listed in [`experiment.py`](../experiments/experiment.py).
- **Faults** are handpicked and synthetic: they show local sensitivity, not a historical catch rate. The [replay](evidence.md) covers real regressions.

```sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
python experiments/fixture_init.py    # random, overfit, pruned, sliced
python experiments/layer_patterns.py  # needs network for the three configs
python experiments/sweep.py prefetch && python experiments/sweep.py run  # every architecture
```
