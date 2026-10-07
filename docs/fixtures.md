# Small fixtures for MICs

Companion to the [MICs proposed standard](../README.md#the-proposed-standard).

PR-level contract checks need a model small enough to run on CPU in seconds that still exercises the real integration path. This document gives the recipe and the local experiments behind it.

## TL;DR

**Conclusion.** Small fixtures work if they are built from the real config and processor files, frozen, and checked numerically. On six families (SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2), on CPU and one A10G:

- A passing forward pass caught **0** synthetic faults. Recorded logits, loss, and targeted gradients caught **126/126** with overfit weights and **120/126** with random weights. All six misses were Whisper, whose random fixture ignores its audio. There were no false alarms.
- Slim fixtures caught all 8 reproducible historical regressions, once the original processor files were copied and the contract called the affected public API under the default runtime (SDPA).
- No fixture produced a NaN: 288 load-and-train checks in float32, bfloat16, and float16. Real weights are not needed. Pruned real layers also fix Whisper but are 8–235× larger, and sliced real tensors behave like random ones.
- Fixtures carry across hardware; baselines do not. Record baselines per platform class (CPU, GPU) and per dtype.
- One generic shrink rule already builds and runs [56% of Transformers architectures](#does-it-scale-to-every-architecture) (314 of 559), including 15 of the 17 in #49103. Most of the rest need per-architecture rules for coupled size fields, or real processor inputs.

**Proposed recipe:** shrink the real config by layer pattern, seed, keep random weights if they pass an input-sensitivity gate and overfit only if they do not, check numeric health, freeze, and record baselines per platform and dtype. [`example/build_fixtures.py`](../example/build_fixtures.py) implements it generically: the gate's perturbations are defined per input name, and the overfit data is the contract's own training step.

**Next moves:**

1. **Redirect the tiny-model work in Transformers to this recipe** (proposed new direction, see [below](#relation-to-todays-tiny-models)): pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103), and change `create_dummy_models.py` to build from real configs, copy processor files, seed, and run the gates (overfitting where the random weights fail).
2. **Close the gap on the [architecture sweep](#does-it-scale-to-every-architecture):** per-architecture rules for coupled size fields (the largest failure group), real processor inputs instead of synthetic ones, and the image path of multimodal models. Retire the smoke-only tests once the rate is acceptable.
3. **Give every contract a training step.** The builder captures the contract's own training call (inputs and labels by name) and overfits on it when needed, so no separate training data is written per family. Contracts for inference-only usage paths need one added.
4. **Publish one fixture per fixture key** as `hf-internal-testing/<name>-slim`, kept out of model discovery, and pin it in the fleet lock.
5. **Calibrate tolerances** on the CPU-only CI runners and the other GPU types, and record bfloat16 baselines.
6. **Measure upkeep in the pilot:** shrink-rule fixes, non-converging overfits, and baseline reviews.

## Recipe

1. **Real structure, small weights.** Start from the real config. Shrink width (hidden 64, intermediate 128, 4 heads in the experiments) and depth [by layer pattern](#shrinking-depth-by-layer-pattern). Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE settings, and MoE structure. Change nothing else: dropout, LayerDrop, and SpecAugment stay as configured, since contracts seed them. Keep the real checkpoint's tensor names too: parameters the real checkpoint lacks are dropped from the fixture, so loading initializes them as it does for users. With both, the Wav2Vec2 fixture catches [#49385](https://github.com/huggingface/transformers/issues/49385) (SpecAugment's mask vector left uninitialized on load), which an earlier version of the recipe hid by saving that tensor and disabling SpecAugment.
2. **Real processor files, byte for byte.** Copy them; never re-save them. One fixture serves every checkpoint with the same fixture key: identical processor files and identical config fields wherever shrinking keeps them ([example](../example/README.md#the-parts)).
3. **Weights:** initialize from a fixed seed and run the input-sensitivity gate. Keep the random weights if they pass. Otherwise overfit on the contract's training batch (AdamW, lr 5e-3, gradient clipping, ≤ 500 steps, until loss < 0.05 twice in a row), and require the gate to pass afterwards. If neither works, fail the build visibly. On the six families, the gate keeps random weights for SmolLM, CLIP, BERT, T5, UMT5, and Wav2Vec2 and overfits Whisper, the one family where random weights missed faults.
4. **Gates, on the saved fixture:**
   - **Input sensitivity:** perturbations defined once per input name (`input_ids`, `attention_mask`, `token_type_ids`, `pixel_values`, `input_features`, `input_values`, ...: swap rows within the batch, reverse, all-ones mask, channel swap, halve) must each move the logits by at least 10× the strictest tolerance the fixture must hold on any platform (1e-3 for CPU float32). A known blind spot can be accepted in the manifest with a reason: Wav2Vec2-base's bias-free first convolution feeds a group norm, so at fixture size the output ignores input amplitude (5e-5) while the real checkpoint moves by 3.9.
   - **Numeric health:** outputs, loss, and gradients must be finite, and a training step must update parameters, in every dtype and on every platform class the fixture will be checked on.

   A failed gate is a fixture build failure, visible as missing coverage, not a contract failure.
5. **Freeze and publish** at a pinned revision, marked as a test artifact and kept out of Hub search. Training and hardware only matter again on a rebuild. Overfit convergence depends on hardware: a different Wav2Vec2 seed failed on CPU and on the A10G.
6. **Record baselines** on a known-good build under the default runtime (for example SDPA, with eager as a variant). Record discrete outputs, selected logits, loss, gradients on named modules, and whether an SGD step updates parameters, separately per platform class and dtype.

## What a fixture result tells an agent

A fixture pass checks the integration path exercised by that fixture. Its overfit answers are not reference outputs for the real checkpoint, and neither result establishes correctness on arbitrary new inputs. Keep fixture results separate from the real-checkpoint baselines and expose which target, inputs, and runtime were checked.

The [agent walkthroughs](evidence.md#how-agents-use-the-hub-today) support tested usage examples, explicit environments, and reference checks; they do not independently validate this shrinking or overfitting recipe. That evidence comes from the experiments below and must be extended in the pilot. Agents should discover a model's contract through a link or lookup from the real model, while the fixture itself remains marked as a test artifact and excluded from candidate-model search.

## Choosing the weights

Four ways to fill the same fixture were compared on the six families with identical faults, controls, and tolerances. Each saved fixture was also reloaded in float32, bfloat16, and float16 for inference and a training step, on the Mac CPU and on the A10G. Detection counts were identical on both ([script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)).

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors cut to fixture width) |
| --- | ---: | ---: | ---: | ---: |
| Faults caught | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Smallest logit move from an input fault, Whisper | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients | 0/108 | 0/108 | 0/36 | 0/36 |
| Weights | 0.4–15 MB | same | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1/18 did not converge | ~1 s plus loading the checkpoint | < 0.1 s plus loading the checkpoint |
| Largest bfloat16 logit drift from float32 | 0.09 | 0.30 | 0.97 | 0.34 |

- **Overfit:** most sensitive, same size as random. Its build is also a one-time check that gradients flow and the model learns.
- **Overfit is also more stable in lower precision.** Reloaded in bfloat16, random fixtures changed their discrete outputs against float32 (Wav2Vec2 on 3 of 3 seeds, T5 on 2 of 3, CPU and A10G alike); overfit fixtures never did. Random fixtures' smallest input signal can also sit below bfloat16 drift (BERT padding: 1.7e-3 signal, 4e-3 drift; overfit: 1.6–2.2 signal, 0.14 drift). Where a platform needs a looser tolerance, the gate's threshold rises with it and selects overfit there. What is not shown yet is a real historical regression that only an overfit fixture catches: all 8 reproducible ones were caught by random fixtures too.
- **Random:** enough everywhere except Whisper, whose tiny decoder barely depends on the audio. Reversing the audio moved logits by 5e-5, below the 1e-4 tolerance.
- **Pruned:** fixes Whisper without training. But size grows with the model: SmolLM-135M at 2 of 30 layers is 142 MB, and Llama-3-8B at 2 layers would be about 1.5B parameters.
- **Sliced:** real weight statistics without the real function behave like random weights.
- **No NaN anywhere.** At width 64 on six families, that is not proof the risk is gone, so the numeric-health gate checks it at build time instead of assuming it.

Two smaller findings from the earlier runs ([SmolLM/CLIP](../experiments/results/smollm_clip.json), [four more families](../experiments/results/more_families.json)): exact overfit answers alone caught only 1–3 of 6–8 faults per fixture, and loss was the only check that caught shuffled labels.

## Shrinking depth by layer pattern

A fixed `num_hidden_layers = 2` is wrong whenever layers differ. 170 Transformers configs define `layer_types`, and MoE configs choose dense or sparse blocks per layer index ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)):

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

**Rule.** A layer's kind is its `layer_types` entry plus whether its MLP is dense or sparse (`first_k_dense_replace`/`moe_layer_freq`, or `mlp_only_layers`/`decoder_sparse_step`). Keep the first layer of each kind, in order, and re-index the per-layer fields. This gives 2 layers for sliding/full, 3 for sliding/full/mamba, and 1 dense + 1 sparse for first-k-dense MoE. All 15 modeling files that read `first_k_dense_replace` only compare it with the layer index.

**Not covered.** Period-based patterns such as Jamba's, and index-dependent logic such as cross-layer KV sharing, need per-architecture rules.

## Does it scale to every architecture?

The recipe was tried on every model type in Transformers (`4d536e1f`, 725 types), with configs only ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints used](../experiments/results/sweep_sources.json)). For each type, a pinned checkpoint's config was shrunk with **one generic rule**: widths by field name in every sub-config, depth by layer pattern, MoE to 4 experts, backbone stages left to defaults. Then the primary head was built with random weights and run forward and backward on synthetic inputs sized from the config.

| | Model types |
| --- | ---: |
| No usable checkpoint (no doc checkpoint and none on the Hub, or only gated) | 86 |
| Component of a composite model only (`*_vision_model`, `*_text_model`), not a usage path | 80 |
| **Usage paths tried** | **559** |
| Built and ran forward and backward, all finite | **314 (56%)** |
| Failed: coupled size fields and other per-architecture constraints | 167 |
| Failed: synthetic inputs the model does not accept (video, time series, extra required inputs) | 61 |
| Failed: config round-trip or optional dependency | 17 |

- **Where it works:** 131 causal LMs, 42 image-text-to-text, 28 image classifiers, 24 masked LMs, 20 seq2seq, and 8 CTC models. That includes 15 of the 17 #49103 models; MusicGen fails a config round-trip, and Qwen2.5-VL its M-RoPE sections. The median fixture has 6.5M parameters. The 11 above 50M keep a large real vocabulary (RemBERT, Gemma 3n) or a component the generic rule does not shrink (PerceptionLM's vision tower, LUKE's entity embeddings).
- **Coupled fields are the main cost.** Typical cases: M-RoPE sections that must sum to half the head dimension (GLM-4V, Qwen-VL), convolution groups that must divide channels, per-stage widths and heads in hierarchical vision models (Swin v2, PVT), neck sizes tied to backbone stages, and cache layouts in mamba hybrids. Each needs a small per-architecture rule, written once per family and reusable by models that share the family's code.
- **Input failures are a limitation of the sweep, not of the recipe.** A contract feeds real processor outputs; the sweep only made synthetic tensors. For the same reason, the 62 multimodal models that passed ran text only, and their image path is untested here.
- **Side finding:** 89 config classes in Transformers document a checkpoint whose config is another model type (for example `segformer` points at Seed-OSS, `granitemoehybrid` at Granite Speech). The sweep used the most downloaded ungated Hub model of the right type instead.

So per-architecture work is unavoidable, but it is bounded: roughly half of all architectures need none, and the rest need a rule per family, not per checkpoint. Contracts multiply by usage path and fixture key, not by checkpoint.

## Hardware and dtype

The same fixture files ran unchanged on the A10G; only baselines are platform- and dtype-specific ([evidence](evidence.md#on-gpu)):

- **Same GPU, float32:** stable, with a max drift of 6.6e-5 over three runs.
- **CPU baselines on GPU:** fine for SmolLM, BERT, and T5. CLIP, Whisper, and Wav2Vec2 drift 1e-3 to 3e-2, because cuDNN runs convolutions in TF32 by default. Keep the default, since users get it, and record GPU baselines.
- **bfloat16 or float16 against float32:** over tolerance everywhere, and sometimes the generated outputs change. Record a baseline in the dtype users get.
- **Discrete outputs are the portable signal across platforms.** Compare logits only against a baseline from the same platform class. Train and save fixtures in float32.

## Relation to today's tiny models

Transformers already builds `tiny-random-*` models (`utils/create_dummy_models.py`, nightly `check_tiny_models`, `hf-tiny-v2`) and is adding fast integration tests on them in [#49103](https://github.com/huggingface/transformers/pull/49103). Those models:

- come from unit-test configs, so real RoPE, sliding-window, and MoE settings are lost;
- use re-saved and resized processors;
- have no seed and no pinned revision;
- only get checks that forward and generate run.

**Proposed new direction, backed by the data above:** stop extending the smoke-only tests and redirect the pipeline to this recipe. A passing forward pass caught 0 of 126 faults, and re-saved processor files hid three real loading regressions. Recorded numerical checks on overfit fixtures from real configs caught all of them. The pipeline's machinery (config shrinking for hundreds of architectures, processor discovery, publishing) is what the recipe needs, so it is kept and changed, not replaced. Its breadth is what the recipe still has to match: it is validated here on six families and three layer-pattern configs.

## How the experiments were run

- **CPU:** ARM Mac, float32, two threads, torch 2.11.0, Transformers `4d536e1f`, `atol=rtol=1e-4`.
- **GPU:** one A10G on the Transformers `aws-g5-4xlarge-cache` runners, torch 2.14.0+cu130, Transformers `8625785e` (the same tree), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml).
- **Fixtures:** seeds 17, 29, and 43. Configs and processors come from pinned real checkpoints listed in [`experiment.py`](../experiments/experiment.py).
- **Faults** are handpicked and synthetic, so they show local sensitivity, not a historical catch rate. The [replay](evidence.md) covers real regressions.

```sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
python experiments/fixture_init.py    # random, overfit, pruned, sliced
python experiments/layer_patterns.py  # needs network for the three configs
python experiments/sweep.py prefetch && python experiments/sweep.py run  # every architecture
```
