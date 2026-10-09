# Small fixtures for MICs

Companion to the [proposed standard](../README.md#the-proposed-standard). PR checks need a model that runs on CPU in seconds yet exercises the real integration path. Below: the recipe and the experiments behind it.

## TL;DR

**Small fixtures work** when built from the real config and processor files, frozen, and checked numerically. Tested on SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2, on CPU and one A10G:

| Finding | Numbers |
| --- | --- |
| "Forward pass runs" catches nothing | 0 synthetic faults caught |
| Recorded logits, loss, and gradients catch faults | **126/126** overfit, **120/126** random (all misses Whisper, whose random fixture ignores audio); no false alarms |
| Slim fixtures catch real regressions | All 8 reproducible ones, given original processor files, the affected call, and the default runtime (SDPA) |
| No NaN; real weights not needed | 288 load-and-train checks in float32, bfloat16, float16. Pruned real layers also fix Whisper but are 8–235× larger; sliced real tensors act like random |
| Fixtures carry across hardware; baselines don't | One baseline per platform class and dtype |
| One generic shrink rule goes far | [56% of architectures](#does-it-scale-to-every-architecture) (314 of 559), 15 of the 17 in #49103 |

**Recipe:** shrink the real config by layer pattern, seed, save in the source checkpoint's dtype, keep random weights if they pass an input-sensitivity gate, else overfit on the contract's training step; check numeric health, freeze, record baselines per platform and dtype. Implemented in [`build_fixtures.py`](../example/build_fixtures.py).

**Next moves:**

1. Redirect Transformers' tiny models to this recipe and pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103) ([why](#relation-to-todays-tiny-models)).
2. Close the [sweep](#does-it-scale-to-every-architecture) gap (coupled size fields, real processor inputs, multimodal image paths), then retire the smoke-only tests.
3. Give every contract a training step, including inference-only ones.
4. Publish one fixture per fixture key as `hf-internal-testing/proto-<name>-slim`. Done for the eight pilot fixtures ([`publish_fixtures.py`](../example/publish_fixtures.py), e.g. [`proto-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/proto-whisper-tiny-slim)); not yet checked: whether the `test-artifact` tag keeps them out of search.
5. Calibrate tolerances on CPU-only CI runners and other GPUs; record bfloat16 baselines.
6. Measure upkeep in the pilot: shrink-rule fixes, non-converging overfits, baseline reviews.

## Recipe

1. **Real structure, small weights, real dtype.** Shrink width (hidden 64, intermediate 128, 4 heads) and depth [by layer pattern](#shrinking-depth-by-layer-pattern). Save in the dtype the source checkpoint loads in by default (float32 if its config names none), so checks run the precision users get: proto-SmolLM-slim is bfloat16 like SmolLM-135M/360M; the others are float32 like their sources. Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE, MoE structure, and Dropout/LayerDrop/SpecAugment (contracts seed them). Keep the checkpoint's tensor names and drop parameters it lacks, so loading initializes them as for users (how the Wav2Vec2 fixture catches [#49385](https://github.com/huggingface/transformers/issues/49385)).
2. **Processor files copied byte for byte**, never re-saved. Checkpoints with identical processor files and kept config fields share one fixture ([example](../example/README.md#the-parts)).
3. **Weights:** seed, then run the input-sensitivity gate.
   - Pass: keep random weights (CLIP, BERT, T5, UMT5, Wav2Vec2).
   - Fail: overfit on the contract's training batch (AdamW, lr 5e-3, clipping, ≤ 500 steps, until loss < 0.05 twice) in float32, then cast to the fixture's dtype; the gate must then pass (Whisper).
   - A recipe can require overfitting with a reviewed reason (SmolLM: random weights pass but generate the same tokens for every prompt; its training rows share "The capital of", so it converges at loss 0.3, not 0.05).
   - Neither: fail the build.
4. **Gates on the saved fixture.** A failed gate is a build failure, not a contract failure.
   - **Input sensitivity:** perturbations per input name (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, ...): swap rows, reverse, all-ones mask, channel swap, halve. Each must move logits ≥ 10× the strictest tolerance (1e-3 for CPU float32).
   - **Known blind spots** can be accepted in the manifest with a reason. Example: Wav2Vec2-base's bias-free first convolution feeds a group norm, so the fixture ignores input amplitude (5e-5); the real checkpoint moves 3.9.
   - **Numeric health:** finite outputs, loss, gradients, and a training step that updates parameters, in every dtype and platform class used.
5. **Freeze and publish** at a pinned revision, as a test artifact out of Hub search. Rebuilds depend on hardware: a different Wav2Vec2 seed failed to converge on CPU and A10G.
6. **Record baselines** on a known-good build under the default runtime (eager as a variant): discrete outputs, selected logits, loss, named-module gradients, and whether an SGD step updates parameters.

## What a fixture result tells an agent

A fixture pass checks only the path it exercises. Overfit answers are not reference outputs for the real checkpoint; no result proves correctness on new inputs. Report fixture and checkpoint results separately, naming target, inputs, and runtime. Agents reach a contract from the real model; the fixture stays out of search.

## Choosing the weights

Four ways to fill a fixture, six families, same faults and tolerances, in three dtypes on Mac CPU and A10G (identical detection on both; [script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)).

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors cut to width) |
| --- | ---: | ---: | ---: | ---: |
| Faults caught | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Smallest Whisper logit move from an input fault | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients | 0/108 | 0/108 | 0/36 | 0/36 |
| Weights | 0.4–15 MB | same | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1/18 did not converge | ~1 s + checkpoint load | < 0.1 s + checkpoint load |
| Largest bfloat16 logit drift vs float32 | 0.09 | 0.30 | 0.97 | 0.34 |

- **Random:** enough except Whisper, whose tiny decoder barely reads audio (reversed audio: 5e-5, under the 1e-4 tolerance).
- **Overfit:** most sensitive, same size; building it checks gradient flow.
- **Pruned:** no training, but grows with the model (SmolLM-135M at 2/30 layers: 142 MB; Llama-3-8B: ~1.5B parameters).
- **Sliced:** acts like random.
- **No NaN anywhere**, which is not proof at width 64; the health gate checks it per build.

**Lower precision favors overfit, but the gate's choice stands.** In bfloat16, random fixtures changed discrete outputs vs float32 (Wav2Vec2 3/3 seeds, T5 2/3), overfit never; a random signal can sit under bfloat16 drift (BERT padding: 1.7e-3 vs 4e-3; overfit 1.6–2.2 vs 0.14). But contract fixtures changed no discrete output in bfloat16 on the A10G, and per-dtype baselines held at `1e-4`. A looser tolerance would also raise the gate threshold: at measured drift (BERT 2.6e-3, CLIP 0.05), random BERT and CLIP would be overfit. No real regression yet needs an overfit fixture.

Earlier runs ([smollm_clip.json](../experiments/results/smollm_clip.json), [more_families.json](../experiments/results/more_families.json)): exact overfit answers alone caught 1–3 of 6–8 faults per fixture; only loss caught shuffled labels.

## Shrinking depth by layer pattern

A fixed `num_hidden_layers = 2` breaks when layers differ: 170 configs define `layer_types`, and MoE configs pick dense or sparse per layer ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)).

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

**Rule:** a layer's kind is its `layer_types` entry plus dense/sparse MLP (`first_k_dense_replace`/`moe_layer_freq`, or `mlp_only_layers`/`decoder_sparse_step`). Keep the first layer of each kind, in order; re-index per-layer fields. Safe: all 15 modeling files reading `first_k_dense_replace` only compare it with the layer index.

**Not covered:** period-based patterns (Jamba) and index-dependent logic (cross-layer KV sharing) need per-architecture rules.

## Does it scale to every architecture?

Every model type in Transformers (`4d536e1f`, 725 types), configs only ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints](../experiments/results/sweep_sources.json)). **One generic rule:** widths by field name in every sub-config, depth by layer pattern, MoE to 4 experts, backbone stages at defaults. The primary head ran forward and backward, random weights, synthetic inputs.

| | Model types |
| --- | ---: |
| No usable checkpoint (none documented or on the Hub, or gated) | 86 |
| Component only (`*_vision_model`, `*_text_model`) | 80 |
| **Usage paths tried** | **559** |
| Built and ran forward and backward, all finite | **314 (56%)** |
| Failed: coupled size fields, other per-architecture constraints | 167 |
| Failed: synthetic inputs not accepted (video, time series, extra inputs) | 61 |
| Failed: config round-trip or optional dependency | 17 |

- **Works for:** 131 causal LMs, 42 image-text-to-text, 28 image classifiers, 24 masked LMs, 20 seq2seq, 8 CTC; 15 of 17 #49103 models (MusicGen fails a config round-trip, Qwen2.5-VL its M-RoPE sections). Median 6.5M parameters; the 11 above 50M keep a large vocabulary (RemBERT, Gemma 3n) or an unshrunk component (PerceptionLM vision tower, LUKE entity embeddings).
- **Coupled fields are the main cost**, one small rule per family: M-RoPE sections summing to half the head dim (GLM-4V, Qwen-VL), conv groups dividing channels, per-stage widths in hierarchical vision (Swin v2, PVT), necks tied to backbone stages, mamba-hybrid cache layouts.
- **Input failures are a sweep limit:** contracts use real processor outputs. The 62 passing multimodal models likewise ran text only.
- **Side finding:** 89 config classes document a checkpoint of another model type (`segformer` → Seed-OSS, `granitemoehybrid` → Granite Speech); the sweep used the most downloaded ungated model of the right type.

**Bottom line:** about half of architectures need no per-model work; the rest need one rule per family, not per checkpoint.

## Hardware and dtype

Fixture files ran unchanged on the A10G; baselines did not carry over (TF32 convolutions, lower dtypes; [numbers](evidence.md#on-gpu)). Record baselines per platform class, in the dtype users get; compare logits only within a platform class; train and save fixtures in float32.

## Relation to today's tiny models

Transformers' `tiny-random-*` models (`utils/create_dummy_models.py`, nightly `check_tiny_models`, `hf-tiny-v2`), now gaining fast integration tests in [#49103](https://github.com/huggingface/transformers/pull/49103):

- come from unit-test configs, losing real RoPE, sliding-window, and MoE settings;
- use re-saved, resized processors;
- have no seed or pinned revision;
- are only checked to run forward and generate.

**Proposal:** redirect the pipeline to this recipe: real configs, copied processor files, seeds, gates. A passing forward caught 0 of 126 faults and re-saved processors hid three real regressions. Keep its machinery (shrinking, processor discovery, publishing); the recipe, validated on six families, has yet to match its breadth.

## How the experiments were run

- **CPU:** ARM Mac, float32, two threads, torch 2.11.0, Transformers `4d536e1f`, `atol=rtol=1e-4`.
- **GPU:** one A10G (`aws-g5-4xlarge-cache`), torch 2.14.0+cu130, Transformers `8625785e` (same tree), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml).
- **Fixtures:** seeds 17, 29, 43; pinned checkpoints in [`experiment.py`](../experiments/experiment.py).
- **Faults** are handpicked and synthetic: local sensitivity, not a catch rate. See the [replay](evidence.md) for real regressions.

```sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
python experiments/fixture_init.py    # random, overfit, pruned, sliced
python experiments/layer_patterns.py  # needs network for the three configs
python experiments/sweep.py prefetch && python experiments/sweep.py run  # every architecture
```
