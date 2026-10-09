# Small fixtures for MICs

PR checks need a model that runs on CPU in seconds yet exercises the real integration path. This doc covers the recipe and the experiments behind it; real-regression results are in [Evidence](evidence.md).

## TL;DR

**Small fixtures work** when built from the real config and processor files, frozen, and checked numerically (SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2; CPU and one A10G):

| Finding | Numbers |
| --- | --- |
| "Forward pass runs" catches nothing | 0 synthetic faults caught |
| Recorded logits, loss, and gradients catch faults | **126/126** overfit, **120/126** random (misses all Whisper); no false alarms |
| Slim fixtures catch real regressions | All 8 reproducible ones, given original processor files, the affected call, and the default runtime (SDPA) |
| No NaN; real weights not needed | 288 load-and-train checks in three dtypes; pruned real layers are 8–235× larger, sliced ones act like random |
| Fixtures carry across hardware; baselines don't | One baseline per platform class and dtype |
| One generic shrink rule goes far | [56% of architectures](#does-it-scale-to-every-architecture) (314 of 559), 15 of the 17 in #49103 |

**Recipe:** shrink the real config by layer pattern, seed, keep random weights if they pass an input-sensitivity gate, else overfit on the contract's training step; save in the source's dtype; check numeric health, freeze, record baselines per platform and dtype. Implemented in [`build_fixtures.py`](../example/build_fixtures.py).

**Next moves:**

1. Redirect Transformers' tiny models to this recipe and pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103) ([why](#relation-to-todays-tiny-models)).
2. Close the [sweep](#does-it-scale-to-every-architecture) gap (coupled size fields, real processor inputs, multimodal image paths), then retire the smoke-only tests.
3. Give every contract a training step, including inference-only ones.
4. Check that the `test-artifact` tag keeps the eight published `hf-internal-testing/proto-*` fixtures out of search.
5. Calibrate tolerances on CPU-only CI runners and other GPUs; record `cuda` baselines for bfloat16 proto-SmolLM-slim.
6. Measure upkeep in the pilot: shrink-rule fixes, non-converging overfits, baseline reviews.

## Recipe

1. **Real structure, small weights, real dtype.** Shrink width (hidden 64, intermediate 128, 4 heads) and depth [by layer pattern](#shrinking-depth-by-layer-pattern). A recipe may set reviewed `config_overrides` where the generic rule misses a field (MusicGen Melody: EnCodec's `num_filters`, and `codebook_dim`, which must match the shrunk width). Save in the source's default dtype (float32 if unnamed): proto-SmolLM-slim is bfloat16, the others float32. Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE, MoE structure, and Dropout/LayerDrop/SpecAugment (contracts seed them). Keep the checkpoint's tensor names and omit parameters it lacks, so loading initializes them as for users (how the Wav2Vec2 fixture catches [#49385](https://github.com/huggingface/transformers/issues/49385)).
2. **Processor files copied byte for byte**, never re-saved. Checkpoints with identical processor files, kept config fields, and dtype share one fixture ([example](../example/README.md#the-parts)).
3. **Weights:** seed, then run the input-sensitivity gate.
   - Pass: keep random weights (CLIP, BERT, T5, UMT5, Wav2Vec2).
   - Fail: overfit on the contract's training batch in float32 (AdamW, lr 5e-3, clipping, ≤ 500 steps, until loss < 0.05 twice), cast to the fixture's dtype, and pass the gate (Whisper).
   - A recipe may require overfitting, with a reviewed reason and its own target loss (SmolLM and MusicGen Melody: random weights generate the same output for every prompt; [why](evidence.md#bfloat16-on-cpu)).
   - Neither: fail the build.
4. **Gates on the saved fixture** (a failure is a build failure, not a contract failure):
   - **Input sensitivity:** per input name (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, ...): swap rows, reverse, all-ones mask, channel swap, halve. Each must move logits ≥ 10× the strictest tolerance (1e-3 for CPU float32).
   - **Known blind spots** are accepted in the manifest with a reason: Wav2Vec2-base's bias-free first convolution feeds a group norm, so the fixture ignores amplitude (5e-5; real checkpoint 3.9).
   - **Numeric health:** finite outputs, loss, gradients, and a parameter-updating training step, in every dtype and platform class used.
5. **Freeze and publish** at a pinned revision, as a test artifact out of Hub search. Rebuilds depend on hardware.
6. **Record baselines** on a known-good build under the default runtime (eager as a variant): discrete outputs, selected logits, loss, named-module gradients, whether an SGD step updates parameters.

## What a fixture result tells an agent

Only that the path it exercises behaves as reviewed on fixed inputs. Overfit answers are not the real checkpoint's outputs. Report fixture and checkpoint results separately; agents reach a contract from the real model, and the fixture stays out of search.

## Choosing the weights

Four ways to fill a fixture, six families, same faults and tolerances, three dtypes, Mac CPU and A10G (identical detection on both; [script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)).

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors cut to width) |
| --- | ---: | ---: | ---: | ---: |
| Faults caught | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Smallest Whisper logit move from an input fault | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients | 0/108 | 0/108 | 0/36 | 0/36 |
| Weights | 0.4–15 MB | same | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1/18 did not converge | ~1 s + checkpoint load | < 0.1 s + checkpoint load |
| Largest bfloat16 logit drift vs float32 | 0.09 | 0.30 | 0.97 | 0.34 |

- **Random** is enough except for Whisper, whose tiny decoder barely reads audio. **Overfit** is most sensitive at the same size, and building it checks gradient flow. **Pruned** needs no training but grows with the model (SmolLM-135M at 2/30 layers: 142 MB). **Sliced** acts like random.
- **No NaN anywhere** proves little at width 64; the health gate checks each build.
- **Lower precision favors overfit, but the gate's choice stands.** In bfloat16, random fixtures changed discrete outputs (Wav2Vec2 3/3 seeds, T5 2/3), overfit never. Yet contract fixtures changed no discrete output in bfloat16 on the A10G, per-dtype baselines held at `1e-4`, and no real regression yet needs an overfit fixture ([replays](evidence.md#overfit-or-random-weights-on-real-regressions)).
- Exact overfit answers alone caught only 1–3 of 6–8 faults per fixture; only loss caught shuffled labels ([smollm_clip.json](../experiments/results/smollm_clip.json), [more_families.json](../experiments/results/more_families.json)).

## Shrinking depth by layer pattern

A fixed `num_hidden_layers = 2` breaks when layers differ: 170 configs define `layer_types`, and MoE configs pick dense or sparse per layer ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)).

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

**Rule:** a layer's kind is its `layer_types` entry plus dense/sparse MLP (`first_k_dense_replace`/`moe_layer_freq`, or `mlp_only_layers`/`decoder_sparse_step`). Keep the first layer of each kind, in order; re-index per-layer fields. Period-based patterns (Jamba) and index-dependent logic (cross-layer KV sharing) need per-architecture rules.

## Does it scale to every architecture?

Every model type in Transformers (`4d536e1f`, 725), configs only ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints](../experiments/results/sweep_sources.json)). **One generic rule:** widths by field name in every sub-config, depth by layer pattern, MoE to 4 experts; primary head forward and backward, random weights, synthetic inputs.

| | Model types |
| --- | ---: |
| No usable checkpoint (none documented or on the Hub, or gated) | 86 |
| Component only (`*_vision_model`, `*_text_model`) | 80 |
| **Usage paths tried** | **559** |
| Built and ran forward and backward, all finite | **314 (56%)** |
| Failed: coupled size fields, other per-architecture constraints | 167 |
| Failed: synthetic inputs not accepted (video, time series, extra inputs) | 61 |
| Failed: config round-trip or optional dependency | 17 |

- **Works for** 131 causal LMs, 42 image-text-to-text, 28 image classifiers, 24 masked LMs, 20 seq2seq, 8 CTC, and 15 of 17 #49103 models (MusicGen fails a config round-trip, Qwen2.5-VL its M-RoPE sections). Median 6.5M parameters.
- **Coupled fields are the main cost**, one small rule per family: M-RoPE sections (GLM-4V, Qwen-VL), conv groups, per-stage widths in hierarchical vision (Swin v2, PVT), necks tied to backbone stages, mamba-hybrid cache layouts.
- **Input failures are a sweep limit:** contracts use real processor outputs; the 62 passing multimodal models ran text only.

**Bottom line:** about half of architectures need no per-model work; the rest need one rule per family, not per checkpoint.

## Hardware and dtype

Fixture files run unchanged on the A10G; baselines don't carry over ([numbers](evidence.md#on-gpu)). Record baselines per platform class in the dtype users get, and compare logits only within a platform class.

## Relation to today's tiny models

Transformers' `tiny-random-*` models (`utils/create_dummy_models.py`, now gaining fast tests in [#49103](https://github.com/huggingface/transformers/pull/49103)) come from unit-test configs (losing real RoPE, sliding-window, and MoE settings), use re-saved, resized processors, have no seed or pinned revision, and are only checked to run.

**Proposal:** redirect that pipeline to this recipe (real configs, copied processor files, seeds, gates) and keep its machinery (shrinking, processor discovery, publishing). A passing forward caught 0 of 126 faults, and re-saved processors hid three real regressions.

## How the experiments were run

- **CPU:** ARM Mac, float32, two threads, torch 2.11.0, Transformers `4d536e1f`, `atol=rtol=1e-4`.
- **GPU:** one A10G, torch 2.14.0+cu130, Transformers `8625785e` (same tree), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml).
- **Fixtures:** seeds 17, 29, 43; pinned checkpoints in [`experiment.py`](../experiments/experiment.py). Faults are handpicked and synthetic: they measure local sensitivity, not a catch rate.

```sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
python experiments/fixture_init.py    # random, overfit, pruned, sliced
python experiments/layer_patterns.py  # needs network for the three configs
python experiments/sweep.py prefetch && python experiments/sweep.py run  # every architecture
```
