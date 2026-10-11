# Small fixtures for MICs

PR checks need a model that runs on CPU in seconds yet exercises the real integration path. This doc covers the recipe and the experiments behind it; real-regression results are in [Evidence](evidence.md).

## TL;DR

**Small fixtures work** when built from the real config and processor files, frozen, and checked numerically (experiments on SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2, CPU and one A10G; MusicGen Melody added since):

| Finding | Numbers |
| --- | --- |
| "Forward pass runs" catches nothing | 0 synthetic faults caught |
| Recorded logits, loss, and gradients catch faults | **126/126** overfit, **120/126** random (six missed Whisper faults); no false alarms |
| Slim fixtures catch real regressions | 8/8 in the original replay, given original processor files, the affected call, and the default runtime (SDPA). Later Whisper #48108 is missed by both weight kinds; no real bug has needed overfit weights ([comparison](#overfit-versus-random-what-can-we-conclude)) |
| No NaN; real weights not needed | 288 load-and-train checks in three dtypes; pruned real layers are 8–235× larger, sliced ones act like random |
| Fixtures carry across hardware; baselines don't | One baseline per platform class and dtype |
| One generic shrink rule goes far | [56% of architectures](#does-it-scale-to-every-architecture) (314 of 559), 15 of the 17 in #49103 |

**Recipe:** shrink the real config by layer pattern, seed, keep random weights if they pass an input-sensitivity gate, else overfit on the contract's training step; save in the source's dtype; check numeric health, freeze, record baselines per platform and dtype. Implemented in [`build_fixtures.py`](../example/build_fixtures.py), with no per-family code beyond reviewed recipe fields.

**Next moves:**

1. Redirect Transformers' tiny models to this recipe and pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103) ([why](#relation-to-todays-tiny-models)).
2. Close the [sweep](#does-it-scale-to-every-architecture) gap (coupled size fields, real processor inputs, multimodal image paths), then retire the smoke-only tests.
3. Give every contract a training step, including inference-only ones.
4. Check that the `test-artifact` tag keeps the nine published `hf-internal-testing/proto-*` fixtures out of search.
5. Calibrate tolerances on CPU-only CI runners and other GPUs.
6. Measure upkeep in the pilot: shrink-rule fixes, non-converging overfits, baseline reviews.

## Recipe

1. **Real structure, small weights, real dtype.** Shrink width (hidden 64, intermediate 128, 4 heads) and depth [by layer pattern](#shrinking-depth-by-layer-pattern). A recipe may set reviewed `config_overrides` where the generic rule misses a field (MusicGen Melody: EnCodec's `num_filters`, and `codebook_dim`, which must match the shrunk width). Save in the source's default dtype (float32 if unnamed): proto-SmolLM-slim is bfloat16, the others float32. Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE, MoE structure, and Dropout/LayerDrop/SpecAugment (contracts seed them). Keep the checkpoint's tensor names and omit parameters it lacks, so loading initializes them as for users (how the Wav2Vec2 fixture catches [#49385](https://github.com/huggingface/transformers/issues/49385)).
2. **Processor files copied byte for byte**, never re-saved. Checkpoints with identical processor files, kept config fields, and dtype share one fixture ([example](../example/README.md#the-parts)).
3. **Weights:** seed, then run the input-sensitivity gate.
   - Pass: keep random weights (CLIP, BERT, T5, UMT5, Wav2Vec2).
   - Fail: overfit on the contract's training batch in float32 (AdamW, lr 5e-3, clipping, ≤ 500 steps, until loss < 0.05 twice), cast to the fixture's dtype, and pass the gate (Whisper).
   - A recipe may require overfitting, with a reviewed reason and its own target loss: the gate reads teacher-forced logits, but random SmolLM and MusicGen Melody generate the same output for every prompt ([SmolLM](evidence.md#bfloat16-on-cpu), [MusicGen Melody](evidence.md#overfit-or-random-weights-on-real-regressions)).
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

- **Random** passes the current gate for CLIP, BERT, T5, UMT5, and Wav2Vec2. Whisper needs overfitting for audio sensitivity; SmolLM and MusicGen Melody need it for generation that responds to the prompt. **Overfit** improves sensitivity in the synthetic-fault study at the same size, and building it checks gradient flow; the real-regression comparisons below do not establish broader superiority. **Pruned** needs no training but grows with the model (SmolLM-135M at 2/30 layers: 142 MB). **Sliced** acts like random.
- **No NaN anywhere** proves little at width 64; the health gate checks each build.
- **Lower precision favors overfit, but the gate's choice stands.** In bfloat16, random fixtures changed discrete outputs (Wav2Vec2 3/3 seeds, T5 2/3), overfit never. Yet contract fixtures changed no discrete output in bfloat16 on the A10G, per-dtype baselines held at `1e-4`, and no real regression yet needs an overfit fixture ([replays](evidence.md#overfit-or-random-weights-on-real-regressions)).
- Exact overfit answers alone caught only 1–3 of 6–8 faults per fixture; only loss caught shuffled labels ([smollm_clip.json](../experiments/results/smollm_clip.json), [more_families.json](../experiments/results/more_families.json)).

### Overfit versus random: what can we conclude?

Overfitting gives us useful, readable input/output pairs for the text-generation examples we train. It has not yet caught a real regression that a matching random fixture misses. Detection, readability, and input sensitivity are separate benefits; equal detection on a few bugs does not establish equal coverage everywhere.

| Criterion | Overfit | Random | What the evidence supports |
| --- | --- | --- | --- |
| Human-readable generation | SmolLM answers name the prompted capital; Whisper labels the tone clips `red`, `green`, `blue`, `yellow`. | Random SmolLM repeats the same text for all prompts; random Whisper ignores the audio in generated answers. | Prefer overfit for examples whose outputs should be understandable and respond to their inputs. These learned answers are fixture outputs, not the checkpoint's predictions. |
| Synthetic-fault detection | 126/126; Whisper 24/24. | 120/126; Whisper 18/24. | Overfit improves measured sensitivity to six synthetic Whisper faults. This is not a real-bug catch rate. |
| Real #48679, MusicGen Melody, original or crossed prompts | Caught in generated codes. | Caught in generated codes. | Equal detection on these tested inputs. |
| Same real bug, identical text across rows | Misses all four tested input/guidance combinations. | Catches all four. | A counterexample to overfit being uniformly superior: overfit catches 8/12 cases, random 12/12. |
| Real #48028, Whisper, controlled unequal first-segment transcripts | Caught in freely generated continuation tokens. | Caught in freely generated continuation tokens. | Reaching the left-padding path matters; either weight kind detects the bug once the first segment is controlled. |
| Real #48108, Whisper, encoder-decoder assistant and speech | No token change. | No token change. | The real tiny checkpoint catches it; neither slim fixture covers this behavior on the tested inputs. |
| Conditioning sensitivity | Guided Melody swaps/drops change 1/40 decoded code IDs in the selected original-prompt case. | No decoded code change in that control. | Extra overfit sensitivity on a selected input, not a real overfit-only catch. Both fixture waveform heads remain zero; this does not show readable or meaningful generated audio. |
| Model size and runtime | Same architecture and weight size as its random twin. | Same architecture and weight size as its overfit twin. | Overfitting adds no parameters. Generated lengths can change runtime; equal size does not imply identical check cost. |
| Construction and upkeep | Training costs 0.3–51 s on CPU or 0.5–6 s on GPU in the six-family study; 1/18 did not converge. Depends on the training examples and gate. | Initialization < 0.1 s in that study; no convergence requirement. Still needs the input-sensitivity gate and reviewed baselines. | Training is an upfront cost for a frozen artifact, but adds a possible build/repair failure. |
| Precision stability | No changed discrete outputs in the study's overfit bfloat16 trials. | Some random T5/Wav2Vec2 seeds change discrete outputs in bfloat16. | An overfit advantage in that study; the chosen contract fixtures still use and pass separate dtype baselines. |

Sources: [weight-initialization study](#choosing-the-weights), [Melody bad/fix pairs and conditioning controls](evidence.md#musicgen-melody-conditioning-controls), [Whisper left-padding replay](evidence.md#whisper-left-padded-previous-segment-prompts), and [assisted-decoding replay](evidence.md#whisper-assisted-decoding-suppression).

**Decision:** prefer overfit when a contract needs readable generation or random weights fail its sensitivity gate. Keep one reviewed, frozen weight set per fixture in routine CI; there is no need to run overfit and random twins for every check. Keep random fixtures where they already satisfy the contract, and reserve paired comparisons for a specific question about detection or a fixture repair. The present evidence supports overfit as a useful choice for some contracts, not replacing every random fixture or declaring it superior on real regressions. We can stop broad random-versus-overfit experiments without turning that operational choice into an unsupported coverage claim.

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
- **Coupled fields are the main cost**, one small rule per family: M-RoPE sections (GLM-4V, Qwen-VL), conv groups, per-stage widths in hierarchical vision (Swin v2, PVT), necks tied to backbone stages, mamba-hybrid cache layouts, EnCodec's `codebook_dim` and doubling `num_filters` (MusicGen Melody's fixture sets both as `config_overrides`).
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
