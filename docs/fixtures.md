# Small fixtures for MICs

PR checks need models that run on CPU in seconds and exercise real integration paths. Recipe and experiments below; real-regression details in [Evidence](evidence.md).

## TL;DR

**Small fixtures work** with real configs and processor files, frozen weights, and numeric checks. The original replay caught **8/8 real regressions**, using the original processors, affected calls, and default runtime (SDPA). Later comparisons show limits, including a shared Whisper #48108 miss; no real bug has required overfit weights ([comparison](#overfit-versus-random-what-can-we-conclude)).

[`build_fixtures.py`](../example/build_fixtures.py) implements the [recipe](#recipe), with reviewed recipe fields as its only per-family specialization. Weight-study numbers, shrink coverage, and platform rules appear below.

**Next moves:**

1. Redirect tiny-model builds to this recipe; pause [#49103](https://github.com/huggingface/transformers/pull/49103)'s smoke-only tests ([why](#relation-to-todays-tiny-models)).
2. Close [sweep](#does-it-scale-to-every-architecture) gaps (coupled fields, real processor inputs, multimodal image paths), then retire smoke-only tests.
3. Keep a training check per trainable model family; inference-only usage paths can share that coverage.
4. Verify `test-artifact` hides the nine published `hf-internal-testing/proto-*` fixtures from search.
5. Calibrate CPU-only CI and other-GPU tolerances.
6. Measure pilot upkeep: shrink-rule fixes, overfit non-convergence, baseline reviews.

## Recipe

1. **Shrink the real config** ([`shrink()`](../experiments/sweep.py)), in every sub-config, innermost first:
   - **Widths:** hidden 64, intermediate 128, 4 heads, by field name.
   - **Attention style:** KV heads become 4 under MHA, 2 otherwise.
   - **MoE:** 4 experts, top-2, one expert group.
   - **Depth:** [by layer pattern](#shrinking-depth-by-layer-pattern), with at least one MoE layer after leading dense layers.
   - **Stage-based models** (Swin, PVT, ConvNeXt) keep their stage count at depth 1 per stage, with head counts that divide each stage width.
   - **Coupled fields follow the shrunk value** they were equal to in the original:
     - a projection that equals the text width, or projects into the language model
     - MLA `head_dim = qk_rope_head_dim`
     - Mllama's `vision_output_dim`, Granite Speech's projector `encoder_hidden_size`, BLT's global width
     - Llama 4's pixel-shuffle adapter, a merged patch width
   - **Layer-index fields** are renumbered into the shrunk depth: cross-attention layers, intermediate and deepstack indices, CLIPSeg `extract_layers`, DPT `backbone_out_indices`.
   - **RoPE:** M-RoPE sections and LongRoPE factors are resampled to the shrunk head.
   - **Dropped:** `quantization_config`, since fixtures are dense. A vanishing `initializer_range` (< 1e-3) becomes 0.02.
   - **Reviewed `config_overrides`** handle missed fields (MusicGen Melody: EnCodec `num_filters`, `codebook_dim`; the latter must match the shrunk width).

   Preserve vocabulary, special tokens, image/patch/audio dimensions, RoPE type, and seeded Dropout/LayerDrop/SpecAugment. Check the result with [`tmic shrink`](cli.md#step-by-step-one-model): it compares the layer kinds, module classes and properties the original builds with the fixture's, and lists what was lost.
   - **Keep the checkpoint's form:** tensor names, prefix, file format (safetensors or `pytorch_model.bin`, sharded or not), and missing parameters, so loading runs the same converters and initializes omissions as for users (Wav2Vec2 [#49385](https://github.com/huggingface/transformers/issues/49385)).
   - **Weights files that are not in a supported format** (`.pkl`, gguf, npz, pdparams) are not copied, and copied files over 128 MiB are refused.
   - **dtype:** save in the source's default dtype, float32 if unspecified: mic-SmolLM-slim is bfloat16, others float32.
2. **Copy processors byte for byte.** Identical processor files, preserved config fields, and dtype permit sharing a fixture ([example](../example/README.md#the-parts)).
3. **Weights:** seed, then run the input-sensitivity gate.
   - Pass: keep random weights (CLIP, BERT, T5, UMT5, Wav2Vec2).
   - Fail: overfit the contract's training batch in float32 (AdamW, lr 5e-3, clipping, ≤ 500 steps, loss < 0.05 twice), cast to fixture dtype, pass the gate (Whisper); otherwise fail the build.
   - Recipes may require overfit with a reviewed reason and target loss: the gate reads teacher-forced logits, while random [SmolLM](evidence.md#bfloat16-on-cpu) and [MusicGen Melody](evidence.md#overfit-or-random-weights-on-real-regressions) generate one answer for every prompt.
4. **Gate the build, then validate the saved fixture.**
   - **Sensitivity:** perturb each named input (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, …) by row swap, reversal, all-ones mask, channel swap, or halving. Threshold: 10× the recipe's `gate_tolerance`, else the largest manifest `atol` (currently 1e-3). Relative/platform tolerances and compared-output sensitivity remain [gaps](problems-and-opened-questions.md#3-the-sensitivity-gate-checks-the-wrong-thing).
   - **Accept blind spots with reasons** in the manifest: Wav2Vec2-base's bias-free first convolution feeds group norm, making the fixture amplitude-insensitive (5e-5; real checkpoint 3.9).
   - **Health:** the builder reloads saved weights and checks finite outputs, loss, and gradients in float32/bfloat16/float16 on its host. Contracts additionally verify SGD updates parameters on each used dtype/platform class. Gate failures block builds; contract checks validate published artifacts.
5. **Freeze and publish** a pinned test artifact outside Hub search; hardware affects rebuilds.
6. **Baseline a known-good build:** default runtime plus eager variant; discrete outputs, selected logits, loss, named-module gradients, and whether SGD updates parameters. See [platform rules](#hardware-and-dtype).

## What a fixture result tells an agent

A pass establishes reviewed behavior only on the exercised path and fixed inputs. Learned answers are fixture outputs, not checkpoint predictions. Report results separately; agents discover contracts through the real model.

## Choosing the weights

Four weight methods; SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2; identical faults/tolerances, three dtypes, Mac CPU and A10G. Detection matched across hardware, with no false alarms ([script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)). These experimental builds differ from the published fleet; MusicGen Melody was added later.

| | Random | Overfit | Pruned (2 real layers, full width) | Sliced (real tensors cut to width) |
| --- | ---: | ---: | ---: | ---: |
| Faults caught | 120/126 | **126/126** | **42/42** | 41/42 |
| Whisper | 18/24 | 24/24 | 8/8 | 7/8 |
| Smallest Whisper logit move from an input fault | 4e-5 | 1.2 | 1.9 | 1.6e-3 |
| Non-finite outputs, loss, or gradients | 0/108 | 0/108 | 0/36 | 0/36 |
| Weights | 0.4–15 MB | same | 94–196 MB | same as random |
| Build | < 0.1 s | 0.3–51 s CPU, 0.5–6 s GPU; 1/18 did not converge | ~1 s + checkpoint load | < 0.1 s + checkpoint load |
| Largest bfloat16 logit drift vs float32 | 0.09 | 0.30 | 0.97 | 0.34 |

- Overfit builds check gradient flow. Pruning needs no training but costs 8–235× more storage (SmolLM-135M, 2/30 layers: 142 MB); slicing behaves like random.
- No NaN in 288 load-and-train checks proves little at width 64; gate each build.
- In bfloat16, random discrete outputs changed for Wav2Vec2 (3/3 seeds) and T5 (2/3); overfit never changed. Selected contract fixtures had no discrete changes on A10G and passed per-dtype baselines at `1e-4`.
- Exact overfit answers alone caught 1–3 of 6–8 faults per fixture; only loss caught shuffled labels ([SmolLM/CLIP](../experiments/results/smollm_clip.json), [other families](../experiments/results/more_families.json)).

### Overfit versus random: what can we conclude?

Readable answers, sensitivity, and bug detection are separate benefits. No real overfit-only catch has been found; tested ties do not establish equal coverage. “Tied” below means both catch, both miss, or no demonstrated advantage.

| Criterion | Observed comparison and limits | Which one |
| --- | --- | --- |
| Human-readable generation | Overfit SmolLM names prompted capitals; Whisper labels tones `red`, `green`, `blue`, `yellow`. Random SmolLM repeats text; random Whisper ignores audio in answers. | **Overfit** |
| Synthetic-fault detection | Six additional Whisper faults caught (counts above); local sensitivity, not a real-bug catch rate. | **Overfit** |
| Real #48679, Melody, original/crossed prompts | Both catch changes in generated codes. | **Tied** |
| Same bug, identical text across rows | Overfit misses all four input/guidance combinations; random catches all four. Across cases: overfit 8/12, random 12/12; overfit is not uniformly superior. | **Random** |
| Real #48028, Whisper | Both catch changes in freely generated continuation tokens once controlled unequal first-segment transcripts reach left-padding. | **Tied** |
| Real #48108, Whisper, encoder-decoder assistant + speech | Neither changes tokens; the real tiny checkpoint catches it. This slim-fixture behavior remains uncovered. | **Tied** |
| Conditioning sensitivity | Guided Melody swaps/drops move 1/40 decoded code IDs for overfit, zero for random, on selected original prompts. An input control, not a real overfit-only catch; both waveform heads stay zero, so no meaningful/readable audio demonstrated. | **Overfit** |
| Model size and runtime | Twins share architecture and weight size; overfit adds no parameters. Generation length can change check cost. | **Tied** |
| Construction and upkeep | Training adds upfront cost and convergence/repair risk (timings above), depends on examples and gate. Random needs no convergence; both need sensitivity gates and reviewed baselines. | **Random** |
| Precision stability | Overfit's study outputs were stable; some random bfloat16 seeds changed. Selected contract fixtures pass separate dtype baselines. | **Overfit** |

Sources: [weight-initialization study](#choosing-the-weights), [Melody bad/fix pairs and conditioning controls](evidence.md#musicgen-melody-conditioning-controls), [Whisper left-padding replay](evidence.md#whisper-left-padded-previous-segment-prompts), and [assisted-decoding replay](evidence.md#whisper-assisted-decoding-suppression).

**Decision:** prefer overfit for readable generation or a failed random sensitivity gate; retain random where it satisfies the contract. Run one reviewed, frozen weight set per fixture in CI. Stop broad paired-weight experiments; reserve twins for specific detection or repair questions. This does not establish overfit's superiority on real regressions or justify replacing every random fixture.

## Shrinking depth by layer pattern

Fixed `num_hidden_layers = 2` can break mixed layers: 170 configs define `layer_types`; MoE also varies dense/sparse MLPs ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)).

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

**Rule:** kind = `layer_types` + dense/sparse MLP (`first_k_dense_replace`/`moe_layer_freq` or `mlp_only_layers`/`decoder_sparse_step`). Keep the first of each kind in order; re-index per-layer fields. When the config computes the kinds from the layer index instead of storing them (a `layer_types` or `layers_block_type` property: Jamba's attention period, Zamba's hybrid layers, RecurrentGemma cycling `block_types`), re-indexing can't keep them, so the fixture keeps the prefix up to each kind's first layer (RecurrentGemma 26 → 3, Jamba 32 → 5, Zamba2 54 → 7). Index-dependent cross-layer KV sharing still needs architecture rules.

## Does it scale to every architecture?

Config-only sweep of all 786 Transformers model types at `bb666bd0` ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints](../experiments/results/sweep_sources.json)): widths by field name in every sub-config, depth by layer pattern, 4 MoE experts; primary-head forward/backward with random weights and synthetic inputs.

| | Model types |
| --- | ---: |
| No usable checkpoint (none documented or on the Hub, or gated) | 123 |
| Component only (`*_vision_model`, `*_text_model`) | 81 |
| **Usage paths tried** | **582** |
| Built and ran forward and backward, all finite | **382 (66%)** |
| Failed in the forward pass: coupled size fields, inputs the model rejects | 138 |
| Failed in shrink or build: per-architecture constraints | 32 |
| Failed: no synthetic inputs (video, time series) | 22 |
| Failed: config load or backward | 8 |

- **Passes:** 155 causal LMs, 73 base models, 55 image-text-to-text, 32 image classifiers, 28 masked LMs, 20 seq2seq, 8 CTC; 16/17 #49103 models (MusicGen fails config round-trip). Median 8.3M parameters.
- **2026-10-29 rerun (314 → 382):** the most downloaded unreachable types now pass—Qwen3.5 (+MoE), Gemma 4, DeBERTa-v2, CLAP, GLM-5.3, ESM, Qwen4-Exp, DeepSeek-V3, Kimi K2.5. Fixes: a text-only head of a composite model takes its text config; zero-padded per-layer keys; `attention_head_size`, `pooler_hidden_size`, `ple_embed_dim` are widths; a 0 width (NoPE) stays 0; MLA `head_dim` follows `qk_rope_head_dim`; layer-id lists move to a kept layer of the same kind; stage-based configs keep their stage count; text-audio dual encoders get mel features. `facebook/esm-1b` fails config load on Transformers `main` (`hidden_dropout_prob: 0` is an int), so ESM sweeps ESM-2.
- **Main repair cost:** family rules for coupled fields—M-RoPE (GLM-4V, Qwen-VL), conv groups, hierarchical per-stage widths (Swin v2, PVT), backbone-tied necks, mamba-hybrid caches, EnCodec `codebook_dim` and doubling `num_filters` (Melody overrides both).
- **Input limit:** contracts use real processor outputs; 81 passing multimodal models ran text only.

About half need no model-specific work; remaining shrink fixes are per family, not checkpoint.

## Hardware and dtype

Fixture files run unchanged on A10G; baselines do not transfer ([numbers](evidence.md#on-gpu)). Baseline each platform class and user dtype; compare logits within a platform class.

## Relation to today's tiny models

Transformers' `tiny-random-*` pipeline (`utils/create_dummy_models.py`, fast tests in [#49103](https://github.com/huggingface/transformers/pull/49103)) uses unit-test configs that lose real RoPE/sliding-window/MoE settings, re-saved/resized processors, no seed or revision pin, and forward-only checks. In our study, a passing forward caught **0/126 synthetic faults**; re-saved processors hid **three real regressions**.

**Proposal:** adopt real configs, copied processors, seeds, and gates; retain the pipeline's shrinking, processor discovery, and publishing machinery.

## How the experiments were run

- **CPU:** ARM Mac, float32, two threads, torch 2.11.0, Transformers `4d536e1f`, `atol=rtol=1e-4`.
- **GPU:** one A10G, torch 2.14.0+cu130, Transformers `8625785e` (same tree), via [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml).
- **Fixtures:** seeds 17, 29, 43; pinned checkpoints in [`experiment.py`](../experiments/experiment.py); handpicked synthetic faults.

```sh
python experiments/experiment.py --real-checkpoints
python experiments/experiment.py --families bert t5 whisper wav2vec2 --max-steps 500 \
  --output experiments/results/more_families.json
python experiments/fixture_init.py    # random, overfit, pruned, sliced
python experiments/layer_patterns.py  # needs network for the three configs
python experiments/sweep.py prefetch && python experiments/sweep.py run  # every architecture
```
