# Fixtures for MICs

PR checks need models that run on CPU in seconds and exercise real integration paths. A fixture is the small model a MIC runs on CPU; see the [vocabulary](../README.md) for fixture, checkpoint, shrink and tiny. This page is the recipe [`build_fixtures.py`](../example/build_fixtures.py) implements, with reviewed recipe fields as its only per-family specialization. [Evidence from building it](#evidence-from-building-the-recipe) follows; real-regression details are in [Evidence](evidence.md).

## Recipe

In short: shrink the real config while keeping every layer kind (step 1), copy the preprocessing files unchanged (step 2), find weights the contract's inputs actually move and check them (steps 3–4), then freeze, publish and record baselines (steps 5–6). Step 0 picks the checkpoint to start from.

0. **Pick a trained source checkpoint:** the model type's documented checkpoint, else its most downloaded ungated one. Never another project's test model (a `*-testing` org, a `tiny-random`/`dummy` name, [`is_test_checkpoint`](../src/transformers_mic/hub.py)): it is untrained and its config and processor files are not the ones users load. The sweep skips them and `tmic create` refuses them. A model type whose only checkpoints are test models is not Reachable.
1. **Shrink the real config** ([`shrink()`](../experiments/sweep.py)), in every sub-config, innermost first. **A source of 20M parameters or fewer is not shrunk** ([`SMALL_SOURCE`](../example/build_fixtures.py)): the fixture is the checkpoint itself, config and trained weights unchanged, since it is no bigger than a fixture (median ~9M) and tests trained behaviour. Its `build.json` records `shrink: none`, `weights: real`.
   - **Widths:** hidden 64, intermediate 128, 4 heads, by field name.
   - **Attention style kept:** KV heads become 4 under MHA, stay 1 under MQA, 2 under GQA.
   - **MoE:** 4 experts, top-2, one expert group.
   - **Depth:** by layer pattern ([why](#shrinking-depth-by-layer-pattern)), with at least one MoE layer after leading dense layers. A layer's kind is `layer_types` + dense/sparse MLP (`first_k_dense_replace`/`moe_layer_freq` or `mlp_only_layers`/`decoder_sparse_step`) + every other categorical per-layer list (strings, booleans or 0/1 flags: `no_rope_layers`, `indexer_types`, `mlp_layer_types`; numeric lists such as per-layer head counts are widths, not kinds) + whether the layer reads another layer's KV cache (`kv_source_layers`). Keep the first of each kind in order, plus the KV source of any kept KV-sharing layer; re-index per-layer fields and point `kv_source_layers` at the kept positions. A top-k sparse attention keeps `index_topk` at most 2, below the contract's shortest sequence. When the config computes the kinds from the layer index instead of storing them (a `layer_types` or `layers_block_type` property: Jamba's attention period, Zamba's hybrid layers, RecurrentGemma cycling `block_types`), re-indexing can't keep them, so the fixture keeps the prefix up to each kind's first layer (RecurrentGemma 26 → 3, Jamba 32 → 5, Zamba2 54 → 7). Cross-layer KV sharing named by layer count (`num_kv_shared_layers`) or by source index (`kv_source_layers`) is handled; other index-dependent sharing still needs architecture rules. The builder then builds the original and shrunk stacks on the meta device and compares their layer kinds (module classes plus `layer_type`). If a kind is lost (one the config derives from the layer index: DeepSeek-V4's hash-routed layers, GLM-5.3's indexer every few MoE layers, Llama 4's `moe_layers`), it keeps the prefix up to each kind's first layer, at most 8 layers.
   - **Stage-based models** (Swin, PVT, ConvNeXt) keep their stage count at depth 1 per stage, with head counts that divide each stage width; ResNet's stem keeps stage 0's width, so its identity shortcut survives.
   - **Coupled fields follow the shrunk value** they were equal to in the original:
     - a projection that equals the text width, or projects into the language model
     - MLA `head_dim = qk_rope_head_dim`
     - Mllama's `vision_output_dim`, Granite Speech's projector `encoder_hidden_size`, BLT's global width
     - Llama 4's pixel-shuffle adapter, a merged patch width
   - **Layer-index fields** are renumbered into the shrunk depth: cross-attention layers, intermediate and deepstack indices, CLIPSeg `extract_layers`, DPT `backbone_out_indices`.
   - **RoPE:** M-RoPE sections and LongRoPE factors are resampled to the shrunk head.
   - **Dropped:** `quantization_config`, since fixtures are dense. A vanishing `initializer_range` (< 1e-3) becomes 0.02.
   - **Widths the rule does not know by name** take the model's own test config: the literal values in its `ModelTester.__init__` (`tests/models/<model>/test_modeling_<model>.py`, parsed, never imported). Only size fields the rule left at the checkpoint's value are taken (DeepSeek-V4-Flash's `index_head_dim` 128 → 16, `o_lora_rank`; OPT's `word_embed_proj_dim`); layer kinds stay with the rule; vocabulary, special tokens, input geometry and behavioural fields (attention `block_size`, chunk sizes, convolution groups) stay real. The result must still build, else the rule's config is kept, and `build.json` records the fields and the tester (`tester_sizes`). A tester can be unrepresentative when a model was added in a hurry, which is why it only fills widths.
   - **Reviewed `config_overrides`** handle missed fields (MusicGen Melody: EnCodec `num_filters`, `codebook_dim`; the latter must match the shrunk width).

   Preserve vocabulary, special tokens, image/patch/audio dimensions, RoPE type, and seeded Dropout/LayerDrop/SpecAugment. Check the result with [`tmic shrink`](cli.md#how-was-the-fixture-shrunk): it compares the layer kinds, module classes and properties the original builds with the fixture's, and lists what was lost. It also flags a shrunk fixture that keeps half or more of the source's parameters (the rule did not apply: Perceiver, DETR's ResNet backbone) and a test-checkpoint source.
   - **Keep the checkpoint's form:** tensor names, prefix, expert layout (one tensor set per expert, or fused stacks), and missing parameters, so loading runs the same converters and initializes omissions as for users (Wav2Vec2 [#49385](https://github.com/huggingface/transformers/issues/49385)). The file is always `model.safetensors`, even for a `pytorch_model.bin` source: loading's conversions read the names, and a `.bin` container tests nothing more. `build.json` records the source's file in `serialization.source_file`. A tied pair is serialized once, under the name the source uses (OPT-125m stores `lm_head.weight`).
   - **Weights files that are not in a supported format** (`.pkl`, gguf, npz, pdparams) are not copied, and copied files over 128 MiB are refused.
   - **dtype:** save in the source's default dtype, float32 if unspecified: mic-SmolLM-slim is bfloat16, others float32.
2. **Copy processors byte for byte.** Identical processor files, preserved config fields, and dtype permit sharing a fixture ([example](../example/README.md#the-parts)).
3. **Weights:** seed, then run the input-sensitivity gate.
   - A small source kept whole runs the gate on its trained weights and is never trained further; a failure means the contract's inputs do not reach the model, so the build fails.
   - Pass: keep random weights (CLIP, BERT, T5, UMT5, Wav2Vec2).
   - Fail: overfit the contract's training batch in float32 (AdamW, lr 5e-3, clipping, ≤ 500 steps, loss < 0.05 twice), cast to fixture dtype, pass the gate (Whisper); otherwise fail the build.
   - Recipes may require overfit with a reviewed reason and target loss: the gate reads teacher-forced logits, while random [SmolLM](evidence.md#bfloat16-on-cpu) and [MusicGen Melody](evidence.md#overfit-or-random-weights-on-real-regressions) generate one answer for every prompt.
4. **Gate the build, then validate the saved fixture.**
   - **Sensitivity:** perturb each named input (`input_ids`, `attention_mask`, `pixel_values`, `input_features`, …) by row swap, reversal, all-ones mask, channel swap, or halving. The outputs must move by more than 10× the recipe's `gate_tolerance`, or, without one, 10× the largest `allclose` `atol` in the manifest (1e-3 for the usual `1e-4`). Relative/platform tolerances and compared-output sensitivity remain [gaps](problems-and-opened-questions.md#3-the-sensitivity-gate-checks-the-wrong-thing).
   - **Accept blind spots with reasons** in the manifest: Wav2Vec2-base's bias-free first convolution feeds group norm, making the fixture amplitude-insensitive (5e-5; real checkpoint 3.9).
   - **Health:** the builder reloads saved weights and checks finite outputs, loss, and gradients in float32/bfloat16/float16 on its host. Contracts additionally verify SGD updates parameters on each used dtype/platform class. Gate failures block builds; contract checks validate published artifacts.
5. **Freeze and publish** the fixture to its `huggingface-mic/<name>-mic` repository, tagged `mic` and `test-artifact`, and pin the Hub commit. CI reuses that snapshot: rebuilding on other hardware gives different bytes.
6. **Baseline a known-good build:** default runtime plus eager variant; discrete outputs, selected logits, loss, named-module gradients, and whether SGD updates parameters. See [platform rules](#hardware-and-dtype).

## What a fixture result tells an agent

A pass establishes reviewed behavior only on the exercised path and fixed inputs. Learned answers are fixture outputs, not checkpoint predictions. Report results separately; agents discover contracts through the real model.

## Hardware and dtype

Fixture files run unchanged on A10G; baselines do not transfer ([numbers](evidence.md#on-gpu)). Baseline each platform class and user dtype; compare logits within a platform class.

## Evidence from building the recipe

Historical: the measurements and decisions that shaped the recipe, as measured at the time; the recipe above is current. **Fixtures work** with real configs and processor files, frozen weights, and numeric checks. The original replay caught **8/8 real regressions**, using the original processors, affected calls, and default runtime (SDPA). Later comparisons show limits, including a shared Whisper #48108 miss; no real bug has required overfit weights ([comparison](#overfit-versus-random-what-can-we-conclude)).

### Choosing the weights

We compared four ways to get fixture weights on six families (SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2), with the same injected faults and tolerances, in three dtypes, on a Mac CPU and an A10G. Detection matched across hardware, with no false alarms ([script](../experiments/fixture_init.py), [CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)). The faults are synthetic (for example zeroed attention, a detached MLP, swapped inputs or shuffled labels), not real regressions. These experimental builds differ from the published fleet, and MusicGen Melody was added later.

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

#### Overfit versus random: what can we conclude?

Readable answers, sensitivity, and bug detection are separate benefits. We have not found a real regression that only the overfit fixture catches. A tie on the bugs we tested doesn't mean the two cover the same ground. “Tied” below means both catch, both miss, or neither shows an advantage.

| Criterion | Observed comparison and limits | Which one |
| --- | --- | --- |
| Human-readable generation | Overfit SmolLM names prompted capitals; Whisper labels tones `red`, `green`, `blue`, `yellow`. Random SmolLM repeats text; random Whisper ignores audio in answers. | **Overfit** |
| Synthetic-fault detection | Six additional Whisper faults caught (counts above); local sensitivity, not a real-bug catch rate. | **Overfit** |
| Real #48679, Melody, original/crossed prompts | Both catch changes in generated codes. | **Tied** |
| Same bug, identical text across rows | Overfit misses all four input/guidance combinations; random catches all four. Across cases: overfit 8/12, random 12/12; overfit is not uniformly superior. | **Random** |
| Real #48028, Whisper | Both catch changes in freely generated continuation tokens once controlled unequal first-segment transcripts reach left-padding. | **Tied** |
| Real #48108, Whisper, encoder-decoder assistant + speech | Neither changes tokens; the real `whisper-tiny` checkpoint catches it. This fixture behavior remains uncovered. | **Tied** |
| Conditioning sensitivity | Guided Melody swaps/drops move 1/40 decoded code IDs for overfit, zero for random, on selected original prompts. An input control, not a real overfit-only catch; both waveform heads stay zero, so no meaningful/readable audio demonstrated. | **Overfit** |
| Model size and runtime | Twins share architecture and weight size; overfit adds no parameters. Generation length can change check cost. | **Tied** |
| Construction and upkeep | Training adds upfront cost and convergence/repair risk (timings above), depends on examples and gate. Random needs no convergence; both need sensitivity gates and reviewed baselines. | **Random** |
| Precision stability | Overfit's study outputs were stable; some random bfloat16 seeds changed. Selected contract fixtures pass separate dtype baselines. | **Overfit** |

Sources: [weight-initialization study](#choosing-the-weights), [Melody bad/fix pairs and conditioning controls](evidence.md#musicgen-melody-conditioning-controls), [Whisper left-padding replay](evidence.md#whisper-left-padded-previous-segment-prompts), and [assisted-decoding replay](evidence.md#whisper-assisted-decoding-suppression).

**Decision:** use overfit weights when the contract needs readable generation or random weights fail the sensitivity gate; keep random weights when they satisfy the contract. CI runs one reviewed, frozen weight set per fixture. We stopped broad paired-weight experiments and build random/overfit twins only for a specific detection or repair question. None of this shows overfit weights catch more real regressions, or justifies replacing every random fixture.

### Shrinking depth by layer pattern

Fixed `num_hidden_layers = 2` can break mixed layers: 170 configs define `layer_types`; MoE also varies dense/sparse MLPs ([script](../experiments/layer_patterns.py), [results](../experiments/results/layer_patterns.json)).

| Real config | Layers | Fixed 2 layers | By pattern |
| --- | --- | --- | --- |
| `google/gemma-3-1b-it` | 26: 5 sliding, 1 full, repeated | 2 sliding; full attention never built | sliding + full |
| `ibm-granite/granite-4.0-h-tiny` | 40: 5 mamba, 1 attention, repeated; MoE | 2 mamba; forward crashes | mamba + attention |
| `deepseek-ai/DeepSeek-V2-Lite` | 27: 1 dense, then sparse | 1 dense + 1 sparse, by luck | dense + sparse |

The rule this led to is recipe step 1 (**Depth**). Later additions: on 2026-10-31 the builder began comparing the layer kinds the modules actually build, which catches kinds no config field names. On 2026-11-02 the kind gained every categorical per-layer list and KV sharing: Mistral-Large-4 went 72 → 6 layers (dense, MoE, NoPE, shared indexer, KV sharing) where the `layer_types`-only rule kept 2, and 4 of 7 prototypes that shrank differently were rebuilt (Laguna 2 → 3 layers, GLM-5.3 7 → 3, DeepSeek-V4-Flash 5 → 4). `index_topk` went to 8, then 2: at 8, DeepSeek-V3.2's 5- and 7-token prompts still selected every key and the reshrunk baselines were unchanged.

### Does it scale to every architecture?

Config-only sweep of all 786 Transformers model types at `bb666bd0` ([script](../experiments/sweep.py), [results](../experiments/results/sweep.json), [checkpoints](../experiments/results/sweep_sources.json)): widths by field name in every sub-config, depth by layer pattern, 4 MoE experts; primary-head forward/backward with random weights and synthetic inputs.

| | Model types |
| --- | ---: |
| No usable checkpoint (none documented or on the Hub, or gated) | 123 |
| Component only (`*_vision_model`, `*_text_model`) | 81 |
| **Usage paths tried** | **582** |
| Built and ran forward and backward, all finite | **383 (66%)** |
| Failed in the forward pass: coupled size fields, inputs the model rejects | 138 |
| Failed in shrink or build: per-architecture constraints | 31 |
| Failed: no synthetic inputs (video, time series) | 22 |
| Failed: config load or backward | 8 |

- **Passes:** 155 causal LMs, 73 base models, 55 image-text-to-text, 32 image classifiers, 28 masked LMs, 20 seq2seq, 8 CTC; 16/17 #49103 models (MusicGen fails config round-trip). Median 8.3M parameters.
- **2026-10-31 rerun (382 → 383):** the layer-kind check, MQA and ResNet-stem rules; RT-DETRv2 now passes, nothing regressed.
- **2026-10-29 rerun (314 → 382):** the most downloaded unreachable types now pass—Qwen3.5 (+MoE), Gemma 4, DeBERTa-v2, CLAP, GLM-5.3, ESM, Qwen4-Exp, DeepSeek-V3, Kimi K2.5. Fixes: a text-only head of a composite model takes its text config; zero-padded per-layer keys; `attention_head_size`, `pooler_hidden_size`, `ple_embed_dim` are widths; a 0 width (NoPE) stays 0; MLA `head_dim` follows `qk_rope_head_dim`; layer-id lists move to a kept layer of the same kind; stage-based configs keep their stage count; text-audio dual encoders get mel features. `facebook/esm-1b` fails config load on Transformers `main` (`hidden_dropout_prob: 0` is an int), so ESM sweeps ESM-2.
- **Main repair cost:** family rules for coupled fields—M-RoPE (GLM-4V, Qwen-VL), conv groups, hierarchical per-stage widths (Swin v2, PVT), backbone-tied necks, mamba-hybrid caches, EnCodec `codebook_dim` and doubling `num_filters` (Melody overrides both).
- **Input limit:** contracts use real processor outputs; 81 passing multimodal models ran text only.

About half need no model-specific work; remaining shrink fixes are per family, not checkpoint.

### Relation to Transformers' tiny-random test models

Transformers' `tiny-random-*` pipeline (`utils/create_dummy_models.py`, fast tests in [#49103](https://github.com/huggingface/transformers/pull/49103)) uses unit-test configs that lose real RoPE/sliding-window/MoE settings, re-saved/resized processors, no seed or revision pin, and forward-only checks. In our study, a passing forward caught **0/126 synthetic faults**; re-saved processors hid **three real regressions**.

**Proposal:** adopt real configs, copied processors, seeds, and gates; retain the pipeline's shrinking, processor discovery, and publishing machinery.

These `tiny-random-*` models are never MIC sources ([recipe step 0](#recipe)). A real model with "tiny" in its name is a trained checkpoint and is shrunk like any other: whisper-tiny 38M → 3.6M, Jamba-tiny-dev 319M → 8.8M.

### How the experiments were run

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
