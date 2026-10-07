# Small fixtures for MICs

Companion to the [MICs proposed standard](../README.md#the-proposed-standard).

PR-level contract checks need a model small enough to run on CPU in seconds that still exercises the real integration path. This document gives the recipe and the local experiments behind it.

## TL;DR

**Conclusion.** Small fixtures work if they are built from the real config and processor files, frozen, and checked numerically. On six families (SmolLM, CLIP, BERT, T5, Whisper, Wav2Vec2), on CPU and one A10G:

- A passing forward pass caught **0** synthetic faults. Recorded logits, loss, and targeted gradients caught **126/126** with overfit weights and **120/126** with random weights. All six misses were Whisper, whose random fixture ignores its audio. There were no false alarms.
- Slim fixtures caught all 8 reproducible historical regressions, once the original processor files were copied and the contract called the affected public API under the default runtime (SDPA).
- No fixture produced a NaN: 288 load-and-train checks in float32, bfloat16, and float16. Real weights are not needed. Pruned real layers also fix Whisper but are 8–235× larger, and sliced real tensors behave like random ones.
- Fixtures carry across hardware; baselines do not. Record baselines per platform class (CPU, GPU) and per dtype.

**Proposed recipe:** shrink the real config by layer pattern, seed, overfit, gate on input sensitivity and numeric health, freeze, and record baselines per platform and dtype.

**Next moves:**

1. **Redirect the tiny-model work in Transformers to this recipe** (proposed new direction, see [below](#relation-to-todays-tiny-models)): pause the smoke-only tests in [#49103](https://github.com/huggingface/transformers/pull/49103), and change `create_dummy_models.py` to build from real configs, copy processor files, overfit from a seed, and run the gates.
2. **Run it across the #49103 model list** to measure the shrink-rule success rate, starting with layer-pattern, MoE, nested multimodal, and Jamba-style configs. Retire the smoke-only tests once the rate is acceptable.
3. **Write training examples per usage path.** Overfitting needs a few labeled inputs and a loss for each task; this is the main new cost of the recipe.
4. **Publish one fixture per fixture key** as `hf-internal-testing/<name>-slim`, kept out of model discovery, and pin it in the fleet lock.
5. **Calibrate tolerances** on the CPU-only CI runners and the other GPU types, and record bfloat16 baselines.
6. **Measure upkeep in the pilot:** shrink-rule fixes, non-converging overfits, and baseline reviews.

## Recipe

1. **Real structure, small weights.** Start from the real config. Shrink width (hidden 64, intermediate 128, 4 heads in the experiments) and depth [by layer pattern](#shrinking-depth-by-layer-pattern). Keep vocabulary, special tokens, image/patch/audio dimensions, RoPE settings, and MoE structure. Disable dropout and SpecAugment.
2. **Real processor files, byte for byte.** Copy them; never re-save them. One fixture serves every checkpoint with the same fixture key: identical processor files and identical config fields wherever shrinking keeps them ([example](../example/README.md#the-parts)).
3. **Weights:** initialize from a fixed seed, then overfit on a few contrasting examples (AdamW, lr 5e-3, gradient clipping, ≤ 500 steps, until answers are correct and loss < 0.05 twice in a row). If training does not converge, fail the build visibly. Fall back to pruned real layers if their size is acceptable, or to the random fixture if it passes the input-sensitivity gate.
4. **Gates, on the saved fixture:**
   - **Input sensitivity:** perturbing or swapping inputs must move logits well beyond tolerance.
   - **Numeric health:** outputs, loss, and gradients must be finite, and a training step must update parameters, in every dtype and on every platform class the fixture will be checked on.

   A failed gate is a fixture build failure, visible as missing coverage, not a contract failure.
5. **Freeze and publish** at a pinned revision, marked as a test artifact and kept out of Hub search. Training and hardware only matter again on a rebuild. Overfit convergence depends on hardware: a different Wav2Vec2 seed failed on CPU and on the A10G.
6. **Record baselines** on a known-good build under the default runtime (for example SDPA, with eager as a variant). Record discrete outputs, selected logits, loss, gradients on named modules, and whether an SGD step updates parameters, separately per platform class and dtype.

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
```
