# Full example: MICs and the runner prototype

A runnable MIC for batched greedy generation on three SmolLM checkpoints, in Transformers and Transformers.js, covering every part of the [proposed standard](../README.md#the-proposed-standard). Published [fixtures](https://huggingface.co/hf-internal-testing?search_models=proto-) and [contracts](https://huggingface.co/hf-internal-testing?search_models=-integration) have their sources in [`hub/`](hub). Five more contracts (CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2) use the same [runner](#runner-prototype).

## Where each piece lives

| Piece ([owners](../docs/design.md#ownership)) | Here |
| --- | --- |
| `HuggingFaceTB/SmolLM-135M`, `-360M`, `-1.7B` (creator) | pinned in the fleet locks |
| [`SmolLM-integration`](https://huggingface.co/hf-internal-testing/SmolLM-integration): contract and manifest | [`hub/SmolLM-integration/`](hub/SmolLM-integration) |
| [`proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) (135M, 360M), [`proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim) (1.7B) | [`hub/`](hub) |
| `onnx-community/SmolLM-135M-ONNX`, `-360M-ONNX`, loaded by Transformers.js | pinned in its lock |
| Fleet lock and expected results per framework | [`transformers/`](transformers/tests/contracts), [`transformers.js/`](transformers.js/tests/contracts) |

| Tool | What it does |
| --- | --- |
| [`run.py`](run.py) | Runner: lock → targets (fixtures, then checkpoints) → one process → compare with baselines |
| [`build_fixtures.py`](build_fixtures.py) | Fixture recipe, fixture key, gates |
| [`perturb.py`](perturb.py) | Input-sensitivity gate |
| [`patches.py`](patches.py) | `local_patches` workarounds for known Transformers bugs |
| [`replay.py`](replay.py) | `run.py` on a regression's bad and fixed builds |
| [`schemas/`](schemas) | JSON schemas, validated on load |

## The parts

**One contract per usage path.** [`integration.json`](hub/SmolLM-integration/integration.json) lists checkpoints, their fixtures, and per framework: entry point, artifacts, variants, unsupported targets with reasons. Adding SmolLM-360M took a manifest entry and a lock pin, no code or fixture.

**Shared fixtures, by [fixture key](../docs/design.md#grouping-checkpoints):**

- SmolLM-135M, -360M: [`proto-SmolLM-slim`](hub/proto-SmolLM-slim), key `a186080c7a70`, 3.2M parameters, bfloat16 (6.1 MB), overfit by reviewed choice (80 steps, `target_loss` 0.3; [why](../docs/evidence.md#bfloat16-on-cpu)).
- SmolLM-1.7B: different tokenizer files and full multi-head attention (32/32 heads vs 9/3, 15/5). [`proto-SmolLM-1.7B-slim`](hub/proto-SmolLM-1.7B-slim), key `8a9d62b3b3b2`, 4/4 heads, float32 (13 MB), overfit by an earlier build script.

**Runnable markdown.** [`usage.md`](hub/SmolLM-integration/integrations/transformers/usage.md) ([doc-builder blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md)) renders as a plain SmolLM-135M example (generate, one training step), passes `pytest -p doc_builder.pytest_plugin usage.md`, and is what `run.py` runs, with each pinned target passed as a local path. Transformers.js: [`integration.js`](hub/SmolLM-integration/integrations/transformers.js/integration.js), a plain script on the `onnx-community` conversions, no training probe.

**Baselines and locks per framework:** one reviewed file per target, platform, and variant ([Transformers](transformers/tests/contracts/expectations/SmolLM/cpu-arm64), [Transformers.js](transformers.js/tests/contracts/expectations/SmolLM/cpu-arm64)); `--record` writes `<name>.candidate.json`, approval is a rename. Each [fleet lock](transformers/tests/contracts/fleet.lock) pins contract, fixtures, checkpoints, and dependencies; the [Transformers.js lock](transformers.js/tests/contracts/fleet.lock) pins the same contract independently, plus two ONNX conversions.

## Run it

```sh
PY=~/Dev/transformers/.venv/bin/python          # Python with transformers, torch, accelerate
pip install git+https://github.com/huggingface/doc-builder   # runnable blocks are on main
(cd example/hub/SmolLM-integration/integrations/transformers.js && npm install)
$PY example/run.py --framework transformers                  # 2 fixtures + 3 checkpoints
$PY example/run.py --framework transformers --variant eager
$PY example/run.py --framework transformers.js
$PY example/run.py --framework transformers.js --variant webgpu                    # WebGPU baselines
$PY example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu  # tokens vs CPU golden
$PY example/run.py --framework transformers --target proto-SmolLM-slim --record
$PY example/run.py --framework transformers --fixtures-only      # what a framework PR runs
$PY example/build_fixtures.py --fixture proto-whisper-tiny-slim        # rebuild one fixture
$PY example/replay.py d7ba937671 --contract T5                    # UMT5 #49135, bad vs fixed build
```

Only fixtures' `README.md` and `build.json` are committed (`--local-fixtures` needs `build_fixtures.py` once); the runner refuses files whose hashes differ from the lock.

Results on CPU (Apple M5, Transformers `5.19.0.dev0`, torch 2.11.0, `@huggingface/transformers` 4.3.0, doc-builder `c69e96c`):

| Framework | Target | Runtime | Generated (default variant) | Default | Eager | Seconds |
| --- | --- | --- | --- | --- | --- | ---: |
| Transformers | proto-SmolLM-slim | bfloat16 | " Paris Paris Paris Paris", " Berlin Berlin Berlin of", " Rome Rome London London", " London London London London" | pass | pass | 2.8 |
| Transformers | proto-SmolLM-1.7B-slim | float32 | " Paris", " Berlin", " Rome", " London" | pass | pass | 2.6 |
| Transformers | SmolLM-135M | bfloat16 | " Paris, the capital", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | pass | 5.8 |
| Transformers | SmolLM-360M | bfloat16 | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | pass | 11.0 |
| Transformers | SmolLM-1.7B | float32 | " Paris.\nThe", " Berlin.\nGermany", " Rome.\nThe", " London.\nThe" | pass | pass | 33.4 |
| Transformers.js | SmolLM-135M | fp32, ONNX | " Paris. It is", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | — | 1.7 |
| Transformers.js | SmolLM-360M | fp32, ONNX | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | — | 2.1 |
| Transformers.js | fixtures, SmolLM-1.7B | — | — | unsupported, with reasons | — | — |

A framework PR runs two fixture checks (~5 s) instead of three checkpoint checks (~50 s, 8.2 GB of weights).

- **Variants and frameworks need own baselines.** SmolLM-135M under eager says " Paris. It is", not " Paris, the capital"; Transformers.js matches Transformers' 135M on 3 of 4 prompts.
- **CPU golden outputs check WebGPU:** fp32 matches within 4e-5; fp16 and q4f16 return all-zero logits, caught from tokens by `--golden cpu` ([evidence](../docs/evidence.md), [repro](../experiments/webgpu_browser/README.md), [numbers](../experiments/results/webgpu_golden.json)). Only fp32 has reviewed baselines: an fp16 one would approve the bug.
- **Contracts can be wrong too.** A left-padded training batch predicted first tokens from fully masked positions, where eager and SDPA disagree (losses 0.2 apart); right padding fixed it.

## Runner prototype

Seven contracts, one per usage path, through `run.py` on CPU (Apple M5, Transformers `4d536e1f`, torch 2.11.0):

| Contract | Usage path | Fixture (weights, why) | Checkpoints |
| --- | --- | --- | --- |
| SmolLM | batched greedy generation | 2 fixtures, overfit: proto-SmolLM-slim by reviewed choice (random weights repeat one answer), 1.7B by an earlier script | 135M, 360M, 1.7B |
| CLIP | zero-shot image-caption matching | random: weakest input move 4.9e-3 | `openai/clip-vit-base-patch32` |
| BERT | fill-mask | random: weakest 2.8e-3 (padding mask) | `google-bert/bert-base-uncased` |
| T5 | text-to-text generation | T5 and UMT5 fixtures, random: weakest 1.3 and 3.1 | `google-t5/t5-small`; `google/umt5-small` unsupported (no `model_type`; Auto classes cannot load it) |
| Whisper | English transcription with `generate` | **overfit**, 200 steps, 24 s: random weights moved logits 8.6e-5 on reversed audio | `openai/whisper-tiny` |
| Wav2Vec2 | CTC transcription | random: weakest 0.60; amplitude blind spot accepted with a reason | `facebook/wav2vec2-base-960h`, with a local workaround (below) |
| MusicgenMelody | text- and melody-conditioned generation with guidance | **overfit** by reviewed choice, 60 steps: random weights pass the gate (weakest 0.018) but generate the same codes for every prompt, melody or not; EnCodec widths set by `config_overrides` (59M → 3.5M parameters) | `facebook/musicgen-melody` |

`build_fixtures.py` applies one [recipe](../docs/fixtures.md#recipe) with no per-family code (a recipe may set reviewed `config_overrides` where the generic shrink rule misses a coupled or unshrunk field); the contract's own model calls feed the [`perturb.py`](perturb.py) gate and the overfit step.

**Batching** (runnable-markdown contracts share the runner's process):

| Run | Batched | One process per target (`--isolate`) |
| --- | ---: | ---: |
| Fixtures only, 8 targets, 6 contracts (median of 3) | **4.9 s** | 27.4 s |
| Everything, 17 targets including SmolLM-1.7B | 73 s | 90 s |

Both: 16 pass, 1 unsupported (`google/umt5-small`). Batched, SmolLM-1.7B took 51 s vs 27 s (likely memory pressure), so large checkpoints may want their own process.

**Replay** (`replay.py`): UMT5 [#49135](https://github.com/huggingface/transformers/pull/49135) caught on the random fixture (log-probs off 1.22, loss 0.20), fixed build passes; T5 batch decode [#47793](https://github.com/huggingface/transformers/pull/47793) missed until the call was [added](../docs/improving-contracts.md).

**Reproducible baselines** (`--record` reruns in a fresh process) found:

- **Wrong seeding:** `torch.manual_seed` misses NumPy (SpecAugment); real Wav2Vec2 loss varied 73–93. Now `transformers.set_seed(0)` before loading.
- **Fragile assertion:** LayerDrop skipped layer 0 under eager; gradients are now checked on the first convolution and `lm_head`.
- **Transformers bug [#49385](https://github.com/huggingface/transformers/issues/49385):** `masked_spec_embed`, absent from `facebook/wav2vec2-base-960h`, is left uninitialized (up to 3e35, sometimes NaN) in `wav2vec2`, `wav2vec2_conformer`, `data2vec`, `wavlm`, `unispeech`, `unispeech_sat`, `sew`, so SpecAugment fine-tuning reads garbage. [`patches.py`](patches.py) adds the missing init (`init.uniform_`, like HuBERT) via `local_patches`. The fixture omits the tensor and keeps SpecAugment, so unpatched it fails every run (loss off 8.1, 6.8).

### On GPU

A10G (torch 2.14.0+cu130, Transformers `8625785e`), `contracts` mode of [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37615109270), [rerun](https://github.com/huggingface/model-integration-contracts/actions/runs/37621572296), [numbers](../experiments/results/runner.json)):

- **`cuda` baselines hold at `1e-4`:** 16 targets × 3 variants (default, eager, bfloat16), reproduced in fresh processes and over two more runs.
- **GPU float32 vs CPU baselines:** tokens and labels identical on 15/16 (SmolLM-135M: " Paris, which is" vs " Paris, the capital"); logits within 4e-6 on text fixtures, 4e-4–2e-3 on convolutional models (TF32); dropout losses differ.
- **bfloat16 vs GPU float32:** no discrete output changed; logits up to 0.05 on fixtures, 2.3 on real Wav2Vec2.
- So no looser tolerance is needed: all drift above `1e-4` crosses platforms or dtypes. Batching and publish-don't-rebuild: [evidence](../docs/evidence.md#summary).

## Not real yet

- Contract edits: change [`hub/`](hub), test with `run.py --local-contracts`, then [`publish_contracts.py`](publish_contracts.py) pins the new revision in every lock. Transformers.js runs its contract from the local folder (for its `npm` packages), checked against the pin.
- Isolation is only `--isolate` (one process per target).
- CI jobs run in this repo, not Transformers ([results](../docs/evidence.md#shadow-backfill)); [`select_contracts.py`](select_contracts.py) selection is recorded but gates nothing.
- `build_fixtures.py` output reproduces on one machine only, so CI downloads the pinned fixtures.
- Variants need a second, hidden `from_pretrained`; a doc-builder load-argument hook would remove it.

## What this example offers an agent

Manifest, example, lock, and baselines let an agent pick a target, reproduce a tested path, and check fixed inputs, which the [walkthroughs](../docs/evidence.md#how-agents-use-the-hub-today) lacked. Missing: [discovery](../docs/design.md#agent-discovery-and-reproduction) from the model page, a published last-passing record with its exact build and environment (`transformers>=5` is not one), and the Florence custom-code workaround. Benefit: [controlled comparison](../docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic).
