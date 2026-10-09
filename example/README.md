# Full example: MICs and the runner prototype

A runnable MIC for batched greedy text generation on three SmolLM checkpoints, in Transformers and Transformers.js, covering every part of the [proposed standard](../README.md#the-proposed-standard). Fixtures ([`hf-internal-testing/proto-*`](https://huggingface.co/hf-internal-testing?search_models=proto-)) and contracts ([`hf-internal-testing/*-integration`](https://huggingface.co/hf-internal-testing?search_models=-integration)) are published; their sources are in [`hub/`](hub). Five more contracts (CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2) run through the same [runner prototype](#runner-prototype).

## Where each piece lives

| Piece | Owner | Here |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM-135M`, `-360M`, `-1.7B` | Creator | not copied: unchanged, pinned in the fleet locks |
| `hf-internal-testing/SmolLM-integration`: contract and manifest | Hugging Face | [`hub/SmolLM-integration/`](hub/SmolLM-integration) |
| [`hf-internal-testing/proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim): fixture for 135M and 360M | Hugging Face | [`hub/proto-SmolLM-slim/`](hub/proto-SmolLM-slim) |
| [`hf-internal-testing/proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim): fixture for 1.7B | Hugging Face | [`hub/proto-SmolLM-1.7B-slim/`](hub/proto-SmolLM-1.7B-slim) |
| `onnx-community/SmolLM-135M-ONNX`, `-360M-ONNX` | Hugging Face | not copied: the conversions Transformers.js loads |
| Transformers: fleet lock and expected results | Transformers maintainers | [`transformers/tests/contracts/`](transformers/tests/contracts) |
| Transformers.js: fleet lock and expected results | Transformers.js maintainers | [`transformers.js/tests/contracts/`](transformers.js/tests/contracts) |

Tooling:

| File | What it does |
| --- | --- |
| [`run.py`](run.py) | Shared runner: fleet lock → targets (fixtures, then checkpoints) → one process → compare with reviewed baselines and per-platform tolerances |
| [`build_fixtures.py`](build_fixtures.py) | One fixture recipe for all models; fixture key; sensitivity and numeric-health gates |
| [`perturb.py`](perturb.py) | Input-sensitivity gate: perturbations per input name |
| [`patches.py`](patches.py) | Workarounds for known Transformers bugs (fleet lock `local_patches`, recorded in results) |
| [`replay.py`](replay.py) | `run.py` on a regression's bad and fixed builds |
| [`schemas/`](schemas) | Manifest, fleet lock, fixture build, baseline, tolerances, result; validated on load |

## The parts

**One contract per usage path.** [`integration.json`](hub/SmolLM-integration/integration.json) lists checkpoints, each one's fixture, and per framework: entry point, artifacts, variants, unsupported targets with reasons. Adding SmolLM-360M took a manifest entry and a fleet-lock pin (plus its ONNX conversion), no new code or fixture.

**Shared fixtures, by fixture key** (identical processor files and kept config fields; the build fails on mismatch). Each fixture: 3.2M parameters, 13 MB, overfit, tokenizer files copied.

- SmolLM-135M, -360M: key `5fe5094035d3`, one fixture.
- SmolLM-1.7B: different `tokenizer.json`, `tokenizer_config.json`, and full multi-head attention (32/32 heads vs 9/3, 15/5). Own fixture: key `8a9d62b3b3b2`, 4/4 heads.

**The contract is runnable markdown.** [`usage.md`](hub/SmolLM-integration/integrations/transformers/usage.md) uses [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md):

- Rendered: a plain SmolLM-135M example (generate, one training step); plumbing marked `# doc-builder: hide`.
- Tested: `pytest -p doc_builder.pytest_plugin usage.md` passes.
- Run: `run.py` collects the same blocks and passes each pinned target as a local path; the code stays `from_pretrained(model_id)`.

Transformers.js: [`integration.js`](hub/SmolLM-integration/integrations/transformers.js/integration.js), a plain script on the `onnx-community` conversions, no training probe.

**Expected results live with each framework**: one reviewed file per target, platform (`cpu-arm64`), and variant (`.eager`), with revision, environment, and runtime ([Transformers](transformers/tests/contracts/expectations/SmolLM/cpu-arm64), [Transformers.js](transformers.js/tests/contracts/expectations/SmolLM/cpu-arm64)). Output-changing PRs update them, not the Hub. `run.py --record` writes `<name>.candidate.json`; approval is a rename.

**Fleet locks.** [Transformers](transformers/tests/contracts/fleet.lock): contract, both fixtures, three checkpoints, dependencies. [Transformers.js](transformers.js/tests/contracts/fleet.lock): same contract, pinned independently, plus two ONNX conversions.

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

Only `README.md` and `build.json` of new fixtures are committed; build once with `$PY example/build_fixtures.py`. The runner refuses files whose hashes differ from the fleet lock. Same-machine rebuilds are byte-identical, trained Whisper included.

Results on CPU (Apple M5, Transformers `5.19.0.dev0`, torch 2.11.0, `@huggingface/transformers` 4.3.0, doc-builder `c69e96c`):

| Framework | Target | Runtime | Generated (default variant) | Default | Eager | Seconds |
| --- | --- | --- | --- | --- | --- | ---: |
| Transformers | proto-SmolLM-slim | float32 | " Paris", " Berlin", " Rome", " London" | pass | pass | 2.8 |
| Transformers | proto-SmolLM-1.7B-slim | float32 | same | pass | pass | 2.6 |
| Transformers | SmolLM-135M | bfloat16 | " Paris, the capital", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | pass | 5.8 |
| Transformers | SmolLM-360M | bfloat16 | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | pass | 11.0 |
| Transformers | SmolLM-1.7B | float32 | " Paris.\nThe", " Berlin.\nGermany", " Rome.\nThe", " London.\nThe" | pass | pass | 33.4 |
| Transformers.js | SmolLM-135M | fp32, ONNX | " Paris. It is", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | — | 1.7 |
| Transformers.js | SmolLM-360M | fp32, ONNX | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | — | 2.1 |
| Transformers.js | fixtures, SmolLM-1.7B | — | — | unsupported, with reasons | — | — |

A framework PR runs two fixture checks (~5 s) instead of three checkpoint checks (~50 s, 8.2 GB of weights).

Findings:

- **Variants need own baselines.** SmolLM-135M (bfloat16) under eager: " Paris. It is" instead of " Paris, the capital". Float32 targets agree.
- **Frameworks need own baselines.** Transformers.js (ONNX fp32) matches Transformers' bfloat16 135M on 3 of 4 prompts.
- **CPU golden outputs check WebGPU** (variants `webgpu`, `webgpu-fp16`, `webgpu-q4f16`; onnxruntime-node 1.30, Apple M5).
  - fp32: identical tokens, logits within 4e-5, deterministic over 3 runs.
  - fp16, q4f16: all logits exactly 0, one token, no error. CPU fp16 is fine (within 0.012).
  - An onnxruntime WebGPU fp16 kernel bug ([repro](../experiments/webgpu_browser/README.md)): reproduces in Chrome 154 (`shader-f16`), only for `torch.onnx` fp16 exports; the onnxruntime-genai SmolLM2 export (fused ops) is fine.
  - `--golden cpu` catches it from tokens. Only fp32 has reviewed baselines; an fp16 baseline would approve the bug. Timings recorded, not compared ([measurements](../experiments/results/webgpu_golden.json)).
- **Contracts can be wrong too.** Left-padding the training batch predicted first tokens from fully masked positions, where eager and SDPA disagree (losses 0.2 apart). The eager variant exposed it; right padding fixed it.

## Runner prototype

Six contracts, one per usage path, through `run.py` on CPU (Apple M5, Transformers `4d536e1f`, torch 2.11.0):

| Contract | Usage path | Fixture (weights, why) | Checkpoints |
| --- | --- | --- | --- |
| SmolLM | batched greedy generation | 2 fixtures, overfit (earlier build script) | 135M, 360M, 1.7B |
| CLIP | zero-shot image-caption matching | random: weakest input move 4.9e-3 | `openai/clip-vit-base-patch32` |
| BERT | fill-mask | random: weakest 2.8e-3 (padding mask) | `google-bert/bert-base-uncased` |
| T5 | text-to-text generation | T5 and UMT5 fixtures, random: weakest 1.3 and 3.1 | `google-t5/t5-small`; `google/umt5-small` unsupported (no `model_type`; Auto classes cannot load it) |
| Whisper | English transcription with `generate` | **overfit**, 200 steps, 24 s: with random weights, reversed audio moved logits 8.6e-5 | `openai/whisper-tiny` |
| Wav2Vec2 | CTC transcription | random: weakest 0.60; amplitude blind spot accepted with a reason | `facebook/wav2vec2-base-960h`, with a local workaround (below) |

**One recipe for every model** (no per-family code in `build_fixtures.py`):

1. Shrink the config with the [sweep](../experiments/sweep.py)'s generic rule; copy non-weight files unchanged.
2. Run the contract on the random fixture; record each model call's arguments by name.
3. Gate ([`perturb.py`](perturb.py)): keep random weights if every perturbation moves logits ≥10× the tolerance (1e-3); else overfit on the contract's training call.
4. Numeric health on the saved fixture: finite outputs, loss, gradients in float32, bfloat16, float16.

Only sizes change in the config (dropout, LayerDrop, SpecAugment stay on). Tensor names match the checkpoint, missing ones included (only `masked_spec_embed` in `facebook/wav2vec2-base-960h`), checked at load. Result: random everywhere except Whisper, as the synthetic-fault study predicts.

**Batching.** Runnable-markdown contracts run in the runner's process; imports are paid once:

| Run | Batched | One process per target (`--isolate`) |
| --- | ---: | ---: |
| Fixtures only, 8 targets, 6 contracts (median of 3) | **4.9 s** | 27.4 s |
| Everything, 17 targets including SmolLM-1.7B | 73 s | 90 s |

Same results in both: 16 pass, 1 unsupported (`google/umt5-small`), with the Wav2Vec2 workaround. In one process SmolLM-1.7B took 51 s vs 27 s (likely memory pressure); large checkpoints may want their own process.

**Replaying a regression** (`replay.py`, unchanged runner, reviewed baselines):

- UMT5 non-causal decoder under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135)): caught on the random fixture (log-probs off 1.22, loss 0.20); fixed build passes. Eager and T5 clean on both, as in the [earlier replay](../docs/evidence.md#historical-regression-replay).
- T5 batch decode ([#47793](https://github.com/huggingface/transformers/pull/47793)): missed; the contract doesn't call batch `tokenizer.decode()`.

**Baselines must reproduce.** `--record` reruns each target in a fresh process and refuses differing outputs, missing manifest outputs, non-finite values, or failed assertions (`all_true`, `true`). Three problems found:

- **Wrong seeding.** `torch.manual_seed` misses NumPy, used by SpecAugment; real Wav2Vec2 loss varied 73–93. Now `transformers.set_seed(0)` before loading.
- **Fragile assertion.** LayerDrop (0.1) skipped layer 0 under eager, failing "gradients reach `encoder.layers.0`". Now checks the first convolution and `lm_head`.
- **A Transformers bug ([#49385](https://github.com/huggingface/transformers/issues/49385)).** `facebook/wav2vec2-base-960h` still gave NaN or garbage loss.
  - `masked_spec_embed`, absent from the checkpoint, is left uninitialized (up to 3e35, sometimes NaN, varying per load): no `_init_weights` branch in `wav2vec2`, `wav2vec2_conformer`, `data2vec`, `wavlm`, `unispeech`, `unispeech_sat`, `sew` (HuBERT, SEW-D, SpeechT5, Wav2Vec2-BERT have one). Fine-tuning with SpecAugment (default) feeds garbage to the encoder.
  - Workaround in `local_patches`: [`patches.py`](patches.py) adds the branch (`init.uniform_`, like HuBERT) in every process, recorded per result. Baselines then reproduced and were approved; re-record after the fix.
  - The first fixture saved the tensor and disabled SpecAugment, so missed it. Now it omits the tensor and keeps SpecAugment: unpatched, it fails every run in both variants (loss off 8.1, 6.8); the real checkpoint fails differently each run.
  - Without a workaround, `known_failures` reports `xfail` and fails once the target passes again.

**Per-platform tolerances.** `expectations/<contract>/<platform>/tolerances.json` overrides manifest tolerances for one platform, with a required reason. None needed so far (see below).

### On GPU

Same contracts and pinned fixtures on an NVIDIA A10G (torch 2.14.0+cu130, Transformers `8625785e` = tree of `4d536e1f`), `contracts` mode of [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37615109270), [numbers](../experiments/results/runner.json)). Fixtures now download from the Hub; a [rerun](https://github.com/huggingface/model-integration-contracts/actions/runs/37621572296) matched the release-asset run.

- **GPU baselines hold at `1e-4`.** 16 targets × 3 variants (default, eager, bfloat16) on `cuda`, reproduced in fresh processes, passed two more runs plus eager; stored in `expectations/<contract>/cuda/`.
- **Across platforms, discrete outputs carry over, numbers don't.** GPU float32 vs CPU baselines:
  - Tokens, labels, predictions identical on 15/16; SmolLM-135M (bfloat16) gives " Paris, which is" vs " Paris, the capital".
  - Logits: up to 4e-6 on text fixtures, 4e-4–2e-3 on convolutional models (TF32).
  - Dropout training losses don't carry over (different masks).
- **bfloat16 changed no discrete output** vs GPU float32; logits moved up to 0.05 on fixtures, 2.3 on real Wav2Vec2.
- **No looser tolerances needed.** All drift above `1e-4` crosses platforms or dtypes, each with its own baseline (`cuda`, `.bfloat16`).
- **Publish fixtures, don't rebuild.** x86 rebuilds differed from the pinned M5 bytes, random ones included. The pin check also caught macOS `tar`'s `._*` files.
- **Batching on GPU:** 8 fixtures, 13.1 s in one process vs 81.1 s.

## Not real yet

- Contracts and fixtures are on `hf-internal-testing`, pinned by revision and file hashes; [`hub/`](hub) holds their sources. Edit a contract there, test with `run.py --local-contracts`, then [`publish_contracts.py`](publish_contracts.py) pins the new revision in every fleet lock using it. Transformers.js still runs its contract from the local folder (where `npm install` put its packages), checked against the pin.
- Contracts run in the runner's process; the only isolation is `--isolate` (one process per target).
- Shadow, backfill, and nightly jobs run in this repo, not Transformers CI ([backfill results](../docs/evidence.md#shadow-backfill)). Selection from changed files exists ([`select_contracts.py`](select_contracts.py), `run.py --changed-files`) but no CI job uses it.
- `build_fixtures.py` output reproduces on one machine only, so GPU and shadow jobs download the pinned fixtures from the Hub.
- Variants need a second, hidden `from_pretrained` in the runnable block; a doc-builder hook for load arguments would remove it.
- Baselines: CPU (Apple M5 and x86), CUDA (A10G), and WebGPU fp32 for Transformers.js.

## What this example offers an agent

Manifest, runnable example, fleet lock, and baselines let an agent pick a target, reproduce a tested path, and check fixed inputs, which agents lack in the [walkthroughs](../docs/evidence.md#how-agents-use-the-hub-today). Missing:

- Discovery from a Hub model page, a published last-passing record, an environment-bootstrap service ([design](../docs/design.md#agent-discovery-and-reproduction); benefit measured by the [controlled comparison](../docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic)).
- A compatibility guarantee: `transformers>=5` is not one; an agent needs the exact build and environment of the passing record.
- The custom-code loader workaround from the Microsoft Florence walkthrough.

A slim pass validates the fixture's path; a real-checkpoint baseline validates fixed inputs on that checkpoint and runtime, not arbitrary prompts.
