# End-to-end tests

Every end-to-end check this repository runs: what it exercises, where it runs, and how to start it. Results and interpretation are in [Evidence](evidence.md) and [Small fixtures](fixtures.md); the proposal they test is in the [README](../README.md).

"End to end" here means real user code (processor or tokenizer, model, public call) loaded from Hub-format files, run against a Transformers build, and compared with a reviewed baseline.

| Test | What it answers | Runs on | Entry point |
| --- | --- | --- | --- |
| [Contract runner](#1-contract-runner) | Do the six MICs still pass on this Transformers build? | Mac CPU, A10G, GitHub CPU | [`example/run.py`](../example/run.py) |
| [Fixture build gates](#2-fixture-build-gates) | Is a slim fixture good enough to catch faults? | Mac CPU | [`example/build_fixtures.py`](../example/build_fixtures.py) |
| [Shadow job](#3-shadow-job) | What would contracts have caught on real Transformers commits and PRs? | GitHub CPU (x86) | [`shadow.yml`](../.github/workflows/shadow.yml) |
| [GPU checks](#4-gpu-checks) | Do fixtures and baselines hold on CUDA, across dtypes and attention? | A10G | [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) |
| [Historical regressions](#5-historical-regressions) | Would slim fixtures have caught bugs that actually shipped? | Mac CPU | [`experiments/replay_case.py`](../experiments/replay_case.py), [`usage_repros.py`](../experiments/usage_repros.py), [`example/replay.py`](../example/replay.py) |
| [Synthetic faults](#6-synthetic-faults) | Random vs overfit vs other weight schemes, on injected faults | Mac CPU, A10G | [`experiments/experiment.py`](../experiments/experiment.py), [`fixture_init.py`](../experiments/fixture_init.py) |
| [Stability](#7-stability) | Do baselines stay quiet on benign changes and dependency upgrades? | Mac CPU, A10G | [`experiments/replay.py`](../experiments/replay.py) |
| [Cost](#8-cost) | What does one check cost, slim vs real? | Mac CPU, A10G | [`experiments/cost.py`](../experiments/cost.py) |
| [Architecture sweep](#9-architecture-sweep) | Does the shrink recipe build and run every architecture? | Mac CPU | [`experiments/sweep.py`](../experiments/sweep.py), [`layer_patterns.py`](../experiments/layer_patterns.py) |
| [WebGPU golden](#10-webgpu-golden) | Can CPU outputs check WebGPU kernels in Transformers.js? | Mac (Node, Chrome) | [`experiments/webgpu_golden.py`](../experiments/webgpu_golden.py), [`webgpu_browser/`](../experiments/webgpu_browser) |
| [Agent walkthroughs](#11-agent-walkthroughs) | How do agents get a Hub model running today? | Mac | [`agent-walkthrough/`](agent-walkthrough/README.md) |

## 1. Contract runner

The [full example](../example/README.md): one contract per usage path, written as doc-builder runnable markdown, run against each slim fixture and then each real checkpoint, and compared with baselines kept in the framework's own folder ([`example/transformers/tests/contracts/`](../example/transformers/tests/contracts)).

| Contract | Usage path | Targets (fixture → checkpoints) |
| --- | --- | --- |
| [SmolLM](../example/hub/SmolLM-integration) | `AutoTokenizer` + `AutoModelForCausalLM.generate`, batched, left padding | [`proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) → SmolLM-135M, -360M; [`proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim) → SmolLM-1.7B |
| [CLIP](../example/hub/CLIP-integration) | `CLIPProcessor` + `CLIPModel`, image-caption matching, batched | [`proto-clip-vit-base-patch32-slim`](https://huggingface.co/hf-internal-testing/proto-clip-vit-base-patch32-slim) → openai/clip-vit-base-patch32 |
| [BERT](../example/hub/BERT-integration) | `AutoTokenizer` + `AutoModelForMaskedLM`, batched, right padding | [`proto-bert-base-uncased-slim`](https://huggingface.co/hf-internal-testing/proto-bert-base-uncased-slim) → google-bert/bert-base-uncased |
| [T5](../example/hub/T5-integration) | `AutoTokenizer` + `AutoModelForSeq2SeqLM.generate`, batched | [`proto-t5-small-slim`](https://huggingface.co/hf-internal-testing/proto-t5-small-slim) → google-t5/t5-small; [`proto-umt5-small-slim`](https://huggingface.co/hf-internal-testing/proto-umt5-small-slim) → google/umt5-small |
| [Whisper](../example/hub/Whisper-integration) | `WhisperProcessor` + `WhisperForConditionalGeneration.generate`, English, batched | [`proto-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/proto-whisper-tiny-slim) → openai/whisper-tiny |
| [Wav2Vec2](../example/hub/Wav2Vec2-integration) | `Wav2Vec2Processor` + `Wav2Vec2ForCTC`, greedy CTC, batched | [`proto-wav2vec2-base-960h-slim`](https://huggingface.co/hf-internal-testing/proto-wav2vec2-base-960h-slim) → facebook/wav2vec2-base-960h |

8 fixture targets and 9 checkpoints. 16 targets run; `google/umt5-small` is reported `unsupported` because its `config.json` has no `model_type` (its fixture covers the UMT5 code path). SmolLM-1.7B has no Transformers.js conversion.

**What each target checks:**

- **Discrete outputs, exact:** generated or predicted tokens and decoded text (CLIP: image-to-text and text-to-image matches).
- **Numeric outputs, `allclose`:** selected logits or label log-probabilities, at `1e-4` per platform.
- **A training step:** the loss (`allclose`), finite gradients on named modules (`all_true`), and that an SGD step updates parameters (`true`).
- **Runtime identity, exact:** device, dtype, attention implementation actually used.
- Non-finite values and failed assertions fail the check and are never recorded as baselines.

**Variants:** `default` (the dtype and attention users get, e.g. SDPA), `eager`, `bfloat16`. Transformers.js (SmolLM only): `default` (CPU), `webgpu`, `webgpu-fp16`, `webgpu-q4f16`.

**Baselines:** one per target, platform, and variant under `expectations/<contract>/<platform>/<target>[.<variant>].json`; `cpu` (Apple M5) and `cuda` (A10G) so far. `--record` writes candidates that must reproduce in a fresh process; a reviewer renames them to approve.

```sh
python example/run.py --framework transformers                       # every contract and target, default variant
python example/run.py --framework transformers --fixtures-only       # what a framework PR would run (~5 s for all 8 on CPU)
python example/run.py --framework transformers --contract T5 --variant eager
python example/run.py --framework transformers --isolate             # one process per target
python example/run.py --framework transformers --golden cpu          # discrete outputs vs another platform's baseline
git diff --name-only A B | python example/run.py --framework transformers --fixtures-only --changed-files -
python example/run.py --framework transformers.js
```

[`select_contracts.py`](../example/select_contracts.py) maps changed Transformers files to contracts: a model directory selects the contracts whose fixtures use it, other library code selects all, files outside the library none. `--run-all` runs everything anyway and records `selected_by` per result.

Fixtures are downloaded from the Hub at the revisions [`fleet.lock`](../example/transformers/tests/contracts/fleet.lock) pins and checked against its file hashes (`--local-fixtures` uses the build folders). [`patches.py`](../example/patches.py) applies the lock's `local_patches` (currently #49385's `masked_spec_embed` workaround) and records them in each result.

## 2. Fixture build gates

[`build_fixtures.py`](../example/build_fixtures.py) builds each fixture from the real checkpoint (config shrunk by layer pattern, processor files copied byte for byte, the real tensor names) and fails the build unless it passes:

- **Fixture key:** every checkpoint assigned to a fixture has byte-identical processor files and matching config fields.
- **Input sensitivity** ([`perturb.py`](../example/perturb.py)): perturbations defined per input name (swap rows, reverse, all-ones mask, channel swap, halve) must each move the logits by at least `1e-3`. Random weights are kept if they pass; otherwise the fixture is overfit on the contract's training step (only Whisper so far).
- **Numeric health:** finite forward, loss, and gradients in float32, bfloat16, and float16.

```sh
python example/build_fixtures.py                                    # all fixtures, ~45 s
python example/build_fixtures.py --fixture proto-whisper-tiny-slim
python example/build_fixtures.py --out /tmp/rebuilt                 # rebuild and compare with the pins
python example/publish_fixtures.py                                  # upload fixtures that match their pin
```

## 3. Shadow job

[`shadow.yml`](../.github/workflows/shadow.yml) runs the fixture contracts against real Transformers commits and PRs, non-blocking and without touching the Transformers repository. It installs the ref on a GitHub-hosted x86 CPU runner with the fleet lock's pinned torch, tokenizers, and numpy, runs every fixture contract with the selection recorded, and appends one JSONL file per run to the `shadow-results` branch (`runs/<date>/<run id>-<attempt>.jsonl`, run metadata under `shadow`).

- **Scheduled:** daily on Transformers `main`; changed files are those since the previous scheduled run.
- **Dispatched:** any commit sha, branch, or `pr/<n>`; `pip_overrides` for old builds.

```sh
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=pr/49386
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=560f36cc56 -f pip_overrides=tokenizers==0.22.2
```

## 4. GPU checks

[`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) runs on the Transformers ARC group `aws-g5-4xlarge-cache` (one A10G) in the `huggingface/transformers-all-latest-gpu` image. [`gpu-probe.yml`](../.github/workflows/gpu-probe.yml) only checks access and the environment.

| `matrix` | What runs |
| --- | --- |
| `full` | Freeze fixtures and real baselines on the runner's CPU and on CUDA; check both on CUDA in float32 eager and SDPA, bfloat16, float16; per-check cost and peak GPU memory |
| `tf32` | CPU baselines checked on GPU in float32 with cuDNN TF32 on and off |
| `init` | [`fixture_init.py`](../experiments/fixture_init.py) on GPU (see [§6](#6-synthetic-faults)) |
| `contracts` | The contract runner on CUDA: record candidates in three variants (each reproduced in a fresh process), two more default runs and an eager run against them, CPU golden, bfloat16 vs float32 golden, fixtures batched vs isolated; plus an informational fixture rebuild on the runner's CPU |

```sh
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=contracts
gh run download <run-id> --repo huggingface/model-integration-contracts -n gpu-checks
```

## 5. Historical regressions

Eleven regressions fixed in Transformers since May 2026 ([`cases.json`](../experiments/cases.json)), each reproduced on the real checkpoint first, then checked on frozen slim fixtures on the bad and fixed builds (extracted with `git archive`, never checked out). Three levels:

- **Real:** the real checkpoint through the affected call.
- **Generic slim:** frozen fixtures, forward and generate only ([`replay_case.py`](../experiments/replay_case.py)).
- **Usage slim:** fixtures through the public call the bug affects ([`usage_repros.py`](../experiments/usage_repros.py)), with re-saved or byte-copied processor files.

| Case | Fix | Call path | Usage-slim verdict |
| --- | --- | --- | --- |
| clip-text-kwargs | #46195 | `CLIPProcessor(..., text_kwargs=...)` | caught (crash) |
| whisper-timestamps-batch-decode | #47997 | `processor.batch_decode(..., decode_with_timestamps=True)` | caught (crash) |
| t5-batch-decode-cleanup | #47793 | `tokenizer.decode(batch_of_ids)` | caught only with punctuation in inputs |
| legacy-additional-special-tokens | #47848 | `AutoTokenizer` on a legacy `tokenizer_config.json` | missed with re-saved files |
| olmo-stray-eos | #47988 | `tokenizer(text)` | real caught; generic slim missed (re-saved tokenizer) |
| umt5-noncausal-sdpa | #49135 | forward/generate under SDPA | caught under SDPA only |
| processor-config-helpers | #47628 | `get_image_processor_config` / `get_feature_extractor_config` | caught with copied files, missed with re-saved |
| bert-variant-clm-noncausal | #48785 | `BigBirdForCausalLM(is_decoder=True)` | caught |
| t5-tied-weights-order | #48238 | `from_pretrained` without `shared.weight` | not reproduced on t5-small |
| mms-set-target-lang | #47440 | `Wav2Vec2CTCTokenizer.set_target_lang` | not reproduced |
| clip-conversion-mapping | #46406 | `CLIPTextModelWithProjection.from_pretrained` | not reproduced |

Ten benign changes run the same way to count false alarms (none). [`example/replay.py`](../example/replay.py) replays a case through the contract runner unchanged; UMT5 #49135 is replayed that way end to end.

```sh
export REPLAY_WORKDIR=/tmp/model-integration-replay
python experiments/replay.py freeze --families text clip bert t5 whisper wav2vec2 --real --out $REPLAY_WORKDIR/frozen --build HEAD
ATTN=sdpa python experiments/replay_case.py <fix-sha>
python experiments/usage_repros.py prepare && python experiments/usage_repros.py <case> copied
```

## 6. Synthetic faults

Hand-written per-family faults (for SmolLM: wrong answer token, padding ignored, causal mask removed, RoPE disabled, attention or MLP zeroed or detached, wrong training labels), patched into the model at check time, against paired fixtures.

- [`experiment.py`](../experiments/experiment.py): frozen-random vs overfit fixtures on identical faults. 42/42 on SmolLM and CLIP; 78/84 random vs 84/84 overfit on BERT, T5, Whisper, Wav2Vec2 (Whisper's random fixture ignores its audio). Results: [`smollm_clip.json`](../experiments/results/smollm_clip.json), [`more_families.json`](../experiments/results/more_families.json).
- [`fixture_init.py`](../experiments/fixture_init.py): four weight schemes (random, overfit, pruned, sliced) on the same faults, plus numeric health in three dtypes; CPU and A10G ([`fixture_init.json`](../experiments/results/fixture_init.json), [`fixture_init_gpu.json`](../experiments/results/fixture_init_gpu.json)).

## 7. Stability

The frozen fixtures and real baselines checked against benign Transformers changes and dependency upgrades, eager and SDPA: torch 2.12.1, 2.13.0, 2.14.1; tokenizers 0.23.2; numpy 2.2.6; Pillow 12.3.0. No flags in 770 checks on CPU ([`stability.json`](../experiments/results/stability.json)). On the A10G: GPU baselines stable over three runs; CPU baselines flagged on convolutional families; bfloat16/float16 need their own baselines ([`gpu.json`](../experiments/results/gpu.json)).

## 8. Cost

[`cost.py`](../experiments/cost.py) times one check phase by phase (process start, imports, load, inference, training probe, comparison) with memory and artifact size, slim vs real, on the Mac ([`cost.json`](../experiments/results/cost.json)) and the A10G including peak GPU memory ([`gpu.json`](../experiments/results/gpu.json)). The runner's batched vs isolated timings are in [`runner.json`](../experiments/results/runner.json).

## 9. Architecture sweep

[`sweep.py`](../experiments/sweep.py) applies the generic shrink rule to every Transformers usage path in the docs (559), builds the fixture from the real config, and runs forward with synthetic inputs, one process per type, offline after `prefetch`. 314 build and run ([`sweep.json`](../experiments/results/sweep.json)). [`layer_patterns.py`](../experiments/layer_patterns.py) checks shrinking depth by layer pattern instead of to a fixed count ([`layer_patterns.json`](../experiments/results/layer_patterns.json)).

## 10. WebGPU golden

[`webgpu_golden.py`](../experiments/webgpu_golden.py) runs the SmolLM Transformers.js contract on WebGPU in Node (onnxruntime-node 1.30) and compares it with the CPU golden; [`webgpu_browser/`](../experiments/webgpu_browser) does the same in Chrome with `puppeteer-core`. fp32 matches within 4e-5; fp16 and q4f16 return all-zero logits on the onnx-community exports ([`webgpu_golden.json`](../experiments/results/webgpu_golden.json)).

```sh
python example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu
```

## 11. Agent walkthroughs

Not a regression test: a recorded task (get a Hub model running locally) given to Claude Opus and Codex, to see what an agent needs from the Hub. Scripts the agents produced are in [`experiments/agent_run/`](../experiments/agent_run); the recipe, prompts, and results are in [Agent walkthroughs](agent-walkthrough/README.md).
