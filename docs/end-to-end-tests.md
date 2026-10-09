# End-to-end tests

Every end-to-end check in this repo (real user code on Hub-format files, run against a Transformers build, compared with a reviewed baseline), where it runs, and how to start it. Results: [Evidence](evidence.md), [Small fixtures](fixtures.md).

| Test | What it answers | Runs on | Entry point |
| --- | --- | --- | --- |
| [Contract runner](#1-contract-runner) | Do the six MICs still pass on this Transformers build? | Mac CPU, A10G, GitHub CPU | [`example/run.py`](../example/run.py) |
| [Fixture build gates](#2-fixture-build-gates) | Is a slim fixture good enough to catch faults? | Mac CPU | [`example/build_fixtures.py`](../example/build_fixtures.py) |
| [Shadow job](#3-shadow-job) | What would contracts have caught on real Transformers commits and PRs? | GitHub CPU (x86) | [`shadow.yml`](../.github/workflows/shadow.yml), [`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) |
| [Nightly checkpoints](#4-gpu-checks) | Do real checkpoints still pass on `main`, CPU and GPU? | A10G runner (CPU and GPU) | [`nightly.yml`](../.github/workflows/nightly.yml) |
| [GPU checks](#4-gpu-checks) | Do fixtures and baselines hold on CUDA, across dtypes and attention? | A10G | [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) |
| [Historical regressions](#5-historical-regressions) | Would slim fixtures have caught bugs that shipped? | Mac CPU | [`experiments/replay_case.py`](../experiments/replay_case.py), [`usage_repros.py`](../experiments/usage_repros.py), [`example/replay.py`](../example/replay.py) |
| [Synthetic faults](#6-synthetic-faults) | Random vs overfit vs other weight schemes, on injected faults | Mac CPU, A10G | [`experiments/experiment.py`](../experiments/experiment.py), [`fixture_init.py`](../experiments/fixture_init.py) |
| [Stability](#7-stability) | Do baselines stay quiet on benign changes and dependency upgrades? | Mac CPU, A10G | [`experiments/replay.py`](../experiments/replay.py) |
| [Cost](#8-cost) | What does one check cost, slim vs real? | Mac CPU, A10G | [`experiments/cost.py`](../experiments/cost.py) |
| [Architecture sweep](#9-architecture-sweep) | Does the shrink recipe build and run every architecture? | Mac CPU | [`experiments/sweep.py`](../experiments/sweep.py), [`layer_patterns.py`](../experiments/layer_patterns.py) |
| [WebGPU golden](#10-webgpu-golden) | Can CPU outputs check WebGPU kernels in Transformers.js? | Mac (Node, Chrome) | [`experiments/webgpu_golden.py`](../experiments/webgpu_golden.py), [`webgpu_browser/`](../experiments/webgpu_browser) |
| [Agent walkthroughs](#11-agent-walkthroughs) | How do agents get a Hub model running today? | Mac | [`agent-walkthrough/`](agent-walkthrough/README.md) |

## 1. Contract runner

The [full example](../example/README.md): one runnable-markdown contract per usage path, run on each fixture and checkpoint, compared with baselines in [`example/transformers/tests/contracts/`](../example/transformers/tests/contracts).

| Contract | Usage path | Targets (fixture → checkpoints) |
| --- | --- | --- |
| [SmolLM](../example/hub/SmolLM-integration) | `AutoTokenizer` + `AutoModelForCausalLM.generate`, batched, left padding | [`proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) → SmolLM-135M, -360M; [`proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim) → SmolLM-1.7B |
| [CLIP](../example/hub/CLIP-integration) | `CLIPProcessor` + `CLIPModel`, image-caption matching, batched | [`proto-clip-vit-base-patch32-slim`](https://huggingface.co/hf-internal-testing/proto-clip-vit-base-patch32-slim) → openai/clip-vit-base-patch32 |
| [BERT](../example/hub/BERT-integration) | `AutoTokenizer` + `AutoModelForMaskedLM`, batched, right padding | [`proto-bert-base-uncased-slim`](https://huggingface.co/hf-internal-testing/proto-bert-base-uncased-slim) → google-bert/bert-base-uncased |
| [T5](../example/hub/T5-integration) | `AutoTokenizer` + `AutoModelForSeq2SeqLM.generate`, batched | [`proto-t5-small-slim`](https://huggingface.co/hf-internal-testing/proto-t5-small-slim) → google-t5/t5-small; [`proto-umt5-small-slim`](https://huggingface.co/hf-internal-testing/proto-umt5-small-slim) → google/umt5-small |
| [Whisper](../example/hub/Whisper-integration) | `WhisperProcessor` + `WhisperForConditionalGeneration.generate`, English, batched | [`proto-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/proto-whisper-tiny-slim) → openai/whisper-tiny |
| [Wav2Vec2](../example/hub/Wav2Vec2-integration) | `Wav2Vec2Processor` + `Wav2Vec2ForCTC`, greedy CTC, batched | [`proto-wav2vec2-base-960h-slim`](https://huggingface.co/hf-internal-testing/proto-wav2vec2-base-960h-slim) → facebook/wav2vec2-base-960h |

8 fixtures + 9 checkpoints; 16 run. `google/umt5-small` is `unsupported` (no `model_type` in its config; its fixture covers UMT5). SmolLM-1.7B has no Transformers.js conversion.

**Checks per target:**

- **Exact:** generated/predicted tokens and decoded text (CLIP: matches both ways); runtime device, dtype, attention.
- **`allclose` at `1e-4` per platform:** selected logits or label log-probabilities; training loss.
- **Training:** finite gradients on named modules (`all_true`), SGD step updates parameters (`true`).
- Non-finite values and failed assertions fail and are never recorded.

**Minimum hardware:** a target's `resources` (devices, GPU count, GPU memory, RAM, disk) are checked first; a host lacking them reports `unsupported` with the reason, so GPU-only checkpoints skip on CPU while their fixtures run.

**Variants:**

- Transformers: `default` (the dtype and attention users get, e.g. SDPA), `eager`, `bfloat16`.
- Transformers.js (SmolLM only): `default` (CPU), `webgpu`, `webgpu-fp16`, `webgpu-q4f16`.

**Baselines:** one per target, platform, and variant, under `expectations/<contract>/<platform>/<target>[.<variant>].json`.

| Platform | Recorded on |
| --- | --- |
| `cpu-arm64` | Apple M5 |
| `cpu-x86_64` | GitHub runners via [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml): AMD EPYC 7R32, matches an EPYC 7763 on all 30 shared targets |
| `cpu-x86_64-bf16` | bfloat16 targets on x86 CPUs with native bfloat16: AMD EPYC 9V45 (GitHub runner) |
| `cuda` | A10G |

- CPU baselines are per architecture: Whisper differs by more than `1e-4`. `--golden cpu` means this machine's CPU platform.
- x86 bfloat16: SmolLM-135M/-360M and proto-SmolLM-slim (bfloat16 by default) use `cpu-x86_64-bf16` on CPUs with `avx512_bf16` or `amx_bf16`: the real checkpoints' logits move up to 0.25 there, and 135M eager generates different tokens ([evidence](evidence.md#bfloat16-on-cpu)).
- `--record` writes candidates that must reproduce in a fresh process; a reviewer renames them to approve.

```sh
python example/run.py --framework transformers                       # every contract and target, default variant
python example/run.py --framework transformers --fixtures-only       # what a framework PR would run (~5 s for all 8 on CPU)
python example/run.py --framework transformers --contract T5 --variant eager
python example/run.py --framework transformers --isolate             # one process per target
python example/run.py --framework transformers --golden cpu          # discrete outputs vs another platform's baseline
git diff --name-only A B | python example/run.py --framework transformers --fixtures-only --changed-files -
python example/run.py --framework transformers.js
```

**Selection** ([`select_contracts.py`](../example/select_contracts.py)): a changed model directory selects the contracts whose fixtures use it, other library code selects all, files outside the library select none. `--run-all` runs everything and records `selected_by`.

**Fixtures and patches:** fixtures download at [`fleet.lock`](../example/transformers/tests/contracts/fleet.lock) revisions and are hash-checked (`--local-fixtures` uses build folders). [`patches.py`](../example/patches.py) applies and records the lock's `local_patches` (#49385's `masked_spec_embed` workaround, Whisper's tensor-batch decode workaround for #49403).

## 2. Fixture build gates

[`build_fixtures.py`](../example/build_fixtures.py) builds each fixture from its checkpoint ([recipe](fixtures.md#recipe)); the build fails unless it passes:

- **Fixture key:** byte-identical processor files and matching config fields across its checkpoints.
- **Input sensitivity** ([`perturb.py`](../example/perturb.py)): each per-input perturbation moves logits ≥ `1e-3`; otherwise overfit (only Whisper so far).
- **Numeric health:** finite forward, loss, gradients in float32, bfloat16, float16.

```sh
python example/build_fixtures.py                                    # all fixtures, ~45 s
python example/build_fixtures.py --fixture proto-whisper-tiny-slim
python example/build_fixtures.py --out /tmp/rebuilt                 # rebuild and compare with the pins
python example/publish_fixtures.py                                  # upload fixtures that match their pin
```

## 3. Shadow job

[`shadow.yml`](../.github/workflows/shadow.yml) (non-blocking) installs a Transformers ref on a GitHub x86 runner with the lock's torch, tokenizers, numpy; runs every fixture contract, recording selection; and appends `runs/<date>/<run id>-<attempt>.jsonl` to the `shadow-results` branch.

- **Scheduled:** daily on `main`, changed files since the previous run.
- **Dispatched:** any sha, branch, or `pr/<n>`; `pip_overrides` for old builds.
- **Backfill** ([`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml)): every first-parent `main` commit in a date window, 10 parallel jobs, one install per job, dependency overrides from each commit's requirements. [Results](evidence.md#shadow-backfill).

```sh
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=pr/49386
gh workflow run shadow-backfill.yml --repo huggingface/model-integration-contracts -f since=2026-08-17 -f until=2026-09-07
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=560f36cc56 -f pip_overrides=tokenizers==0.22.2
```

## 4. GPU checks

All on the Transformers ARC group `aws-g5-4xlarge-cache` (one A10G).

| Workflow | What it does |
| --- | --- |
| [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) | The experiments below, in the `huggingface/transformers-all-latest-gpu` image |
| [`gpu-probe.yml`](../.github/workflows/gpu-probe.yml) | Checks access and environment only |
| [`nightly.yml`](../.github/workflows/nightly.yml) | Every target (real checkpoints and fixtures), nightly on Transformers `main` |
| [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) | Records `cpu-x86_64` candidates, checks against them, uploads them for review |

- **Nightly:** lock environment with CUDA torch; runner CPU with CUDA hidden (`cpu-x86_64`: default, eager) and GPU (`cuda`: default, eager, bfloat16); appends to `runs/nightly/`, or writes `cuda` candidates with `mode=record`. `cuda` baselines use torch 2.11.0+cu128; against the image's 2.14, only CLIP and Wav2Vec2 numbers moved, no discrete output. First run on `2b727b6d86` passed all 80 checks, including CPU with CUDA torch against CPU-torch baselines ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37645864098)).
- **CPU baselines:** `machine=github` (7 GB) or `machine=gpu-host` (64 GB, CPU-only torch).

`gpu-checks.yml` matrices:

| `matrix` | What runs |
| --- | --- |
| `full` | Freeze fixtures and real baselines on the runner's CPU and on CUDA; check both on CUDA in float32 eager and SDPA, bfloat16, float16; per-check cost and peak GPU memory |
| `tf32` | CPU baselines checked on GPU in float32 with cuDNN TF32 on and off |
| `init` | [`fixture_init.py`](../experiments/fixture_init.py) on GPU (see [§6](#6-synthetic-faults)) |
| `contracts` | Contract runner on CUDA: candidates recorded in three variants (each reproduced in a fresh process), two more default runs and an eager run against them, CPU golden, bfloat16 vs float32 golden, fixtures batched vs isolated; plus an informational fixture rebuild on the runner's CPU |

```sh
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=contracts
gh run download <run-id> --repo huggingface/model-integration-contracts -n gpu-checks
```

## 5. Historical regressions

Eleven regressions fixed since May 2026 ([`cases.json`](../experiments/cases.json)), each reproduced on the real checkpoint, then checked on bad and fixed builds (`git archive`, never checked out) at three levels: **real** checkpoint through the affected call; **generic slim**, forward and generate only ([`replay_case.py`](../experiments/replay_case.py)); **usage slim**, fixtures through the affected call with re-saved or copied processor files ([`usage_repros.py`](../experiments/usage_repros.py)).

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

Ten benign changes, same method: no false alarm. [`example/replay.py`](../example/replay.py) replays a case through the unchanged contract runner (done end to end for UMT5 #49135).

```sh
export REPLAY_WORKDIR=/tmp/model-integration-replay
python experiments/replay.py freeze --families text clip bert t5 whisper wav2vec2 --real --out $REPLAY_WORKDIR/frozen --build HEAD
ATTN=sdpa python experiments/replay_case.py <fix-sha>
python experiments/usage_repros.py prepare && python experiments/usage_repros.py <case> copied
```

## 6. Synthetic faults

Hand-written faults patched in at check time (SmolLM: wrong answer token, padding ignored, causal mask removed, RoPE off, attention or MLP zeroed or detached, wrong labels).

- [`experiment.py`](../experiments/experiment.py): random vs overfit ([1](../experiments/results/smollm_clip.json), [2](../experiments/results/more_families.json)). SmolLM, CLIP: 42/42. BERT, T5, Whisper, Wav2Vec2: 78/84 random vs 84/84 overfit (Whisper's random fixture ignores audio).
- [`fixture_init.py`](../experiments/fixture_init.py): random, overfit, pruned, sliced, plus numeric health in three dtypes, CPU and A10G ([CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)).

## 7. Stability

Frozen fixtures and real baselines under benign changes and dependency upgrades, eager and SDPA.

- **CPU:** torch 2.12.1, 2.13.0, 2.14.1; tokenizers 0.23.2; numpy 2.2.6; Pillow 12.3.0. No flags in 770 checks ([`stability.json`](../experiments/results/stability.json)).
- **A10G** ([`gpu.json`](../experiments/results/gpu.json)): GPU baselines stable over three runs. CPU baselines flagged on convolutional families. bfloat16/float16 need their own baselines.

## 8. Cost

[`cost.py`](../experiments/cost.py) times each phase of one check (start, imports, load, inference, training, comparison), with memory and size, slim vs real: Mac [`cost.json`](../experiments/results/cost.json), A10G [`gpu.json`](../experiments/results/gpu.json), batched vs isolated [`runner.json`](../experiments/results/runner.json).

## 9. Architecture sweep

- [`sweep.py`](../experiments/sweep.py): generic shrink rule on all 559 usage paths, synthetic inputs, one process per type, offline after `prefetch`; 314 run ([`sweep.json`](../experiments/results/sweep.json)).
- [`layer_patterns.py`](../experiments/layer_patterns.py): depth by layer pattern vs fixed count ([`layer_patterns.json`](../experiments/results/layer_patterns.json)).

## 10. WebGPU golden

[`webgpu_golden.py`](../experiments/webgpu_golden.py) checks the SmolLM Transformers.js contract on WebGPU (Node, onnxruntime-node 1.30) against the CPU golden; [`webgpu_browser/`](../experiments/webgpu_browser) does the same in Chrome via `puppeteer-core`. fp32 matches within 4e-5; fp16 and q4f16 return all-zero logits on the onnx-community exports ([`webgpu_golden.json`](../experiments/results/webgpu_golden.json)).

```sh
python example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu
```

## 11. Agent walkthroughs

Not a regression test: Claude Opus and Codex get the same task (run a Hub model locally). Scripts in [`experiments/agent_run/`](../experiments/agent_run); recipe and results in [Agent walkthroughs](agent-walkthrough/README.md).
