# End-to-end tests

Every check in this repo, where it runs, and how to start it. Results: [evidence](evidence.md), [small fixtures](fixtures.md).

| Test | Question | Runs on | Entry point |
| --- | --- | --- | --- |
| [Contract runner](#1-contract-runner) | Do the MICs pass on this build? | Mac CPU, A10G, GitHub CPU | [`example/run.py`](../example/run.py) |
| [Fixture build gates](#2-fixture-build-gates) | Is a fixture good enough to catch faults? | Mac CPU | [`example/build_fixtures.py`](../example/build_fixtures.py) |
| [Shadow job](#3-shadow-job) | What would contracts catch on real commits and PRs? | GitHub CPU (x86) | [`shadow.yml`](../.github/workflows/shadow.yml), [`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) |
| [Nightly checkpoints](#4-gpu-checks) | Do real checkpoints pass on `main`, CPU and GPU? | A10G runner | [`nightly.yml`](../.github/workflows/nightly.yml) |
| [GPU checks](#4-gpu-checks) | Do fixtures and baselines hold on CUDA? | A10G | [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) |
| [Historical regressions](#5-historical-regressions) | Would fixtures have caught shipped bugs? | Mac CPU | [`experiments/replay_case.py`](../experiments/replay_case.py), [`usage_repros.py`](../experiments/usage_repros.py), [`example/replay.py`](../example/replay.py) |
| [Synthetic faults](#6-synthetic-faults) | Which weight scheme catches injected faults? | Mac CPU, A10G | [`experiments/experiment.py`](../experiments/experiment.py), [`fixture_init.py`](../experiments/fixture_init.py) |
| [Stability](#7-stability) | Quiet on benign changes and upgrades? | Mac CPU, A10G | [`experiments/replay.py`](../experiments/replay.py) |
| [Cost](#8-cost) | What does a check cost, slim vs real? | Mac CPU, A10G | [`experiments/cost.py`](../experiments/cost.py) |
| [Architecture sweep](#9-architecture-sweep) | Does the shrink recipe run every architecture? | Mac CPU | [`experiments/sweep.py`](../experiments/sweep.py), [`layer_patterns.py`](../experiments/layer_patterns.py) |
| [WebGPU golden](#10-webgpu-golden) | Can CPU outputs check WebGPU kernels? | Mac (Node, Chrome) | [`experiments/webgpu_golden.py`](../experiments/webgpu_golden.py), [`webgpu_browser/`](../experiments/webgpu_browser) |
| [Agent walkthroughs](#11-agent-walkthroughs) | How do agents run a Hub model today? | Mac | [`agent-walkthrough/`](agent-walkthrough/README.md) |

## 1. Contract runner

One contract per usage path ([full example](../example/README.md)), run on each fixture and checkpoint, compared with [baselines](../example/transformers/tests/contracts).

| Contract | Usage path | Targets (fixture → checkpoints) |
| --- | --- | --- |
| [SmolLM](../example/hub/SmolLM-integration) | `AutoTokenizer` + `AutoModelForCausalLM.generate`, batched, left padding | [`proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) → SmolLM-135M, -360M; [`proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim) → SmolLM-1.7B |
| [CLIP](../example/hub/CLIP-integration) | `CLIPProcessor` + `CLIPModel`, image-caption matching, batched | [`proto-clip-vit-base-patch32-slim`](https://huggingface.co/hf-internal-testing/proto-clip-vit-base-patch32-slim) → openai/clip-vit-base-patch32 |
| [BERT](../example/hub/BERT-integration) | `AutoTokenizer` + `AutoModelForMaskedLM`, batched, right padding | [`proto-bert-base-uncased-slim`](https://huggingface.co/hf-internal-testing/proto-bert-base-uncased-slim) → google-bert/bert-base-uncased |
| [T5](../example/hub/T5-integration) | `AutoTokenizer` + `AutoModelForSeq2SeqLM.generate`, batched | [`proto-t5-small-slim`](https://huggingface.co/hf-internal-testing/proto-t5-small-slim) → google-t5/t5-small; [`proto-umt5-small-slim`](https://huggingface.co/hf-internal-testing/proto-umt5-small-slim) → google/umt5-small |
| [Whisper](../example/hub/Whisper-integration) | `WhisperProcessor` + `WhisperForConditionalGeneration.generate`, English, batched | [`proto-whisper-tiny-slim`](https://huggingface.co/hf-internal-testing/proto-whisper-tiny-slim) → openai/whisper-tiny |
| [WhisperSpeech](../example/hub/WhisperSpeech-integration) | Frozen speech: greedy versus encoder-decoder assisted generation | Checkpoint-only: openai/whisper-tiny; slim path declared unsupported |
| [Wav2Vec2](../example/hub/Wav2Vec2-integration) | `Wav2Vec2Processor` + `Wav2Vec2ForCTC`, greedy CTC, batched | [`proto-wav2vec2-base-960h-slim`](https://huggingface.co/hf-internal-testing/proto-wav2vec2-base-960h-slim) → facebook/wav2vec2-base-960h |
| [MusicgenMelody](../example/hub/MusicgenMelody-integration) | `AutoProcessor` + `MusicgenMelodyForConditionalGeneration.generate`, text + melody prompts, classifier-free guidance, batched | [`proto-musicgen-melody-slim`](https://huggingface.co/hf-internal-testing/proto-musicgen-melody-slim) → facebook/musicgen-melody |

Eight contracts use 9 unique fixtures and 10 unique checkpoints. Per variant, 19 checks run (Whisper tiny has two usage paths), with two declared exclusions: `google/umt5-small` has no `model_type`, and WhisperSpeech has no fixture coverage. Nightly runs 95 checks across its five platform/variant combinations, plus 10 declared exclusions.

- **Checks:** exact tokens, text, and runtime; `allclose` at `1e-4` on selected logits and training loss; finite gradients on named modules; an SGD step updates parameters. Non-finite values fail and are never recorded. A host lacking a target's [`resources`](design.md#minimum-hardware) reports it `unsupported`.
- **Variants:** Transformers `default` (users' dtype and attention), `eager`, `bfloat16`; Transformers.js (SmolLM) `default` (CPU), `webgpu`, `webgpu-fp16`, `webgpu-q4f16`.
- **Baselines:** `expectations/<contract>/<platform>/<target>[.<variant>].json`, per platform:

| Platform | Recorded on |
| --- | --- |
| `cpu-arm64` | Apple M5 |
| `cpu-x86_64` | [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml): AMD EPYC 7R32; float32 targets agree on the 7763 and 9V45 ([evidence](evidence.md#bfloat16-on-cpu)) |
| `cpu-x86_64-bf16` | bfloat16 targets (SmolLM-135M/-360M, proto-SmolLM-slim) on CPUs with `avx512_bf16`/`amx_bf16`: AMD EPYC 9V45 |
| `cuda` | A10G |

`--record` writes candidates that must reproduce in a fresh process; a reviewer renames them to approve.

```sh
python example/run.py --framework transformers                       # every contract and target, default variant
python example/run.py --framework transformers --fixtures-only       # current fleet: 9 fixtures
python example/run.py --framework transformers --contract T5 --variant eager
python example/run.py --framework transformers --isolate             # one process per target
python example/run.py --framework transformers --golden cpu-arm64    # discrete outputs vs the Mac baseline
git diff --name-only A B | python example/run.py --framework transformers --fixtures-only --changed-files -
python example/run.py --framework transformers.js
```

[`select_contracts.py`](../example/select_contracts.py): model-directory changes select contracts using it (including sub-models); shared library code and `setup.py`/`pyproject.toml` select all; other files select none. `--run-all` runs everything and records `selected_by`. Fixtures download at hash-checked [`fleet.lock`](../example/transformers/tests/contracts/fleet.lock) revisions (`--local-fixtures` uses build folders); [`patches.py`](../example/patches.py) applies the lock's workarounds (#49385, #49403, #49456).

## 2. Fixture build gates

[`build_fixtures.py`](../example/build_fixtures.py) builds each fixture from its checkpoint ([recipe](fixtures.md#recipe)) and fails unless it passes:

- **Fixture key:** byte-identical processor files and matching config fields across its checkpoints.
- **Input sensitivity** ([`perturb.py`](../example/perturb.py)): each perturbation moves logits ≥ `1e-3`, else overfit. Overfit: Whisper (gate failed); SmolLM and MusicGen Melody (reviewed choice: random weights generate the same output for every prompt).
- **Reviewed `config_overrides`** where the generic shrink misses a field (MusicGen Melody: EnCodec `num_filters`, `codebook_dim`).
- **Numeric health:** reload saved weights; finite forward, loss, gradients in float32/bfloat16/float16 on the build host. Parameter updates are checked by the contracts, not this gate.

```sh
python example/build_fixtures.py                                    # all fixtures, ~45 s
python example/build_fixtures.py --fixture proto-whisper-tiny-slim
python example/build_fixtures.py --out /tmp/rebuilt                 # rebuild and compare with the pins
python example/build_fixtures.py --out /tmp/random --keep-random    # random-weight twins, for overfit-vs-random replays
python example/publish_fixtures.py                                  # upload fixtures that match their pin
```

## 3. Shadow job

[`shadow.yml`](../.github/workflows/shadow.yml) installs a Transformers ref with the lock's pins, runs every fixture on GitHub x86, records selection, and appends `runs/<date>/<run id>-<attempt>.jsonl` to `shadow-results`: daily on `main`, or dispatched on a sha, branch, or `pr/<n>` (`pip_overrides` for old builds). [`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) checks every first-parent commit in a window, in 20 parallel chunks, with dependency overrides ([results](evidence.md#shadow-backfill)).

```sh
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=pr/49386
gh workflow run shadow-backfill.yml --repo huggingface/model-integration-contracts -f since=2026-08-17 -f until=2026-09-07
gh workflow run shadow.yml --repo huggingface/model-integration-contracts -f transformers_ref=560f36cc56 -f pip_overrides=tokenizers==0.22.2
```

## 4. GPU checks

On the Transformers ARC group `aws-g5-4xlarge-cache` (one A10G), except `cpu-baselines.yml machine=github` (GitHub-hosted).

| Workflow | What it does |
| --- | --- |
| [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) | The experiments below, in the `huggingface/transformers-all-latest-gpu` image |
| [`gpu-probe.yml`](../.github/workflows/gpu-probe.yml) | Access and environment only |
| [`nightly.yml`](../.github/workflows/nightly.yml) | Every target, nightly on Transformers `main` |
| [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) | Records and checks x86 CPU candidates for review; `machine=github` (7 GB) or `machine=gpu-host` (64 GB, CPU-only torch) |

**Dependency stability:** `gpu-checks.yml matrix=dependency-stability` keeps the known-good Transformers build, artifacts, baselines and tolerances fixed. It checks tokenizers 0.23.2, NumPy 2.2.6 and Pillow 12.3.0 individually and together, then a no-torchaudio control and torch 2.14.1 / torchvision 0.29.1 with CUDA 13. Each environment gets two fresh-process fleet passes per default/eager/bfloat16 variant. `dependency_profiles=torch-cu130` retries that environment alone; the job restores its CUDA wheel libraries before the runtime preflight. The observer saves actual outputs and numeric drift; `summarize_dependency_study.py` compares repeats. MusicGen Melody requires torchaudio, so its absence is a coverage limit, not a successful check.

```sh
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=dependency-stability
python experiments/summarize_dependency_study.py /path/to/artifact/dependency-stability --output /tmp/dependency-summary.json
```

**Wav2Vec2 normalization:** `gpu-checks.yml matrix=wav2vec2-normalization` compares NumPy 1, native NumPy 2, and an explicit arithmetic control through both targets' inference and training calls. It asserts exact recovery of processor bytes and reported outputs across default/eager/bfloat16, with two fresh-process repeats.

**Whisper speech:** `gpu-checks.yml matrix=whisper-speech` records and checks its five worker baselines, then asserts that #48108 fails on its parent and passes on its fix, through the contract runner on CPU and CUDA.

**Nightly:** lock environment with CUDA torch (2.11.0+cu128); runner CPU with CUDA hidden (`cpu-x86_64`: default, eager) and GPU (`cuda`: default, eager, bfloat16); appends to `runs/nightly/`, or writes `cuda` candidates with `mode=record`. First run: 80/80 ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37645864098)).

`gpu-checks.yml` matrices:

| `matrix` | What runs |
| --- | --- |
| `full` | Freeze fixtures and real baselines on CPU and CUDA; check on CUDA in float32 (eager, SDPA), bfloat16, float16; cost and peak GPU memory |
| `tf32` | CPU baselines on GPU in float32, cuDNN TF32 on and off |
| `init` | [`fixture_init.py`](../experiments/fixture_init.py) on GPU ([§6](#6-synthetic-faults)) |
| `contracts` | Runner on CUDA: record three variants, rerun against them, CPU and float32 goldens, batched vs isolated, informational fixture rebuild |

```sh
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=contracts
gh run download <run-id> --repo huggingface/model-integration-contracts -n gpu-checks
```

## 5. Historical regressions

Eleven regressions since May 2026 ([`cases.json`](../experiments/cases.json)) and ten benign changes, on bad and fixed builds (`git archive`) at three levels: **real** checkpoint through the affected call; **generic slim**, forward and generate ([`replay_case.py`](../experiments/replay_case.py)); **usage slim**, fixtures through the affected call with re-saved or copied processor files ([`usage_repros.py`](../experiments/usage_repros.py)). [`example/replay.py`](../example/replay.py) replays a case through the runner. Verdicts: [evidence](evidence.md#historical-regression-replay); overfit fixtures vs random twins on #48679, #48028, #48108: [evidence](evidence.md#overfit-or-random-weights-on-real-regressions).

Later targeted GPU harnesses: [`melody_conditioning.py`](../experiments/melody_conditioning.py), [`whisper_left_padding.py`](../experiments/whisper_left_padding.py), and [`whisper_assisted_suppression.py`](../experiments/whisper_assisted_suppression.py). Their GPU dispatch jobs are not part of `gpu-checks.yml`; run the scripts on a GPU host.

```sh
export REPLAY_WORKDIR=/tmp/model-integration-replay
python experiments/replay.py freeze --families text clip bert t5 whisper wav2vec2 --real --out $REPLAY_WORKDIR/frozen --build HEAD
ATTN=sdpa python experiments/replay_case.py <fix-sha>
python experiments/usage_repros.py prepare && python experiments/usage_repros.py <case> copied
```

## 6. Synthetic faults

Hand-written faults patched in at check time (wrong token, padding ignored, causal mask removed, RoPE off, modules zeroed or detached, wrong labels). [`experiment.py`](../experiments/experiment.py): random vs overfit ([1](../experiments/results/smollm_clip.json), [2](../experiments/results/more_families.json)); [`fixture_init.py`](../experiments/fixture_init.py): plus pruned, sliced, and numeric health in three dtypes ([CPU](../experiments/results/fixture_init.json), [GPU](../experiments/results/fixture_init_gpu.json)). Results: [choosing the weights](fixtures.md#choosing-the-weights).

## 7. Stability

Frozen fixtures and real baselines under benign changes and dependency upgrades (torch, tokenizers, numpy, Pillow), eager and SDPA, on CPU ([`stability.json`](../experiments/results/stability.json)) and A10G ([`gpu.json`](../experiments/results/gpu.json)). Results: [evidence](evidence.md#stability-and-cost).

## 8. Cost

[`cost.py`](../experiments/cost.py) times each phase of a check, with memory and size, slim vs real: [Mac](../experiments/results/cost.json), [A10G](../experiments/results/gpu.json), [batched vs isolated](../experiments/results/runner.json).

## 9. Architecture sweep

[`sweep.py`](../experiments/sweep.py): the generic shrink rule on 559 usage paths, offline after `prefetch`; 314 run ([results](../experiments/results/sweep.json)). [`layer_patterns.py`](../experiments/layer_patterns.py): depth by layer pattern vs fixed count ([results](../experiments/results/layer_patterns.json)).

## 10. WebGPU golden

[`webgpu_golden.py`](../experiments/webgpu_golden.py) checks the SmolLM Transformers.js contract on WebGPU (Node) against the CPU golden; [`webgpu_browser/`](../experiments/webgpu_browser) in Chrome ([results](../experiments/results/webgpu_golden.json)).

```sh
python example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu
```

## 11. Agent walkthroughs

Not a regression test: two agents run a Hub model locally ([scripts](../experiments/agent_run), [recipe and results](agent-walkthrough/README.md)).
