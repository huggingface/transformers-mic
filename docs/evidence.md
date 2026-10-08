# MICs: evidence so far

What the local experiments, the historical regression replay, and the shadow backfill over Transformers `main` show. Companion to the [proposed standard](../README.md#the-proposed-standard). Fixture experiments are detailed in [Small fixtures](fixtures.md); mechanics in the [technical design](design.md).

## Summary

**Detection**

- **Numerical checks make small fixtures sensitive.** On synthetic faults across six families, overfit fixtures caught 126 of 126 cases and random ones 120; a passing forward pass caught none. All random misses were Whisper, whose random fixture ignores its audio.
- **The contract must call what users call.** In the [replay](#historical-regression-replay), generic forward/generate checks missed every processor and tokenizer bug. The same fixtures caught them through the affected public API.
- **Five months of `main`, frozen contracts: 1 caught, 4 missed, 1 false alarm.** The UMT5 regression failed from its introducing commit. The four misses were calls no contract made ([shadow backfill](#shadow-backfill)).
- **With those four calls added, all five are caught**, with no new false alarm and no added cost. The new Whisper call also found a bug still on `main`, [#49403](https://github.com/huggingface/transformers/issues/49403) ([details](#after-adding-the-missed-calls)).
- **The runner prototype found a real bug.** Seven audio models leave `masked_spec_embed` uninitialized when a checkpoint lacks it ([#49385](https://github.com/huggingface/transformers/issues/49385), fix [#49386](https://github.com/huggingface/transformers/pull/49386)). The slim Wav2Vec2 fixture catches it on every run ([runner prototype](../example/README.md#runner-prototype)).
- **Golden outputs catch silent backend bugs.** Transformers.js on WebGPU (onnxruntime-node 1.30, Apple M5) returned all-zero logits in fp16 and q4f16, with no error; fp32 matched. Chrome 154 reproduces it. It is an onnxruntime WebGPU kernel bug triggered by the `onnx-community` exports ([repro](../experiments/webgpu_browser/README.md)).

**Fixtures**

- **Real weights are not needed.** Across random, overfit, pruned, and sliced weights, six families, three dtypes, CPU and A10G: 288 load-and-train checks, all finite. Pruned real layers caught 42 of 42 but are 8–235× larger ([details](fixtures.md#choosing-the-weights)).
- **Random weights pass the sensitivity gate on five of six families.** Only Whisper needed overfitting, matching the synthetic-fault results. UMT5 #49135 was caught on a random fixture.
- **The recipe reaches about half of all architectures without per-model work:** 314 of 559 usage paths built and ran, all finite. Most failures are coupled size fields needing one rule per family ([details](fixtures.md#does-it-scale-to-every-architecture)).
- **Fixtures must be published, not rebuilt.** Rebuilt on x86, every fixture's weights differed from the Apple M5 build. They are now published as `hf-internal-testing/proto-*`, pinned by revision and file hashes.

**Baselines and cost**

- **No false alarms on ten benign changes** from the replay window, under eager and SDPA.
- **Fixtures carry across hardware; baselines do not.** CPU baselines held on the A10G only for families without convolutions. bfloat16 and float16 exceed tolerance against float32 everywhere ([on GPU](#on-gpu)).
- **Per-platform baselines remove the need for looser tolerances.** On an A10G, 16 targets × 3 variants reproduced at `1e-4`. Larger drift appeared only across platforms or dtypes ([numbers](../experiments/results/runner.json)).
- **Batching makes slim checks cheap.** Eight fixture checks: 4.9 s in one process vs 27.4 s with one process each on the Mac; 13.1 s vs 81.1 s on the GPU runner.

**Scope.** These are small experiments: ten families, eleven candidate regressions (eight reproducible), ten benign changes, five months of `main` on six contracts. CPU only for the replay and backfill; one GPU for stability and cost. Cases with commits and verdicts: [cases.json](../experiments/cases.json).

## How agents use the Hub today

Two [recorded walkthroughs](agent-walkthrough/README.md), by [Claude Code](agent-walkthrough/claude-opus.md) and [Codex](agent-walkthrough/codex.md), used the same prompt: choose a small captioning model, then run `microsoft/Florence-2-base` for captioning and detection. Both ran locally on an ARM Mac, without a MIC. They show current friction; they do not measure what a MIC improves.

| Observation | What a MIC covering that path could provide |
| --- | --- |
| Claude chose Qwen3.5-0.8B, whose card had no in-process snippet. Codex chose BLIP, then reused cached Florence weights when downloads failed. | A tested local example and resource requirements help choose a runnable path. Availability and cache state still matter. |
| Codex inferred Transformers 4.41.2 from the config's `transformers_version: 4.41.0.dev0` and cached packages. | A last-passing record (model revision, framework build, dependencies, hardware) replaces a guess from serialization metadata. |
| Both hit Florence's `flash_attn` import failure on 4.41.2. Codex filtered that optional dependency from the import scanner for CPU. | Running that CPU path in CI would expose the failure, so a fix or explicit workaround variant could be tested. A manifest does not repair the loader. |
| `transformersInfo` suggested a native loader while `config.auto_map` selected custom code (Claude). Codex saw the same conflict in generated Hub usage; its API requests failed. | Docs and generated snippets can use the same tested entry point for the exact checkpoint and environment. |
| Both judged outputs visually. Codex also drew boxes and checked coordinate bounds. Neither compared with a reviewed reference. | Fixed sample inputs and reviewed real-checkpoint outputs give a reproducible check. |

**Same repo, different outcomes.** Claude produced Task B with `florence-community/Florence-2-base` after the Microsoft path failed. Codex ran the requested Microsoft revision `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` with Transformers 4.41.2, torch 2.10.0, CPU float32, eager attention, and the scanner workaround. So a working combination existed but was undocumented. That workaround does not show the unmodified card example works.

**Reference checks have a bounded meaning.** Matching a reviewed baseline shows an installation reproduces known behavior on fixed inputs, not that every new output is correct. A slim fixture checks its own integration; agents need the real-checkpoint result for the model they use.

**Not all costs are MIC-addressable.** Codex took about 6m 28s for Task A and 6m 10s for Task B (overlapping windows). DNS failures, sandbox limits, a `uv` panic, and report mistakes were separate problems. Caches, model choice, and tool accounting differed, so the logs support no speed ranking.

**What the runs do not show:** the value of overfitting, CI savings, ownership choices, or maintenance cost. Agent benefit also depends on [discoverability](design.md#agent-discovery-and-reproduction). The initial design defers remote-code models, so it does not cover this Florence path.

**Next:** a [controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic) with comparable cache and network, fresh sessions, measuring completion on the requested repo, time to verified result, adaptation, failures, and incorrect success claims.

## Stability and cost

### On CPU

**Dependency upgrades: no flags in 770 checks** ([stability.json](../experiments/results/stability.json)). Fixtures and real-checkpoint baselines were frozen with torch 2.11.0, tokenizers 0.23.1, numpy 1.26.4, Pillow 12.2.0. They were re-checked changing one dependency at a time (torch 2.12.1, 2.13.0, 2.14.1; tokenizers 0.23.2; numpy 2.2.6; Pillow 12.3.0), under eager and SDPA. Max drift: 1.3e-5 on slim fixtures, 8.4e-5 on real checkpoints (Wav2Vec2 with numpy 2.x, UMT5 under SDPA). Real checkpoints sit closer to the `1e-4` limit.

**Cost of one check.** Fresh process, three runs, CPU, warm OS file cache, no downloads ([cost.py](../experiments/cost.py), [cost.json](../experiments/results/cost.json)):

| Family | Slim work | Real work | Slim weights | Real weights |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 0.27 s | 0.25 s | 13 MB | 538 MB |
| CLIP | 0.40 s | 0.41 s | 14 MB | 605 MB |
| BERT | 0.09 s | 0.14 s | 8 MB | 440 MB |
| T5 | 0.14 s | 0.18 s | 9 MB | 242 MB |
| Whisper | 0.57 s | 1.26 s | 15 MB | 151 MB |
| Wav2Vec2 | 0.28 s | 0.47 s | 0.4 MB | 378 MB |
| UMT5 | 0.76 s | 1.15 s | 66 MB | 1,227 MB |

"Work" is input preparation, load, first inference, comparison, and (slim only) the training probe. Each process also spent about 2.4 s importing torch and Transformers.

- **Process start dominates.** Run many contracts per process.
- **Little time saved locally against small real checkpoints.** These are each family's smallest, already cached. Inference was 5–20× slower on real weights; loading and preprocessing were similar.
- **The savings are transfer and size.** Slim weights are 10–1,300× smaller. The slow tests for the same eight families load about 144 GB (two Llama 7–8B, three OLMo up to 7B, Whisper large variants, twelve Wav2Vec2 checkpoints), mostly not runnable on CPU.
- **Memory:** 0.5–1.3 GB peak RSS for slim checks (mostly the framework), 0.7–2 GB for these real checkpoints.

### On GPU

One run on `aws-g5-4xlarge-cache` (one A10G, x86 CPU, torch 2.14.0+cu130, CUDA 13.0), Transformers `8625785e`, the same tree as the CPU study's `4d536e1f` ([workflow](../.github/workflows/gpu-checks.yml), [gpu.json](../experiments/results/gpu.json)). Six families, random and overfit fixtures on three seeds plus each real checkpoint: 42 rows. Baselines frozen in float32 on the runner's CPU and on the GPU, then checked on the GPU at `1e-4`.

| Baseline recorded on | Checked on GPU as | Flagged (of 42) | Generated outputs changed | Max drift |
| --- | --- | ---: | ---: | ---: |
| GPU, float32 | float32, SDPA (three runs) | 0 | 0 | 6.6e-5 |
| GPU, float32 | float32, eager | 0 | 0 | 0 |
| CPU, float32 | float32, SDPA or eager | 17 | 0 | 2.9e-2 |
| CPU, float32 | float32, SDPA, cuDNN TF32 off (4 families) | 0 of 28 | 0 | 1.9e-4 |
| CPU or GPU, float32 | bfloat16, SDPA | 42 | 6 | 2.3 |
| CPU or GPU, float32 | float16, SDPA | 42 | 1 | 0.30 |

- **Fixtures are not tied to CPU.** They ran unchanged on the GPU, and GPU baselines held at `1e-4` across three runs. Real checkpoints use most of the margin (6.6e-5, Wav2Vec2); slim fixtures stay near 1e-5.
- **CPU baselines carry over only without convolutions.** SmolLM, BERT, and T5 stayed within 5e-5. CLIP, Whisper, and Wav2Vec2 drifted 1e-3 to 3e-2, slim and real alike.
- **The cause is TF32.** PyTorch lets cuDNN run convolutions in TF32 by default on Ampere and newer. With `torch.backends.cudnn.allow_tf32 = False`, 0 of 28 were flagged ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37512092438)). Users get TF32, so keep the default and record GPU baselines. Generated tokens and labels never changed in float32.
- **Lower precision needs its own baseline.** Against float32, every bfloat16 and float16 row was flagged. Generated outputs changed for real SmolLM (both dtypes), two T5 fixtures, and three Wav2Vec2 fixtures (bfloat16). A bfloat16 run needs a bfloat16 baseline on the same platform class. The [full example](../example/README.md) hit the same issue with eager vs SDPA on CPU.

**Cost on GPU.** float32, SDPA, three fresh processes per check, weights cached ([gpu.json](../experiments/results/gpu.json), `cost`):

| Family | Slim work | Real work | Peak GPU memory, slim | Peak GPU memory, real (float32 / bfloat16) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 1.0 s | 1.5 s | 125 MB | 564 / 295 MB |
| CLIP | 1.5 s | 2.0 s | 134 MB | 631 / 338 MB |
| BERT | 0.6 s | 1.2 s | 103 MB | 456 / 251 MB |
| T5 | 1.5 s | 1.8 s | 105 MB | 266 / 150 MB |
| Whisper | 1.4 s | 1.4 s | 193 MB | 302 / 168 MB |
| Wav2Vec2 | 1.1 s | 1.5 s | 76 MB | 443 / 259 MB |

Imports took about 4.9 s per process, twice the Mac's, so batching matters more here. Warm GPU inference: 3–150 ms. Slim fixtures need under 200 MB of GPU memory, so many can share a GPU.

Not measured yet: CPU-only CI runners, environment setup, Hub downloads, cold file cache, other GPUs, multi-GPU.

## Shadow backfill

[`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) ran today's frozen fixture contracts on every first-parent commit of Transformers `main` from 2026-05-02 to 2026-10-07: 1,468 commits × 8 fixtures = 11,744 results, no runner error.

How it ran:

- 10–20 parallel jobs on GitHub-hosted x86 runners, against the `cpu-x86_64` baselines.
- Each job installed the fleet lock's environment once (Transformers editable), then checked out commit after commit, about 15 s each.
- Where a commit's dependency table rejects a pin, the newest accepted version is used: `tokenizers` 0.22 on the 893 commits before 2026-08-17, with no effect on any result.
- Each result records the commit's selection (`selected_by`, from its diff). Results are on the `shadow-results` branch (`runs/backfill/`).

Every change in outcome, in first-parent order:

| Fixture | Commits | What differs from today's baseline | Starts | Ends | Verdict |
| --- | --- | --- | --- | --- | --- |
| `proto-t5-small-slim`, `proto-umt5-small-slim` | 761 | `runtime`: eager instead of SDPA | window start | [#47014](https://github.com/huggingface/transformers/pull/47014) (`560f36cc56`, 2026-07-30), "Enable SDPA for T5" | Intended: T5-family SDPA did not exist before #47014 |
| `proto-umt5-small-slim` | 597 | `label_logprobs` off by 1.22, training loss by 0.20 | #47014 (`560f36cc56`) | [#49135](https://github.com/huggingface/transformers/pull/49135) (`d7ba937671`, 2026-09-28) | **Regression caught at its introducing commit** (UMT5 decoder not causal under SDPA) |
| `proto-wav2vec2-base-960h-slim` | 513 | training loss off by 0.39 | window start | [#46259](https://github.com/huggingface/transformers/pull/46259) (`f196737b0c`, 2026-07-03) | **Benign change flagged:** `super()._init_weights()` added, shifting the load-time RNG stream |

Every other result passed on every commit.

- **The one covered regression was caught on its introducing commit.** #47014 enabled SDPA for T5 and made UMT5's decoder non-causal. It touched shared code, so it selected every contract. A reviewer would have seen T5's intended runtime change next to UMT5's outputs moving by 1.22, two months before the fix.
- **One false alarm in five months.** #46259 made the parent `_init_weights` draw random numbers for every Wav2Vec2 module first. The fixture lacks `masked_spec_embed`, like the real checkpoint, so that tensor's random init, and the SpecAugment training loss, follow the RNG stream. Nothing user-visible changed. This is the price of catching [#49385](https://github.com/huggingface/transformers/issues/49385): one baseline update.
- **Four regressions missed, all for the same reason: no contract made the call.** CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195), fixed 2026-05-29), processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628), 2026-08-04), Whisper timestamp batch decoding ([#47997](https://github.com/huggingface/transformers/pull/47997), 2026-08-16), T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793), 2026-09-01). All four were caught in the [replay](#historical-regression-replay) once the check made the call (T5 only with punctuated inputs). Detection depends on usage paths, not fixtures.
- **Selection missed nothing.** All three outcome-changing commits selected the affected contract. Overall, selection skipped every contract on 768 commits (52%), selected all on 679 (46%), and some on 21 (1%).
- **No other baseline drift** across three CPU models (AMD EPYC 7763, 9V45, 9V74, with and without native bfloat16) and two `tokenizers` versions.

### After adding the missed calls

Following the [improvement loop](improving-contracts.md), the four calls were added:

- CLIP: per-modality `text_kwargs` and `get_image_processor_config`.
- Whisper: `generate(return_timestamps=True)` with `batch_decode(..., decode_with_timestamps=True)`, and `get_feature_extractor_config`.
- Wav2Vec2: the same config read.
- T5: `decode` on a punctuated batch.

Every existing baseline stayed identical on all three platforms (98 candidates reviewed). The same window was backfilled again (1,470 commits, [run](https://github.com/huggingface/model-integration-contracts/actions/runs/37664740099)):

| Regression | Fixture result while present | Ends at |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | `proto-clip-vit-base-patch32-slim` errors (`'list' object has no attribute 'shape'`), 206 commits | its fix, `420c80aad0` |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | CLIP `image_processor_config_keys` and Wav2Vec2 `feature_extractor_config_keys` come back empty, 595 and 801 commits | its fix, `c62820a73c` |
| Whisper timestamp decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)) | `proto-whisper-tiny-slim` errors (`'>=' not supported between instances of 'list' and 'int'`), 890 commits | its fix, `64b6595f0c` |
| T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)) | `proto-t5-small-slim` and `proto-umt5-small-slim`: `decoded_roundtrip` keeps the spaces before punctuation, 1,067 commits | its fix, `4ff98bf236` |
| UMT5 causal mask ([#49135](https://github.com/huggingface/transformers/pull/49135)) | unchanged: caught from its introducing commit #47014 | its fix, `d7ba937671` |

- **All five caught, each failing exactly until its fix.** Four were introduced before the window, so only UMT5 shows detection at the introducing commit.
- **No new false alarm.** The only benign failure is still #46259.
- **No added cost:** about 13 s per commit for 8 fixtures, before and after.
- **Post-hoc.** These checks were written knowing the bugs. They show the loop closes gaps, not the rate on unknown bugs. One call did find an unknown bug: Whisper's `batch_decode` still fails on the tensor `generate` returns ([#49403](https://github.com/huggingface/transformers/issues/49403)), worked around by a local patch until fixed.

## Historical regression replay

Documented Transformers regressions fixed since May 2026, replayed against real bugs rather than synthetic faults ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)).

Method:

1. Freeze random and overfit slim fixtures, plus real-checkpoint baselines, on a known-good build (`main` at `4d536e1f`).
2. Extract the bad build (the fix's parent) and the fixed build.
3. Run the frozen checks on both without regenerating anything. Caught = bad build flagged, fixed build clean.
4. Separately, run the affected public call on the real checkpoint and the fixture under both builds.

Builds before 2026-08-17 need `tokenizers` 0.22.x, and builds before mid-June an older `kernels`, so they ran with a dependency overlay. The fleet lock must pin the dependency environment, not just model revisions.

| Regression (fix PR) | Real checkpoint | Generic slim checks | Slim fixture, affected public call |
| --- | --- | --- | --- |
| CLIP processor drops `return_tensors` when `text_kwargs` is passed ([#46195](https://github.com/huggingface/transformers/pull/46195)) | crash | missed | crash, caught |
| Whisper batch `decode_with_timestamps` ([#47997](https://github.com/huggingface/transformers/pull/47997)) | crash | missed | crash, caught |
| T5 batch `decode` ignores `clean_up_tokenization_spaces` ([#47793](https://github.com/huggingface/transformers/pull/47793)) | caught | missed | caught only with punctuation in the inputs |
| `additional_special_tokens` lost with legacy tokenizer config ([#47848](https://github.com/huggingface/transformers/pull/47848)) | caught | missed | missed: re-saved tokenizer hides it |
| OLMo tokenizer appends a stray EOS ([#47988](https://github.com/huggingface/transformers/pull/47988)) | caught | missed: re-saved tokenizer hides it | — |
| UMT5 decoder self-attention not causal under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135)) | caught under SDPA only | missed under eager; caught under SDPA on all 6 fixtures (logit delta 1.1–15) | caught under SDPA: future-token leak 2.9–15 vs 2e-6 |
| Processor config helpers return `{}` for `preprocessor_config.json`-only repos ([#47628](https://github.com/huggingface/transformers/pull/47628)) | caught | — | caught with copied processor files; missed with re-saved files |
| BigBird/RemBERT/RoFormer causal LM attends to future tokens ([#48785](https://github.com/huggingface/transformers/pull/48785)) | caught: future-token leak 16.8 vs 5e-5 | — | caught: leak 0.0098 vs 2e-7, logits differ by ~2e-3 |
| T5 tied-weights order ([#48238](https://github.com/huggingface/transformers/pull/48238)) | not reproduced with `t5-small` | clean | — |
| MMS `set_target_lang` leaves a stale added-token entry ([#47440](https://github.com/huggingface/transformers/pull/47440)) | not reproduced: internal state only, identical ids and text | — | not reproduced |
| CLIP conversion mapping applied at the wrong nesting level ([#46406](https://github.com/huggingface/transformers/pull/46406)) | not reproduced with `CLIPTextModelWithProjection` | — | not reproduced |

- **Generic checks are not enough.** Forward plus greedy generate caught only UMT5, and only under the default attention. The other bugs live in processors, tokenizers, and decoding helpers.
- **The usage path catches them, and the fixture is enough.** Whenever the contract made the affected call, the fixture caught the bug as the real checkpoint did.
- **Fixture packaging matters.** Three loading regressions were hidden by processor files re-saved with a fixed build. Copying the original files caught the processor-config case on all three families tested.
- **Runtime defaults matter.** The UMT5 bug exists only under SDPA, the default. Eager-pinned checks missed it everywhere.
- **Choose attention at load time.** On `main`, `set_attn_implementation("sdpa")` after loading a T5-family model updates the top-level config but silently leaves encoder and decoder on eager ([#49372](https://github.com/huggingface/transformers/issues/49372)). The first SDPA pass used it and never exercised SDPA. Contracts should set attention at load and record the one in use.

**False alarms: none.** 10 benign changes from the same window touching the same paths (RoPE, greedy decoding, image processing, masking, attention refactors, auto-generated inits): 20 builds × 42 frozen checks, clean under eager and under SDPA chosen at load time. That includes the Wav2Vec2 move to SDPA dispatch and a continuous-batching attention refactor. The May 2026 build has no T5 SDPA, so that variant counts as unsupported, not a detection.

**Bugs are mostly in shared code.** Documented regressions inside one family are rare in five months; most hit shared processor, tokenizer, and attention code. This supports organizing contracts around public usage paths, not per-model forward checks.

Sample size: eleven candidates (three not reproducible on standard checkpoints), ten benign changes, CPU only, no GPU-specific case.
