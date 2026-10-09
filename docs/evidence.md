# MICs: evidence so far

Results from local experiments, the regression replay, and the shadow backfill over Transformers `main`. Companion to the [proposed standard](../README.md#the-proposed-standard); fixture experiments in [Small fixtures](fixtures.md), mechanics in the [technical design](design.md).

## Summary

**Detection**

- **Numerical checks make small fixtures sensitive.** Synthetic faults: overfit caught 126/126, random 120 (misses all Whisper), a bare forward pass none.
- **Contracts must call what users call.** Generic checks missed every processor and tokenizer bug; the affected API call caught them ([replay](#historical-regression-replay)).
- **Five months of `main`: 1 caught, 4 missed, 1 false alarm**; the misses were calls no contract made ([backfill](#shadow-backfill)). With those calls added, all five caught, no new false alarm, plus a live bug ([#49403](https://github.com/huggingface/transformers/issues/49403)) ([details](#after-adding-the-missed-calls)).
- **The scheduled runs caught a regression the day it merged:** #48282 (2026-10-08) made `generate()` raise on every original Whisper checkpoint (`suppress_tokens` in `config.json` read as a user change to the model config). The next nightly and shadow runs failed Whisper on every target, platform, and dtype. Bisected to the commit and reported as [#49456](https://github.com/huggingface/transformers/issues/49456); a `local_patches` workaround keeps the other Whisper checks running. The PR's CI could not see it ([why](https://github.com/huggingface/transformers/issues/49456#issuecomment-6076843277)): Whisper's model tests build configs without `suppress_tokens`; the fast ASR pipeline tests load `openai/whisper-tiny` but went through the pipeline, which passed its own `generation_config` and so skipped the check (they started failing hours later, when #49269 removed that, pointing at the wrong PR); and the slow Whisper tests that call `generate()` were not in the PR's `run-slow` list. The contract calls `model.generate()` directly on the checkpoint's config, as users do.
- **The runner found a real bug:** seven audio models leave `masked_spec_embed` uninitialized ([#49385](https://github.com/huggingface/transformers/issues/49385), fix [#49386](https://github.com/huggingface/transformers/pull/49386)); the slim Wav2Vec2 fixture catches it every run ([runner](../example/README.md#runner-prototype)).
- **Golden outputs catch silent backend bugs:** Transformers.js on WebGPU (onnxruntime-node 1.30, Apple M5; Chrome 154) returned all-zero fp16/q4f16 logits with no error, an onnxruntime kernel bug triggered by `onnx-community` exports ([repro](../experiments/webgpu_browser/README.md)).

**Fixtures**

- **Real weights are not needed.** 288 load-and-train checks across weight kinds, dtypes, CPU and A10G, all finite; pruned real layers caught 42/42 at 8–235× the size ([details](fixtures.md#choosing-the-weights)).
- **Random weights pass the sensitivity gate on five of six families**; only Whisper needed overfitting. UMT5 #49135 was caught on a random fixture.
- **The recipe covers ~half of architectures unaided:** 314 of 559 usage paths built and ran, all finite; most failures are coupled size fields ([details](fixtures.md#does-it-scale-to-every-architecture)).
- **Publish fixtures, don't rebuild them:** rebuilt on x86, every fixture's weights differed from the M5 build. Now `hf-internal-testing/proto-*`, pinned by revision and hashes.

**Baselines and cost**

- **No false alarms on ten benign changes**, eager and SDPA.
- **Fixtures carry across hardware; baselines don't** ([GPU](#on-gpu)). Per platform and dtype, 16 targets × 3 variants reproduced at `1e-4` on A10G, so no looser tolerance is needed ([numbers](../experiments/results/runner.json)).
- **Batching is the cost lever:** eight fixture checks take 4.9 s in one process vs 27.4 s one-per-process (Mac); 13.1 s vs 81.1 s (GPU runner).

**Scope:** ten families, eleven candidate regressions (eight reproducible), ten benign changes, five months on six contracts; CPU only for replay and backfill, one GPU for stability and cost. Cases: [cases.json](../experiments/cases.json).

## How agents use the Hub today

Two [walkthroughs](agent-walkthrough/README.md) ([Claude Code](agent-walkthrough/claude-opus.md), [Codex](agent-walkthrough/codex.md)), same prompt, ARM Mac, no MIC: pick a small captioning model, then run `microsoft/Florence-2-base` for captioning and detection. They show friction, not a measured MIC gain.

| Observation | What a MIC on that path could provide |
| --- | --- |
| Claude chose Qwen3.5-0.8B (no in-process snippet on its card). Codex chose BLIP, then reused cached Florence weights when downloads failed. | A tested local example and resource requirements. Availability and cache still matter. |
| Codex inferred Transformers 4.41.2 from `transformers_version: 4.41.0.dev0` and cached packages. | A last-passing record (revision, build, dependencies, hardware) instead of a guess. |
| Both hit Florence's `flash_attn` import failure on 4.41.2; Codex filtered it from the import scanner for CPU. | CI on that path exposes the failure, so a fix or workaround variant can be tested. A manifest doesn't repair the loader. |
| `transformersInfo` suggested a native loader while `config.auto_map` chose custom code (Claude); Codex saw the same conflict and its API requests failed. | Docs and snippets share one tested entry point per checkpoint and environment. |
| Both judged output visually; Codex also drew boxes and checked bounds. Neither used a reviewed reference. | Fixed inputs and reviewed real-checkpoint outputs. |

- **Same repo, different outcomes.** Claude finished Task B on `florence-community/Florence-2-base`. Codex ran Microsoft's revision `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` with Transformers 4.41.2, torch 2.10.0, CPU float32, eager, plus the scanner workaround. A working combination existed but was undocumented; the unmodified card example is not shown to work.
- **Reference checks are bounded.** A baseline match shows known behavior on fixed inputs, not correctness on new ones. Agents need the real-checkpoint result, not the slim one.
- **Not all costs are MIC-addressable.** Codex: ~6m 28s (Task A), ~6m 10s (Task B), overlapping. DNS, sandbox, a `uv` panic, and report mistakes were separate. Differing caches and accounting rule out a speed ranking.
- **Not shown:** overfitting value, CI savings, ownership, maintenance cost. Benefit also needs [discoverability](design.md#agent-discovery-and-reproduction). Remote-code models (this Florence path) are deferred.
- **Next:** a [controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic): matched cache and network, fresh sessions; measure completion on the requested repo, time to verified result, adaptation, failures, false success claims.

## Stability and cost

### On CPU

**Dependency upgrades: 0 flags in 770 checks** ([stability.json](../experiments/results/stability.json)). Frozen with torch 2.11.0, tokenizers 0.23.1, numpy 1.26.4, Pillow 12.2.0; re-checked one change at a time (torch 2.12.1/2.13.0/2.14.1, tokenizers 0.23.2, numpy 2.2.6, Pillow 12.3.0), eager and SDPA. Max drift 1.3e-5 slim, 8.4e-5 real (Wav2Vec2 with numpy 2.x, UMT5 SDPA): real checkpoints sit closer to `1e-4`.

**Cost per check:** fresh process, three runs, warm file cache, no downloads ([cost.py](../experiments/cost.py), [cost.json](../experiments/results/cost.json)). "Work" = input prep, load, first inference, comparison, and (slim) training probe; plus ~2.4 s import per process.

| Family | Slim work | Real work | Slim weights | Real weights |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 0.27 s | 0.25 s | 13 MB | 538 MB |
| CLIP | 0.40 s | 0.41 s | 14 MB | 605 MB |
| BERT | 0.09 s | 0.14 s | 8 MB | 440 MB |
| T5 | 0.14 s | 0.18 s | 9 MB | 242 MB |
| Whisper | 0.57 s | 1.26 s | 15 MB | 151 MB |
| Wav2Vec2 | 0.28 s | 0.47 s | 0.4 MB | 378 MB |
| UMT5 | 0.76 s | 1.15 s | 66 MB | 1,227 MB |

- **Process start dominates:** run many contracts per process.
- **Little local time saved** against these smallest, cached checkpoints: real inference 5–20× slower, loading similar.
- **Savings are transfer and size:** slim weights 10–1,300× smaller. Slow tests for these eight families load ~144 GB (two Llama 7–8B, three OLMo ≤7B, Whisper large, twelve Wav2Vec2), mostly not CPU-runnable.
- **Memory:** 0.5–1.3 GB peak RSS slim (mostly framework), 0.7–2 GB real.

### On GPU

One run on `aws-g5-4xlarge-cache` (A10G, torch 2.14.0+cu130, CUDA 13.0), Transformers `8625785e` (same tree as CPU's `4d536e1f`) ([workflow](../.github/workflows/gpu-checks.yml), [gpu.json](../experiments/results/gpu.json)). 42 rows: six families, random and overfit fixtures on three seeds, plus real checkpoints. float32 baselines from the runner's CPU and GPU, checked on GPU at `1e-4`.

| Baseline recorded on | Checked on GPU as | Flagged (of 42) | Generated outputs changed | Max drift |
| --- | --- | ---: | ---: | ---: |
| GPU, float32 | float32, SDPA (three runs) | 0 | 0 | 6.6e-5 |
| GPU, float32 | float32, eager | 0 | 0 | 0 |
| CPU, float32 | float32, SDPA or eager | 17 | 0 | 2.9e-2 |
| CPU, float32 | float32, SDPA, cuDNN TF32 off (4 families) | 0 of 28 | 0 | 1.9e-4 |
| CPU or GPU, float32 | bfloat16, SDPA | 42 | 6 | 2.3 |
| CPU or GPU, float32 | float16, SDPA | 42 | 1 | 0.30 |

- **Fixtures run unchanged on GPU**; real checkpoints use most of the margin (6.6e-5, Wav2Vec2), slim stays near 1e-5.
- **CPU baselines carry over only without convolutions:** SmolLM, BERT, T5 within 5e-5; CLIP, Whisper, Wav2Vec2 drift 1e-3–3e-2, slim and real alike.
- **The cause is TF32,** which cuDNN uses for convolutions by default on Ampere+. With `torch.backends.cudnn.allow_tf32 = False`: 0 of 28 flagged ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37512092438)). Users get TF32, so keep it and record GPU baselines. float32 tokens and labels never changed.
- **Lower precision needs its own baseline.** Generated outputs changed for real SmolLM (both dtypes), two T5 and three Wav2Vec2 fixtures (bfloat16). The [full example](../example/README.md) hit the same with eager vs SDPA on CPU.

**GPU cost:** float32, SDPA, three fresh processes, weights cached ([gpu.json](../experiments/results/gpu.json), `cost`).

| Family | Slim work | Real work | Peak GPU memory, slim | Peak GPU memory, real (float32 / bfloat16) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 1.0 s | 1.5 s | 125 MB | 564 / 295 MB |
| CLIP | 1.5 s | 2.0 s | 134 MB | 631 / 338 MB |
| BERT | 0.6 s | 1.2 s | 103 MB | 456 / 251 MB |
| T5 | 1.5 s | 1.8 s | 105 MB | 266 / 150 MB |
| Whisper | 1.4 s | 1.4 s | 193 MB | 302 / 168 MB |
| Wav2Vec2 | 1.1 s | 1.5 s | 76 MB | 443 / 259 MB |

Imports ~4.9 s (2× the Mac), so batching matters more. Warm inference 3–150 ms. Slim fixtures need <200 MB, so many share a GPU.

Not measured: CPU-only CI runners, setup, Hub downloads, cold cache, other GPUs, multi-GPU.

## Shadow backfill

[`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) ran today's frozen fixture contracts on every first-parent `main` commit, 2026-05-02 to 2026-10-07: 1,468 × 8 = 11,744 results, no runner error.

- 10–20 parallel jobs, GitHub-hosted x86, `cpu-x86_64` baselines; fleet-lock environment installed once per job (Transformers editable), ~15 s per commit.
- Where a commit rejects a pin, its newest accepted version is used: `tokenizers` 0.22 on the 893 commits before 2026-08-17, no effect on results.
- Each result records its selection (`selected_by`); all on `shadow-results` (`runs/backfill/`).

Every change in outcome (all other results passed on every commit):

| Fixture | Commits | What differs from today's baseline | Starts | Ends | Verdict |
| --- | --- | --- | --- | --- | --- |
| `proto-t5-small-slim`, `proto-umt5-small-slim` | 761 | `runtime`: eager instead of SDPA | window start | [#47014](https://github.com/huggingface/transformers/pull/47014) (`560f36cc56`, 2026-07-30), "Enable SDPA for T5" | Intended: T5-family SDPA did not exist before #47014 |
| `proto-umt5-small-slim` | 597 | `label_logprobs` off by 1.22, training loss by 0.20 | #47014 (`560f36cc56`) | [#49135](https://github.com/huggingface/transformers/pull/49135) (`d7ba937671`, 2026-09-28) | **Regression caught at its introducing commit** (UMT5 decoder not causal under SDPA) |
| `proto-wav2vec2-base-960h-slim` | 513 | training loss off by 0.39 | window start | [#46259](https://github.com/huggingface/transformers/pull/46259) (`f196737b0c`, 2026-07-03) | **Benign change flagged:** `super()._init_weights()` added, shifting the load-time RNG stream |

- **Caught at introduction, two months before the fix.** #47014 touched shared code, so it selected every contract; a reviewer would have seen T5's intended runtime change beside UMT5 moving by 1.22.
- **One false alarm.** The fixture omits `masked_spec_embed` (as the real checkpoint does) to catch [#49385](https://github.com/huggingface/transformers/issues/49385), so its random init and the SpecAugment loss follow the RNG stream. Nothing user-visible changed; cost: one baseline update.
- **Four misses, one cause: no contract made the call.** CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195), fixed 2026-05-29), processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628), 2026-08-04), Whisper timestamp batch decoding ([#47997](https://github.com/huggingface/transformers/pull/47997), 2026-08-16), T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793), 2026-09-01). All caught in the [replay](#historical-regression-replay) once the call was made (T5 only with punctuation).
- **Selection missed nothing:** all three outcome-changing commits selected the affected contract. Overall it skipped all contracts on 768 commits (52%), selected all on 679 (46%), some on 21 (1%).
- **No other drift** across three CPUs (EPYC 7763, 9V45, 9V74; with and without native bfloat16) and two `tokenizers` versions.

### After adding the missed calls

Calls added via the [improvement loop](improving-contracts.md):

- CLIP: per-modality `text_kwargs`, `get_image_processor_config`.
- Whisper: `generate(return_timestamps=True)` + `batch_decode(..., decode_with_timestamps=True)`, `get_feature_extractor_config`.
- Wav2Vec2: the same config read.
- T5: `decode` on a punctuated batch.

All existing baselines unchanged on all three platforms (98 candidates). Re-backfilled over 1,470 commits ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37664740099)):

| Regression | Fixture result while present | Ends at |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | `proto-clip-vit-base-patch32-slim` errors (`'list' object has no attribute 'shape'`), 206 commits | its fix, `420c80aad0` |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | CLIP `image_processor_config_keys` and Wav2Vec2 `feature_extractor_config_keys` come back empty, 595 and 801 commits | its fix, `c62820a73c` |
| Whisper timestamp decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)) | `proto-whisper-tiny-slim` errors (`'>=' not supported between instances of 'list' and 'int'`), 890 commits | its fix, `64b6595f0c` |
| T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)) | `proto-t5-small-slim` and `proto-umt5-small-slim`: `decoded_roundtrip` keeps the spaces before punctuation, 1,067 commits | its fix, `4ff98bf236` |
| UMT5 causal mask ([#49135](https://github.com/huggingface/transformers/pull/49135)) | unchanged: caught from its introducing commit #47014 | its fix, `d7ba937671` |

- **Each fails exactly until its fix.** Four predate the window, so only UMT5 shows detection at introduction.
- **No new false alarm** (only #46259); **no added cost** (~13 s per commit, 8 fixtures).
- **Post-hoc:** written knowing the bugs, so this shows the loop closes gaps, not the rate on unknown bugs. One call found an unknown one: Whisper `batch_decode` fails on the tensor `generate` returns ([#49403](https://github.com/huggingface/transformers/issues/49403)), locally patched until fixed.

## Historical regression replay

Documented regressions fixed since May 2026 ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)):

1. Freeze random and overfit fixtures and real-checkpoint baselines on a good build (`main` at `4d536e1f`).
2. Extract the bad build (fix's parent) and the fixed build.
3. Run frozen checks on both, regenerating nothing. Caught = bad flagged, fixed clean.
4. Separately run the affected public call on checkpoint and fixture under both builds.

Builds before 2026-08-17 need `tokenizers` 0.22.x (before mid-June, an older `kernels`) via an overlay: the fleet lock must pin dependencies, not just model revisions.

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

- **Generic checks miss most bugs:** forward + greedy generate caught only UMT5, only under default attention. The rest live in processors, tokenizers, decoding helpers.
- **Making the affected call is enough;** the fixture then caught every bug the checkpoint did.
- **Packaging matters:** processor files re-saved by a fixed build hid three loading regressions; copied originals caught the processor-config case on all three families tested.
- **Runtime defaults matter:** the UMT5 bug exists only under SDPA, the default; eager-pinned checks missed it everywhere.
- **Choose attention at load time.** `set_attn_implementation("sdpa")` after loading a T5-family model silently leaves encoder and decoder on eager ([#49372](https://github.com/huggingface/transformers/issues/49372)); the first SDPA pass never exercised SDPA. Set it at load and record the one in use.
- **No false alarms:** 10 benign changes on the same paths (RoPE, greedy decoding, image processing, masking, attention refactors, auto-generated inits), 20 builds × 42 checks, clean under eager and SDPA, including Wav2Vec2's move to SDPA dispatch and a continuous-batching refactor. The May 2026 build lacks T5 SDPA: counted unsupported, not detected.
- **Bugs are mostly in shared code** (processors, tokenizers, attention), rarely in one family: organize contracts around public usage paths, not per-model forward checks.

Sample: eleven candidates (three not reproducible on standard checkpoints), ten benign changes, CPU only, no GPU-specific case.
