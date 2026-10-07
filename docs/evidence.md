# MICs: evidence so far

Companion to the [proposed standard](../README.md#the-proposed-standard). This document collects what the local experiments, the historical regression replay, and the shadow backfill over Transformers `main` show. The synthetic-fault experiments behind the fixture design are detailed in [Small fixtures](fixtures.md); mechanics are in the [technical design](design.md).

## Summary

- **Small fixtures detect what matters, if the checks are numerical.** On synthetic faults across six model families, frozen fixtures with recorded logits, loss, and gradient checks caught 126 of 126 fault cases when overfit and 120 of 126 when left random; a passing forward pass caught none. All random misses were Whisper, whose random fixture ignores its audio.
- **The contract must call what users call.** Replaying eleven candidate Transformers regressions (eight reproducible), generic forward/generate checks missed every processor and tokenizer bug; the same small fixtures called through the affected public API caught them.
- **No false alarms so far.** Fixtures frozen on today's `main` stayed clean across ten benign changes from the past two months.
- **No fixture went non-finite, and real weights are not needed for detection.** Random, overfit, pruned real layers, and sliced real tensors were compared on the six families, in float32, bfloat16, and float16, on CPU and the A10G: 288 load-and-train checks, all finite. Overfit fixtures caught 126 of 126 fault cases, random 120, sliced (real tensors at fixture width) 41 of 42, pruned real layers 42 of 42 at 8–235× the size ([details](fixtures.md#choosing-the-weights)).
- **The fixture recipe reaches about half of all architectures without per-model work.** A generic shrink rule built and ran 314 of 559 Transformers usage paths, all finite; most failures are coupled size fields that need one rule per family ([details](fixtures.md#does-it-scale-to-every-architecture)).
- **Golden outputs catch silent backend bugs.** Checked against the CPU golden, Transformers.js on WebGPU (onnxruntime-node 1.30, Apple M5) matched in fp32 but returned all-zero logits in fp16 and q4f16, with no error. Chrome 154 reproduces it; an fp16 export with fused ops does not, so it is an onnxruntime WebGPU kernel bug triggered by the `onnx-community` exports ([repro](../experiments/webgpu_browser/README.md)).
- **Fixtures carry across hardware; baselines do not.** On one GPU (NVIDIA A10G), the same fixtures ran unchanged, and baselines recorded on that GPU were stable run to run (max drift 6.6e-5 in float32). Baselines recorded on CPU carried over to the GPU only for the families without convolutions, and bfloat16 or float16 runs exceeded tolerance everywhere against a float32 baseline. Baselines have to be recorded per platform class and dtype.

- **The runner prototype found a real Transformers bug.** Requiring baselines to reproduce in a fresh process exposed that seven audio models leave `masked_spec_embed` uninitialized when a checkpoint lacks it, so fine-tuning `facebook/wav2vec2-base-960h` with SpecAugment reads garbage or NaN ([#49385](https://github.com/huggingface/transformers/issues/49385), fix [#49386](https://github.com/huggingface/transformers/pull/49386)). Once fixtures kept the real checkpoint's tensor names and SpecAugment, the slim Wav2Vec2 fixture caught it on every run ([runner prototype](../example/README.md#runner-prototype)).
- **Random weights pass the input-sensitivity gate on five of six families.** With perturbations defined per input name and a threshold of 10× the comparison tolerance, the generic fixture builder kept random weights for CLIP, BERT, T5, UMT5, and Wav2Vec2 and overfit only Whisper, matching the synthetic-fault results. UMT5 #49135 was caught through the runner on a random fixture.
- **Per-platform baselines remove the need for looser tolerances.** On an A10G, 16 targets × 3 variants (float32 default and eager, bfloat16) reproduced at `1e-4` on `cuda`. Drift above `1e-4` appeared only between platforms (CPU → GPU, up to 2e-3 from TF32; SmolLM-135M, bfloat16 by default, generated differently) or dtypes (bfloat16 → float32, no discrete output changed). Training losses with dropout do not carry across platforms ([numbers](../experiments/results/runner.json)).
- **Fixtures must be published, not rebuilt.** Rebuilt on an x86 CPU, every fixture's weights differed from the bytes built on Apple M5, random ones included. They are now published as `hf-internal-testing/proto-*` and pinned by revision and file hashes.
- **Five months of Transformers `main`, replayed: the one covered regression caught at its introducing commit, one benign change flagged, four regressions missed.** The [shadow backfill](#shadow-backfill) ran the 8 fixture contracts, frozen, on all 1,468 first-parent commits from 2026-05-02 to 2026-10-07. The UMT5 causal-mask bug failed from the commit that introduced it ([#47014](https://github.com/huggingface/transformers/pull/47014), which selected every contract) to its fix ([#49135](https://github.com/huggingface/transformers/pull/49135), 597 commits later). The only other numeric change was a benign one: [#46259](https://github.com/huggingface/transformers/pull/46259) shifted the random-number stream at load time, which moved the Wav2Vec2 fixture's training loss (one baseline update). Four regressions fixed inside the window in the same families (CLIP `text_kwargs`, processor config helpers, Whisper timestamp decoding, T5 batch decoding) were missed, because no contract makes the affected call. Selection never skipped a contract on a commit that changed its result. After adding the four missing calls to the contracts, the same backfill caught all five, each failing until its fix, with no new false alarm and no added cost; the added Whisper call also found a bug still on `main` ([#49403](https://github.com/huggingface/transformers/issues/49403)) ([details](#after-adding-the-missed-calls)).
- **Batching makes slim checks cheap.** Eight fixture checks across six contracts take 4.9 s in one process vs 27.4 s with one process per target on the Mac, and 13.1 s vs 81.1 s on the GPU runner.

These are small experiments, not fleet-level evidence: ten model families, synthetic faults, eleven candidate regressions (eight reproducible), ten benign changes, five months of `main` (1,468 commits) on six contracts, CPU only for the replay and the backfill, plus one GPU for stability and cost. The full case list, with commits and verdicts, is in [cases.json](../experiments/cases.json).

## How agents use the Hub today

Two [recorded walkthroughs](agent-walkthrough/README.md), one by [Claude Code](agent-walkthrough/claude-opus.md) and one by [Codex](agent-walkthrough/codex.md), used the same two-task prompt: choose a small model for captioning, then run `microsoft/Florence-2-base` for captioning and detection. Both ran locally on an ARM Mac. Neither had a MIC. These are observations of current friction, not a controlled measurement of improvement from MICs.

| Observation | What a MIC covering that path could provide |
| --- | --- |
| Claude chose Qwen3.5-0.8B, whose card had no in-process snippet. Codex initially chose BLIP, then reused cached Florence weights when downloads failed. | A tested local example and resource requirements help choose a runnable path. Availability and cache state still affect the choice. |
| Codex inferred Transformers 4.41.2 from the config's `transformers_version: 4.41.0.dev0` and available cached packages. | A last-passing record with the exact model revision, framework build, dependencies, and hardware replaces a guess based on serialization metadata. |
| Both agents encountered Florence's `flash_attn` import failure on 4.41.2. Codex inspected the loader and the guarded model import, then filtered that optional dependency from the scanner for CPU execution. | Executing that CPU path in CI would expose the failure. A reviewed fix or explicit workaround variant could then be published and tested; a manifest does not repair the loader by itself. |
| Claude saw `transformersInfo` suggest a native loader while `config.auto_map` selected custom code. Codex saw the analogous conflict between generated Hub usage and the config; its API requests failed. | Documentation and generated snippets can use the same tested entry point, associated with the exact checkpoint and environment. |
| Both used visual judgments. Codex additionally drew the detection boxes, checked coordinate bounds, and verified the requested image against its temporary local fallback. Neither compared model output with a reviewed reference. | Fixed, versioned sample inputs and reviewed real-checkpoint outputs provide a reproducible integration check, with comparison rules appropriate to the runtime. |

**Different outcomes on the same requested repo make the compatibility scope concrete.** Claude produced Task B's result with `florence-community/Florence-2-base` after failing on the Microsoft path. Codex ran the requested Microsoft revision `5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac` with Transformers 4.41.2, torch 2.10.0, CPU float32, eager attention, and the import-scanner workaround. The evidence is therefore that a working combination was undocumented and had to be reconstructed, not that the Microsoft checkpoint cannot run. Success with that workaround does not establish that the unmodified card example or native loader works.

**Reference checks have a bounded meaning.** Agreement with a reviewed baseline shows that an installation reproduces known behavior on fixed inputs. It does not prove that every caption or detection on a new image is semantically correct. A slim fixture checks its own integration behavior; agents need the separate real-checkpoint result when assessing the model they intend to use. The UMT5 replay below illustrates why plausible output alone is weak evidence of integration correctness.

**The observed costs are not all MIC-addressable.** Codex reached the exact-file Task A result in about 6m 28s and Task B's caption plus detection in about 6m 10s from the shared session start. These windows overlap. DNS failures, sandbox restrictions, a `uv` panic, and report-assembly mistakes were separate problems. Existing caches, model choice, prior knowledge, and tool-call accounting differed across agents; the logs do not support a speed ranking or a time-saving estimate.

The runs support tested usage code, explicit compatibility records, reproducible environments, and reference checks. They do not establish the value of overfitting fixtures, fleet-wide CI savings, ownership choices, or maintenance cost; those need the other experiments and the pilot. Direct agent benefit also depends on [discoverability](design.md#agent-discovery-and-reproduction). The initial design defers remote-code models, so it does not yet cover the Microsoft Florence path that caused this compatibility trouble.

The next measurement is a [controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic) with comparable cache/network conditions and fresh sessions, measuring completion on the requested repo, verified-result time, adaptation, failures, and incorrect success claims. Agent reuse is a separate hypothesis to test alongside the CI proposal.

## Stability and cost

### On CPU

**Dependency upgrades.** Fixtures and real-checkpoint baselines frozen with torch 2.11.0, tokenizers 0.23.1, numpy 1.26.4, and Pillow 12.2.0 were re-checked with one dependency changed at a time: torch 2.12.1, 2.13.0, and 2.14.1 (with matching torchvision), tokenizers 0.23.2, numpy 2.2.6, and Pillow 12.3.0, each under eager and SDPA. **No flags in 770 checks** ([stability.json](../experiments/results/stability.json)). The largest drift was 1.3e-5 on slim fixtures and 8.4e-5 on real checkpoints (Wav2Vec2 with numpy 2.x, UMT5 under SDPA), so real-checkpoint tolerances sit closer to the `1e-4` limit than slim ones.

**Cost of one check.** Each check ran in a fresh process, three times, on CPU with warm OS file caches and no downloads ([cost.py](../experiments/cost.py), [cost.json](../experiments/results/cost.json)):

| Family | Slim work | Real work | Slim weights | Real weights |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 0.27 s | 0.25 s | 13 MB | 538 MB |
| CLIP | 0.40 s | 0.41 s | 14 MB | 605 MB |
| BERT | 0.09 s | 0.14 s | 8 MB | 440 MB |
| T5 | 0.14 s | 0.18 s | 9 MB | 242 MB |
| Whisper | 0.57 s | 1.26 s | 15 MB | 151 MB |
| Wav2Vec2 | 0.28 s | 0.47 s | 0.4 MB | 378 MB |
| UMT5 | 0.76 s | 1.15 s | 66 MB | 1,227 MB |

"Work" is processor and input preparation, model load, first inference, comparison, and (slim only) the training probe. On top of that, every fresh process spent about 2.4 s importing torch and Transformers.

What this says:

- **Process start dominates.** A runner should execute many contracts in one process, or the import cost will outweigh the checks themselves.
- **Against these small real checkpoints, slim fixtures save little time locally.** The real checkpoints here are each family's smallest variant, already cached. Real inference was 5–20× slower, but loading and preprocessing are similar.
- **The savings are in transfer and model size.** Slim weights are 10–1,300× smaller than these checkpoints. The slow tests for the same eight families currently load about 144 GB of weights (two Llama 7–8B, three OLMo up to 7B, Whisper large variants, and twelve Wav2Vec2 checkpoints), most of which cannot run on CPU at all.
- **Memory follows the same pattern:** 0.5–1.3 GB peak RSS for slim checks, mostly the framework itself, versus 0.7–2 GB for these small real checkpoints.

### On GPU

One run on the Transformers GPU runners (`aws-g5-4xlarge-cache`: one NVIDIA A10G, x86 CPU, torch 2.14.0+cu130, CUDA 13.0) with Transformers `8625785e`, the same tree as the CPU study's `4d536e1f` ([workflow](../.github/workflows/gpu-checks.yml), [gpu.json](../experiments/results/gpu.json)). Six families, random and overfit fixtures on three seeds plus each real checkpoint: 42 rows per configuration. Everything was frozen twice in float32, once on the runner's CPU and once on the GPU, then checked on the GPU at the same `1e-4` tolerance.

| Baseline recorded on | Checked on GPU as | Flagged (of 42) | Generated outputs changed | Max drift |
| --- | --- | ---: | ---: | ---: |
| GPU, float32 | float32, SDPA (three runs) | 0 | 0 | 6.6e-5 |
| GPU, float32 | float32, eager | 0 | 0 | 0 |
| CPU, float32 | float32, SDPA or eager | 17 | 0 | 2.9e-2 |
| CPU, float32 | float32, SDPA, cuDNN TF32 off (4 families) | 0 of 28 | 0 | 1.9e-4 |
| CPU or GPU, float32 | bfloat16, SDPA | 42 | 6 | 2.3 |
| CPU or GPU, float32 | float16, SDPA | 42 | 1 | 0.30 |

What this says:

- **The fixtures are not tied to CPU.** The same weights and processor files ran on the GPU without changes, and GPU baselines held at `1e-4` across three runs. As on CPU, real checkpoints use most of the margin (6.6e-5, Wav2Vec2) while slim fixtures stay near 1e-5.
- **CPU baselines carry over to GPU only for some families.** SmolLM, BERT, and T5 stayed within 5e-5. CLIP, Whisper, and Wav2Vec2 drifted by 1e-3 to 3e-2, on slim fixtures and real checkpoints alike. The cause is TF32: PyTorch lets cuDNN run convolutions in TF32 by default on Ampere and newer GPUs, and these are the three families with convolutions (patch embedding, audio front end). A rerun with `torch.backends.cudnn.allow_tf32 = False` brought every CPU baseline within tolerance: 0 of 28 flagged across SmolLM, CLIP, Whisper, and Wav2Vec2, against 17 with the default ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37512092438)). Users get TF32, so GPU contracts should keep the default and record GPU baselines instead of turning it off. Generated tokens and labels never changed in float32; only logits moved.
- **Lower precision needs its own baseline.** Against a float32 baseline, every bfloat16 and float16 row was flagged, and generated outputs changed for real SmolLM (both dtypes), two T5 fixtures, and three Wav2Vec2 fixtures (bfloat16). A contract run in bfloat16, which is what users get for checkpoints stored in bfloat16, needs a reviewed baseline recorded in bfloat16 on the same platform class. The [full example](../example/README.md) hit the same issue with eager versus SDPA on CPU.

**Cost on GPU.** Same measurement as on CPU, float32 and SDPA, three fresh processes per check, with weights already cached on the runner ([gpu.json](../experiments/results/gpu.json), `cost`):

| Family | Slim work | Real work | Peak GPU memory, slim | Peak GPU memory, real (float32 / bfloat16) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 1.0 s | 1.5 s | 125 MB | 564 / 295 MB |
| CLIP | 1.5 s | 2.0 s | 134 MB | 631 / 338 MB |
| BERT | 0.6 s | 1.2 s | 103 MB | 456 / 251 MB |
| T5 | 1.5 s | 1.8 s | 105 MB | 266 / 150 MB |
| Whisper | 1.4 s | 1.4 s | 193 MB | 302 / 168 MB |
| Wav2Vec2 | 1.1 s | 1.5 s | 76 MB | 443 / 259 MB |

Imports took about 4.9 s per process on the runner, twice the Mac's 2.4 s, so batching contracts per process matters even more there. Warm inference on the GPU took 3–150 ms. The slim fixtures need under 200 MB of GPU memory, so many of them can share one GPU.

Not measured yet: CPU-only CI runners, environment setup, downloads from the Hub, a cold OS file cache, other GPU types, and multi-GPU.

## Shadow backfill

[`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) ran the fixture contracts, frozen as they are today, on every first-parent commit of Transformers `main` from 2026-05-02 to 2026-10-07: 1,468 commits × 8 fixtures = 11,744 results, with no runner error. Each window ran in 10–20 parallel jobs on GitHub-hosted x86 runners; each job installed the fleet lock's environment once, with Transformers editable, and checked out one commit after another (about 15 s per commit). Baselines were the `cpu-x86_64` ones. Where a commit's own dependency table rejects a pin, the newest accepted version is installed for it: `tokenizers` 0.22 on the 893 commits before 2026-08-17, with no effect on any result. Each result records the commit's selection (`selected_by`, from its diff against its parent); all are on the `shadow-results` branch (`runs/backfill/`).

Every change in outcome, in first-parent order:

| Fixture | Commits | What differs from today's baseline | Starts | Ends | Verdict |
| --- | --- | --- | --- | --- | --- |
| `proto-t5-small-slim`, `proto-umt5-small-slim` | 761 | `runtime`: eager instead of SDPA | window start | [#47014](https://github.com/huggingface/transformers/pull/47014) (`560f36cc56`, 2026-07-30), "Enable SDPA for T5" | Intended: T5-family SDPA did not exist before #47014 |
| `proto-umt5-small-slim` | 597 | `label_logprobs` off by 1.22, training loss by 0.20 | #47014 (`560f36cc56`) | [#49135](https://github.com/huggingface/transformers/pull/49135) (`d7ba937671`, 2026-09-28) | **Regression caught at its introducing commit** (UMT5 decoder not causal under SDPA) |
| `proto-wav2vec2-base-960h-slim` | 513 | training loss off by 0.39 | window start | [#46259](https://github.com/huggingface/transformers/pull/46259) (`f196737b0c`, 2026-07-03) | **Benign change flagged:** `super()._init_weights()` added, shifting the load-time RNG stream |

All other results passed: SmolLM (both fixtures), CLIP, BERT, Whisper, and T5 after #47014, on every commit.

**What this shows.**

- **The one regression the contracts cover was caught on its introducing commit.** #47014 enabled SDPA for the T5 family; the same commit made UMT5's decoder non-causal. Run as a non-blocking check, the fixture would have flagged it on that PR, which touched shared code and so selected every contract, two months before the fix. A reviewer would have seen T5's runtime change, which is intended, next to UMT5's outputs moving by 1.22, which is not.
- **One false alarm in five months.** #46259 added `super()._init_weights(module)` to Wav2Vec2, so the parent init draws random numbers for every module before the model's own branches, as its commit message notes for RegNet. The fixture deliberately lacks `masked_spec_embed`, like the real checkpoint, so that tensor is randomly initialized on load, and its values, and with them the SpecAugment training loss, depend on the RNG stream. Nothing a user sees changed beyond the random init. This is the price of omitting the tensor to catch [#49385](https://github.com/huggingface/transformers/issues/49385); it would have cost one baseline update.
- **Four regressions in the same families were missed, all for the same reason.** Fixed inside the window but present before it: CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195), fixed 2026-05-29), the processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628), 2026-08-04), Whisper timestamp batch decoding ([#47997](https://github.com/huggingface/transformers/pull/47997), 2026-08-16), and T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793), 2026-09-01). No fixture result changed at their fixes. Each was caught in the [replay](#historical-regression-replay) once the check made the affected call (T5 batch decoding only with punctuation in the inputs); today's contracts make none of these calls. Detection depends on the usage paths a contract exercises, not on the fixture.
- **Selection missed nothing.** The three commits where a result changed (#47014, #46259, #49135) all selected the affected contract: the first two through shared code, the fix through `models/umt5`. Overall, selection skipped every contract on 768 commits (52%), selected all of them on 679 (46%), and narrowed to some on 21 (1%).
- **No baseline drift otherwise.** Across 1,468 commits, three CPU models (AMD EPYC 7763, 9V45, 9V74, with and without native bfloat16), and two `tokenizers` versions, no other result moved.

### After adding the missed calls

Following the [improvement loop](improving-contracts.md), the four calls were added to the contracts (CLIP per-modality `text_kwargs` and `get_image_processor_config`; Whisper `generate(return_timestamps=True)` with `batch_decode(..., decode_with_timestamps=True)` and `get_feature_extractor_config`; the same config read for Wav2Vec2; T5 `decode` on a punctuated batch). Every existing baseline stayed identical on all three platforms (98 candidates reviewed), and the same five months were backfilled again (1,470 commits, [run](https://github.com/huggingface/model-integration-contracts/actions/runs/37664740099)):

| Regression | Fixture result while present | Ends at |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | `proto-clip-vit-base-patch32-slim` errors (`'list' object has no attribute 'shape'`), 206 commits | its fix, `420c80aad0` |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | CLIP `image_processor_config_keys` and Wav2Vec2 `feature_extractor_config_keys` come back empty, 595 and 801 commits | its fix, `c62820a73c` |
| Whisper timestamp decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)) | `proto-whisper-tiny-slim` errors (`'>=' not supported between instances of 'list' and 'int'`), 890 commits | its fix, `64b6595f0c` |
| T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)) | `proto-t5-small-slim` and `proto-umt5-small-slim`: `decoded_roundtrip` keeps the spaces before punctuation, 1,067 commits | its fix, `4ff98bf236` |
| UMT5 causal mask ([#49135](https://github.com/huggingface/transformers/pull/49135)) | unchanged: caught from its introducing commit #47014 | its fix, `d7ba937671` |

- **All four misses are now caught, and each failure ends exactly at the bug's fix.** Five of the five regressions in the contracts' families during the window fail their fixture while present. Four of them were introduced before the window, so only UMT5 shows detection on the introducing commit.
- **No new false alarm.** SmolLM and BERT passed on every commit, and every other result passed once the bugs above were fixed; the only benign failure is still #46259's RNG shift on the Wav2Vec2 training loss.
- **No added cost.** About 13 s per commit for the 8 fixtures, before and after.
- **This is post-hoc.** These four checks were written knowing the bugs, so they show that the loop closes gaps, not how many unknown bugs the contracts would catch. The fifth call it produced did find an unknown one: Whisper's `batch_decode` still fails on the tensor `generate` returns ([#49403](https://github.com/huggingface/transformers/issues/49403)), which a local patch works around until it is fixed.

## Historical regression replay

To test real bugs rather than synthetic faults, I replayed documented Transformers regressions fixed since May 2026 ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)).

Method:

1. Freeze random and overfit slim fixtures, plus real-checkpoint baselines, on a known-good build (`main` at `4d536e1f`).
2. Extract the bad build (the fix commit's parent) and the fixed build.
3. Run the frozen checks on both builds without regenerating anything. A case counts as caught when the bad build is flagged and the fixed build is not.
4. Separately, run the affected public call on the real checkpoint and on the slim fixture under both builds.

Builds before 2026-08-17 need `tokenizers` 0.22.x (and builds before mid-June an older `kernels`), so those ran with a dependency overlay. Old builds cannot always run in today's environment; the fleet lock needs to pin the dependency environment, not just model revisions.

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

The pattern is consistent:

- **Generic checks are not enough.** A forward pass plus greedy generate on frozen inputs caught only the model-level regression (UMT5), and only when run under the default attention implementation. The others live in processors, tokenizers, and decoding helpers.
- **The usage path catches them, and the small fixture is enough for it.** Whenever the contract made the affected call, the slim fixture caught the bug just as the real checkpoint did. None of these needed real weights.
- **Fixture packaging matters.** Three loading regressions were visible on the real checkpoint but not on a fixture whose processor files had been re-saved by a fixed build. For the processor-config case, a fixture with the original files copied unchanged caught it on all three families tested.
- **Runtime defaults matter.** The UMT5 bug only exists under SDPA, the default. Eager-pinned checks missed it on both the real checkpoint and the fixtures; SDPA checks caught it everywhere.
- **Switching attention after loading is not reliable.** On current `main`, `set_attn_implementation("sdpa")` after loading a T5-family model updates the top-level config but leaves the encoder and decoder on eager, silently ([#49372](https://github.com/huggingface/transformers/issues/49372)). My first SDPA pass used it and never exercised SDPA for T5/UMT5. Contracts should choose the attention implementation at load time and record the one actually in use.

To measure false alarms, I replayed 10 benign changes from the same window that touched the same code paths (RoPE, greedy decoding, image processing, masking, attention refactors, auto-generated inits). Across 20 builds and 42 frozen checks per build, there were **no false alarms** with eager attention. Rechecking all 20 builds under SDPA, chosen at load time, also produced no false alarms, including the change that moved Wav2Vec2 to SDPA dispatch and an attention refactor for continuous batching. (The May 2026 build does not support SDPA for T5, so that variant fails on both bad and fixed builds; it counts as unsupported, not as a detection.)

The candidate search also showed that documented regressions inside a given model family are relatively rare in a five-month window; most bugs that matter to these models are in shared processor, tokenizer, and attention code. That supports organizing contracts around public usage paths rather than per-model forward checks.

This is a small sample: eleven candidate regressions (three not reproducible on standard checkpoints), ten benign changes, CPU only, and no GPU-specific case.
