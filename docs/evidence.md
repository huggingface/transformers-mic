# MICs: evidence so far

Results from local experiments, the regression replay, the shadow backfill over Transformers `main`, and the job in Transformers' PR CI. Fixture experiments: [Small fixtures](fixtures.md); mechanics: [technical design](design.md).

## Summary

**Detection**

- **Contracts must call what users call:** generic checks missed every processor and tokenizer bug; the affected call caught all of them ([replay](#historical-regression-replay)).
- **Five months of `main`: 1 caught, 4 missed, 1 false alarm**; the misses were calls no contract made ([backfill](#shadow-backfill)). With those calls added, all five are caught with no new false alarm, and the new Whisper call found a live bug, [#49403](https://github.com/huggingface/transformers/issues/49403) ([details](#after-adding-the-missed-calls)).
- **A call added for one bug caught another nobody targeted:** Whisper's assisted `generate` raised for 41 commits after [#48981](https://github.com/huggingface/transformers/pull/48981), from its introducing commit to the fix [#49075](https://github.com/huggingface/transformers/pull/49075) ([third round](#third-round-second-round-calls-and-bfloat16-smollm)).
- **Scheduled runs caught a regression the night it merged:** #48282 (2026-10-08) made `generate()` raise on every original Whisper checkpoint, and the next nightly and shadow runs failed Whisper everywhere ([#49456](https://github.com/huggingface/transformers/issues/49456); a `local_patches` workaround here, a narrowed known failure in the PR CI job). The PR's CI could not see it ([why](https://github.com/huggingface/transformers/issues/49456#issuecomment-6076843277)): model tests build configs without `suppress_tokens`, pipeline tests pass their own `generation_config`, and the slow `generate()` tests were not in its `run-slow` list; the contract calls `generate()` on the checkpoint's config, as users do.
- **In Transformers' PR CI, a reverted fix fails as in the backfill:** reverting #49135 turns the job red with the same UMT5 deltas, on Intel with the CI image's packages, and the summary's repro command gives the same failure on a Mac ([proof](#transformers-ci-proof)).
- **The runner found a real bug:** seven audio models leave `masked_spec_embed` uninitialized ([#49385](https://github.com/huggingface/transformers/issues/49385), fix [#49386](https://github.com/huggingface/transformers/pull/49386)); the slim Wav2Vec2 fixture catches it every run ([runner](../example/README.md#runner-prototype)).
- **No real regression has needed an overfit fixture yet:** random twins caught the MusicGen Melody conditioning drop ([#48679](https://github.com/huggingface/transformers/pull/48679)) and Whisper's left-padded long-form bug ([#48028](https://github.com/huggingface/transformers/pull/48028)) as well as the overfit fixtures. Whisper needed controlled unequal first-segment lengths to reach the trigger ([follow-up](#whisper-left-padded-previous-segment-prompts)); assisted decoding #48108 remains missed.
- **Most catches needed an assertion:** of 18 regressions caught, 6–7 raised and 11 returned a wrong value with no error ([smoke vs assertions](smoke-vs-assertions.md)).
- **Golden outputs catch silent backend bugs:** Transformers.js on WebGPU returned all-zero fp16/q4f16 logits with no error ([repro](../experiments/webgpu_browser/README.md)).

**Fixtures, baselines, cost**

- **Fixtures:** synthetic-fault detection, weight choice, and the architecture sweep are in [Small fixtures](fixtures.md#tldr). Publish them, don't rebuild: x86 rebuilds differed from the M5 bytes.
- **Baselines are per platform and dtype; fixtures are shared** ([GPU](#on-gpu), [bfloat16](#bfloat16-on-cpu)). So split, 16 targets × 3 variants reproduced at `1e-4` on A10G ([numbers](../experiments/results/runner.json)); ten benign changes raised no false alarm.
- **Batching is the cost lever:** eight fixture checks take 4.9 s in one process vs 27.4 s one per process (Mac); 13.1 s vs 81.1 s (GPU runner).

**Original study:** ten families, eleven candidate regressions (eight reproducible), ten benign changes, five months on six contracts; replay/backfill on CPU, one GPU for stability/cost ([cases.json](../experiments/cases.json)). The fleet now has eight contracts and nine fixtures. The five-month backfills and original cost tables predate MusicGen Melody; the weekly backfill and targeted GPU replays below include later coverage.

## How agents use the Hub today

Two [walkthroughs](agent-walkthrough/README.md), same prompt, ARM Mac, no MIC: pick a small captioning model, then run `microsoft/Florence-2-base` for captioning and detection. They show friction, not a measured MIC gain.

| Observation | What a MIC on that path could provide |
| --- | --- |
| Claude chose Qwen3.5-0.8B (no in-process snippet on its card); Codex chose BLIP, then reused cached Florence weights when downloads failed. | A tested local example and resource requirements. |
| Codex inferred Transformers 4.41.2 from `transformers_version: 4.41.0.dev0` and cached packages. | A last-passing record (revision, build, dependencies, hardware). |
| Both hit Florence's `flash_attn` import failure on 4.41.2; Codex filtered it from the import scanner. | CI exposes the failure so a workaround can be tested as a variant; a manifest doesn't repair the loader. |
| `transformersInfo` suggested a native loader while `config.auto_map` chose custom code. | One tested entry point per checkpoint and environment. |
| Both judged output visually; neither used a reviewed reference. | Fixed inputs and reviewed real-checkpoint outputs. |

- **Same repo, different outcomes:** Claude finished Task B on `florence-community/Florence-2-base`; Codex ran Microsoft's revision `5ca5edf5bd01` with Transformers 4.41.2, torch 2.10.0, CPU float32, eager, plus the scanner workaround. A working combination existed but was undocumented.
- **Not shown:** CI savings, ownership, maintenance cost; DNS, sandbox, and tooling failures are not MIC-addressable, and differing caches rule out a speed ranking. Benefit also needs [discoverability](design.md#agent-discovery-and-reproduction). Next: a [controlled comparison](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic).

## Stability and cost

### On CPU

**Dependency upgrades: 0 flags in 770 checks** ([stability.json](../experiments/results/stability.json)). Frozen with torch 2.11.0, tokenizers 0.23.1, numpy 1.26.4, Pillow 12.2.0; re-checked one upgrade at a time (torch 2.12.1/2.13.0/2.14.1, tokenizers 0.23.2, numpy 2.2.6, Pillow 12.3.0), eager and SDPA. Max drift 1.3e-5 slim, 8.4e-5 real: real checkpoints sit closer to `1e-4`.

**Cost per check:** experimental builds, fresh process, three runs, warm cache ([cost.json](../experiments/results/cost.json)). "Work" = input prep, load, first inference, comparison, (slim) training probe; imports add ~2.4 s per process. These are not current published-fixture sizes (e.g. published Wav2Vec2 is 4.3M parameters, about 17 MB).

| Family | Slim work | Real work | Slim weights | Real weights |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 0.27 s | 0.25 s | 13 MB¹ | 538 MB |
| CLIP | 0.40 s | 0.41 s | 14 MB | 605 MB |
| BERT | 0.09 s | 0.14 s | 8 MB | 440 MB |
| T5 | 0.14 s | 0.18 s | 9 MB | 242 MB |
| Whisper | 0.57 s | 1.26 s | 15 MB | 151 MB |
| Wav2Vec2 | 0.28 s | 0.47 s | 0.4 MB | 378 MB |
| UMT5 | 0.76 s | 1.15 s | 66 MB | 1,227 MB |

¹ float32 build; the bfloat16 fixture is 6.1 MB.

- **Process start dominates**, so run many contracts per process; on small cached checkpoints, slim saves little local time.
- **Savings are transfer and size:** slim weights are 10–1,300× smaller; slow tests for these families load ~144 GB, mostly not CPU-runnable.
- **Memory:** 0.5–1.3 GB peak RSS slim (mostly framework), 0.7–2 GB real.

### bfloat16 on CPU

SmolLM-135M and -360M ship bfloat16, which users get by default; fixtures used to be float32. [`build_fixtures.py`](../example/build_fixtures.py) now saves each fixture in its source's dtype: only mic-SmolLM-slim changed (float32 fixtures rebuild byte-identical).

- **Random weights pass the gate but are useless for generation:** logits moved ≥ 0.24 under every perturbation (gate: 1e-3), yet all four prompts generated " is is is is". Overfit in float32, then cast: " Paris…", " Berlin…", " Rome…", " London…" (80 steps; 3.2M parameters, 6.1 MB).
- **The training batch has an irreducible loss:** rows share "The capital of" then name a country, so ln 4 at that position can't be learned; training plateaus at 0.25, so the SmolLM recipe sets `target_loss: 0.3`.

**Native bfloat16 changes real-checkpoint results, not the fixture's.** GitHub runners at `8625785e`, EPYC 7763 (`avx2` only; run 37916317747) vs EPYC 9V45 (`avx512_bf16`; 37916321632):

| Target (bfloat16) | Max logit delta | Loss delta | Generated tokens |
| --- | ---: | ---: | --- |
| mic-SmolLM-slim, SDPA / eager | 0 / 0 | 1.5e-5 / 1.9e-5 | same |
| SmolLM-135M, SDPA / eager | 0.25 / 0.25 | 0.0074 / 0.011 | same / **differ** |
| SmolLM-360M, SDPA / eager | 0.125 / 0.25 | 0.010 / 0.013 | same |

- bfloat16 runs on x86 CPUs with `avx512_bf16` or `amx_bf16` use the `cpu-x86_64-bf16` platform, which the [runner](../example/run.py) picks from `/proc/cpuinfo`. GitHub's x86 runners are mixed (one of three shadow runs on 2026-10-08/09 had `avx512_bf16`).
- The fixture agrees across both CPUs (hidden size 64 likely never reaches the native kernels) but follows the same rule.
- Emulated bfloat16 on x86 and the M5 generate the same tokens on every SmolLM target. The 24 shared float32 targets agree across the 7763, 9V45, and 7R32.
- Intel (`amx_bf16`) vs AMD (`avx512_bf16`): the fixture matches ([PR CI runner](#on-the-transformers-pr-ci-runner)); real bfloat16 checkpoints on Intel are untested.

### On the Transformers PR CI runner

Phase 1 of the [CI proof](transformers-ci-proof.md): the eight fixtures on `aws-m8i-l-cache` in `huggingface/transformers-torch-light`, the runner and image of Transformers' PR test jobs, from the `contracts-proof` branch of `huggingface/transformers` ([run 37932247464](https://github.com/huggingface/transformers/actions/runs/37932247464), 2026-10-09), on the baselines' build `8625785e`.

| Packages | Differ from the lock | Default | Eager | Run (8 fixtures, one process) |
| --- | --- | ---: | ---: | ---: |
| Image as is | Python 3.11.17, torch 2.14.0, numpy 2.3.5, tokenizers 0.23.2, + 8 more | 8/8 | 8/8 | 37–39 s |
| Lock's pins installed | Python 3.11.17 only | 8/8 | 8/8 | 36–57 s |

- **The `cpu-x86_64` baselines hold on Intel**, recorded on AMD EPYC 7R32/7763/9V45; no pin install needed (16 s if it were).
- **Intel `amx_bf16` matches AMD `avx512_bf16` on the bfloat16 fixture** (Xeon 6975P-C against the 9V45's `cpu-x86_64-bf16` baseline). Weak evidence: the fixture also matches emulated bfloat16.
- **Slower than a 2-core GitHub runner** (~13 s) on 128 cores, from thread oversubscription on tiny tensors: `OMP_NUM_THREADS=8` brings the eight fixtures to 17 s in the [CI job](#transformers-ci-proof).
- Installing Transformers and the runner's dependencies over the image: 14 s.

### Transformers CI proof

The contracts job on the `contracts-proof` branch of `huggingface/transformers` ([plan](transformers-ci-proof.md)): the branch's own build, PR CI runner and image (`aws-m8i-l-cache`, `transformers-torch-light`), contracts selected from changed files, no `local_patches` (open bugs as narrowed `known_failures`). Demo changes pushed as `proof-*` branches, 2026-10-09:

| Change | Selected | Result | Runner time |
| --- | --- | --- | ---: |
| Docs page | none | green | 11.5 s |
| No-op in `generation/utils.py` | all 6 (shared code) | 7 pass, Whisper `xfail` (#49456) | 17.8 s |
| No-op in `models/clip/` | CLIP | pass | 11.5 s |
| Revert of the UMT5 fix #49135 | T5 (`models/umt5`) | **red**: `mic-umt5-small-slim` `label_logprobs` off by 1.22, loss by 0.204; `mic-t5-small-slim` passes | 13.4 s |

- **The reverted regression fails with the backfill's exact deltas** on another CPU vendor (Intel Xeon 6975P-C, not AMD EPYC) and package set (the image's torch 2.14, numpy 2.3), against baselines recorded on AMD with the lock's pins.
- **Selection worked as designed:** model-scoped changes ran one contract, shared code all six, docs none.
- **Known failures without patches:** #49456 is `xfail` on its message; #49385 needs Wav2Vec2's loss and gradients skipped on this runner (non-finite gradients from uninitialized memory, unlike on the M5).
- **Failures reproduce on a Mac:** with `cpu-arm64` baselines on the branch (`fd5ecc0768`), the summary's repro command gives 7 pass + Whisper `xfail` on the branch and the same UMT5 deltas (1.22, 0.204) on the revert.
- Runner time excludes setup; install over the image is ~15 s, and fixtures are cached per lock. A workflow bug broke full runs (pushes to `contracts-proof`, not the `proof-*` runs above) until `46fa53214f` ([run 37938424467](https://github.com/huggingface/transformers/actions/runs/37938424467), green; [details](transformers-ci-proof.md)).

### GPU dependency upgrades

2026-10-11–12: the eight frozen contracts, known-good Transformers `8625785e`, A10G, existing CUDA baselines and tolerances. Each environment gets two fresh-process fleet passes in default, eager and bfloat16. [Initial matrix](https://github.com/huggingface/model-integration-contracts/actions/runs/38146798276); [harness](../experiments/dependency_study.py), [summary tool](../experiments/summarize_dependency_study.py). No artifacts, approved outputs or tolerances were changed.

| Environment | Pass / fail / error | Repeated output pairs |
| --- | --- | --- |
| Pinned control | 114 / 0 / 0 | 57 exact |
| tokenizers 0.23.2 | 114 / 0 / 0 | 57 exact |
| Pillow 12.3.0 | 114 / 0 / 0 | 57 exact |
| NumPy 2.2.6 | 102 / 12 / 0 | 57 exact |
| Combined processor upgrades | 102 / 12 / 0 | 57 exact |
| No-torchaudio control | 102 / 0 / 12 | 51 exact; 6 unavailable |
| torch 2.14.1 / torchvision 0.29.1, CUDA 13 | 84 / 18 / 12 | 51 exact; 6 unavailable |

Each row also has 12 declared exclusions (UMT5 checkpoint and WhisperSpeech fixture, across six passes). Tokenizers and Pillow have zero numeric drift. NumPy and the combined profile fail only Wav2Vec2, fixture and checkpoint, in all three variants; generated IDs and text stay unchanged. Float32 logit drift reaches `8.38e-4`; real bfloat16 reaches `0.125`, with training loss changing `0.121`. These are reproducible environment changes, not run-to-run noise.

An ARM preprocessing diagnostic holds raw tone bytes fixed: normalization inputs change by up to `1.19e-7`. The denominator changes from float64 to float32 under [NumPy 2 scalar promotion](https://numpy.org/doc/2.0/numpy_2_0_migration_guide.html#changes-to-numpy-data-type-promotion). An explicit scalar-precision control restores the NumPy 1 processor bytes exactly; the GPU full-call control below verifies its effect on the model. [Diagnostic](../experiments/numpy_wav2vec2_preprocessing.py). Keep the NumPy pin; investigate normalization arithmetic before widening tolerances or approving new expectations.

The no-torchaudio control loses both MusicGen Melody targets, as expected from their processor dependency; these errors are not passes. The PyTorch/CUDA upgrade also needs a runtime preflight: CUDA 12 and 13 distributions can share library paths, so version metadata alone may describe libraries whose bytes were overwritten. The [completed retry](https://github.com/huggingface/model-integration-contracts/actions/runs/38212201139) restores the CUDA 13 wheel files and prioritizes their libraries. Its 18 numeric failures affect CLIP (fixture float32; checkpoint bfloat16) and both Wav2Vec2 targets in all three variants. Every produced output repeats exactly; the sampled inference IDs/text and other exact assertions still pass. This is a torch/torchvision/CUDA-library bundle comparison, not an isolated torch change.

**Total:** 798 target attempts: 732 pass, 42 numeric failures, 24 missing-torchaudio errors; another 84 declared exclusions. All 387 pairs with outputs repeat exactly; 12 MusicGen pairs are unavailable. Warm default fleet passes take about 54–56 s, eager 81–84 s and bfloat16 83–87 s in the processor profiles. First passes include cache effects; this does not measure cold/warm total cost. Keep the production pins and tolerances; the next investigation is explicit normalization arithmetic and the CUDA-bundle drift. [Full report, versions, hashes and per-target results](../experiments/results/gpu_dependency_stability.json).

#### Wav2Vec2 normalization control

2026-10-12: [A10G validation](https://github.com/huggingface/model-integration-contracts/actions/runs/38214412697), pinned Transformers `8625785e`, torch 2.11.0+cu128. Both published Wav2Vec2 targets ran their full inference and SGD training contract in default/eager/bfloat16, twice in fresh processes per condition: **36 target calls**.

| Condition | Existing baselines | Compared with NumPy 1 |
| --- | --- | --- |
| NumPy 1.26.4, native normalization | 12 pass | Reference |
| NumPy 2.2.6, native normalization | 12 fail | Raw audio identical; normalized inputs and numeric model outputs differ |
| NumPy 2.2.6, explicit arithmetic | 12 pass | Normalized input bytes and every reported model output match exactly; zero logit/loss drift |

The control computes `sqrt(float64(var) + float64(1e-7))`, rounds the denominator to float32, then divides the float32 audio. It preserves the NumPy 1 scalar arithmetic without changing mean/variance accumulation. All **18 repeat pairs** are exact. This confirms normalization arithmetic accounts for the earlier NumPy flags on these GPU calls; wider tolerances or new expectations are unnecessary for this control. It remains an experimental override, scoped to the float32 processor audio in these fixed inputs; gradient tensors and updated weights are not recorded outputs. Production code, artifacts and baselines are unchanged. [Harness](../experiments/wav2vec2_normalization_control.py), [validation and provenance](../experiments/results/wav2vec2_normalization_control.json). A framework fix still needs upstream tests for the supported normalization inputs.

### On GPU

One run on `aws-g5-4xlarge-cache` (A10G, torch 2.14.0+cu130), Transformers `8625785e` ([workflow](../.github/workflows/gpu-checks.yml), [gpu.json](../experiments/results/gpu.json)). 42 rows: six families, random and overfit fixtures on three seeds, plus real checkpoints; float32 baselines from the runner's CPU and GPU, checked on GPU at `1e-4`.

| Baseline recorded on | Checked on GPU as | Flagged (of 42) | Generated outputs changed | Max drift |
| --- | --- | ---: | ---: | ---: |
| GPU, float32 | float32, SDPA (three runs) | 0 | 0 | 6.6e-5 |
| GPU, float32 | float32, eager | 0 | 0 | 0 |
| CPU, float32 | float32, SDPA or eager | 17 | 0 | 2.9e-2 |
| CPU, float32 | float32, SDPA, cuDNN TF32 off (4 families) | 0 of 28 | 0 | 1.9e-4 |
| CPU or GPU, float32 | bfloat16, SDPA | 42 | 6 | 2.3 |
| CPU or GPU, float32 | float16, SDPA | 42 | 1 | 0.30 |

- **Fixtures run unchanged on GPU**; real checkpoints use most of the margin (6.6e-5), slim stays near 1e-5.
- **CPU baselines carry over only without convolutions:** SmolLM, BERT, T5 within 5e-5; CLIP, Whisper, Wav2Vec2 drift 1e-3–3e-2. The cause is cuDNN's default TF32 ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37512092438) with it off: 0 of 28 flagged). Users get TF32, so record GPU baselines.
- **Lower precision needs its own baseline:** bfloat16 changed generated outputs for real SmolLM and five fixtures; float32 tokens and labels never changed.

**GPU cost:** float32, SDPA, three fresh processes, weights cached.

| Family | Slim work | Real work | Peak GPU memory, slim | Peak GPU memory, real (float32 / bfloat16) |
| --- | ---: | ---: | ---: | ---: |
| SmolLM | 1.0 s | 1.5 s | 125 MB | 564 / 295 MB |
| CLIP | 1.5 s | 2.0 s | 134 MB | 631 / 338 MB |
| BERT | 0.6 s | 1.2 s | 103 MB | 456 / 251 MB |
| T5 | 1.5 s | 1.8 s | 105 MB | 266 / 150 MB |
| Whisper | 1.4 s | 1.4 s | 193 MB | 302 / 168 MB |
| Wav2Vec2 | 1.1 s | 1.5 s | 76 MB | 443 / 259 MB |

Imports take ~4.9 s (2× the Mac), so batching matters more; slim fixtures need < 200 MB, so many share a GPU. This cost study excludes CPU-only CI runners, setup/downloads, cold cache, other GPUs, multi-GPU, and MusicGen Melody; the PR CI proof above measures setup separately.

## Shadow backfill

[`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) ran the frozen fixture contracts on every first-parent `main` commit from 2026-05-02 to 2026-10-07: 1,468 × 8 = 11,744 results, no runner error, ~15 s per commit on GitHub x86 with `cpu-x86_64` baselines. Commits that reject a pin use their newest accepted version (`tokenizers` 0.22 before 2026-08-17; no effect). Results, with `selected_by`, are on `shadow-results` (`runs/backfill/`).

Every change in outcome (all else passed on every commit):

| Fixture | Commits | Differs from today's baseline | Starts | Ends | Verdict |
| --- | --- | --- | --- | --- | --- |
| `mic-t5-small-slim`, `mic-umt5-small-slim` | 761 | `runtime`: eager, not SDPA | window start | [#47014](https://github.com/huggingface/transformers/pull/47014) (`560f36cc56`, 2026-07-30), "Enable SDPA for T5" | Intended |
| `mic-umt5-small-slim` | 597 | `label_logprobs` off by 1.22, training loss by 0.20 | #47014 | [#49135](https://github.com/huggingface/transformers/pull/49135) (`d7ba937671`, 2026-09-28) | **Regression caught at its introducing commit** (UMT5 decoder not causal under SDPA) |
| `mic-wav2vec2-base-960h-slim` | 513 | training loss off by 0.39 | window start | [#46259](https://github.com/huggingface/transformers/pull/46259) (`f196737b0c`, 2026-07-03) | **Benign change flagged:** load-time RNG stream shifted |

- **Caught at introduction, two months before the fix:** #47014 touched shared code, so it selected every contract; its reviewer would have seen UMT5 move by 1.22 beside T5's intended runtime change.
- **One false alarm:** the fixture omits `masked_spec_embed`, as the checkpoint does, to catch [#49385](https://github.com/huggingface/transformers/issues/49385), so its loss follows the RNG stream. Cost: one baseline update.
- **Four misses, one cause: no contract made the call.** CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)), processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)), Whisper timestamp batch decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)), T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)).
- **Selection missed nothing:** all three outcome-changing commits selected the affected contract. It selected none on 768 commits (52%), all on 679 (46%), some on 21.
- **No other drift** across three CPUs (EPYC 7763, 9V45, 9V74) and two `tokenizers` versions.

### After adding the missed calls

Calls added via the [improvement loop](improving-contracts.md#what-happened-the-first-time); no existing baseline moved on any platform (98 candidates). Re-backfilled over 1,470 commits (2026-05-01 to 2026-10-08, [run](https://github.com/huggingface/model-integration-contracts/actions/runs/37664740099)):

| Regression | Fixture result while present | Ends at (fix) |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | `mic-clip-vit-base-patch32-slim` errors, 206 commits | `420c80aad0` |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | CLIP and Wav2Vec2 config keys empty, 595 and 801 commits | `c62820a73c` |
| Whisper timestamp decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)) | `mic-whisper-tiny-slim` errors, 890 commits | `64b6595f0c` |
| T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)) | T5 and UMT5 `decoded_roundtrip` keeps spaces before punctuation, 1,067 commits | `4ff98bf236` |
| UMT5 causal mask ([#49135](https://github.com/huggingface/transformers/pull/49135)) | caught from its introducing commit #47014 | `d7ba937671` |

Each fails exactly until its fix (four predate the window); no new false alarm, no added cost (~13 s per commit). **Post-hoc:** the calls were written knowing the bugs, so this shows the loop closes gaps, not the rate on unknown bugs.

### Third round: second-round calls and bfloat16 SmolLM

After the [second round](improving-contracts.md#the-second-round-triaging-the-model-tier-misses) added Whisper `max_length_tokens`, `assisted_tokens`, `longform_tokens`, and mic-SmolLM-slim moved to bfloat16 ([why](#bfloat16-on-cpu)). 1,480 commits, 2026-05-01 to 2026-10-08 ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37930845944)); every other segment as in the second round.

| Window | Commits | Fixture result | Verdict |
| --- | ---: | --- | --- |
| #47997 fix (`64b6595f0c`) to [#48000](https://github.com/huggingface/transformers/pull/48000) (`a5f41e095a`), 2026-08-16/17 | 2 | `mic-whisper-tiny-slim` errors in the assisted call: `UnboundLocalError: is_updated` | Caught; the call was added for this bug (post-hoc) |
| [#48981](https://github.com/huggingface/transformers/pull/48981) (`87d34bfa80`, 2026-09-22, "Always correctly restrict assisted decoding with max length/eos token") to [#49075](https://github.com/huggingface/transformers/pull/49075) (`010f0ca2ec`, 2026-09-24, "Fix assisted eos token condition") | 41 | same call errors: `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar` in `_assisted_decoding` | **New catch, not targeted: caught at its introducing commit** |

- **Not post-hoc for #48981:** `assisted_tokens` was added for #48000, and `find_misses.py`'s model tier never listed #49075 (shared generation code only). #48981 changes `generation/`, so selection would have run Whisper on that PR; it also deleted 9 lines from `tests/generation/test_utils.py`.
- **bfloat16 SmolLM fixture: no false alarm**, all 1,480 commits, on `cpu-x86_64` (888: EPYC 7763, 9V74 without native bfloat16, Xeon 8370C/8573C) and `cpu-x86_64-bf16` (592: EPYC 9V45, 9V74 with `avx512_bf16`).
- **Cost:** 6.7 s per commit summed over targets, vs 6.2 s (the three added Whisper calls).

### Weekly check: 2026-10-11

The scheduled [shadow](https://github.com/huggingface/model-integration-contracts/actions/runs/38109078172) passed 9/9; [nightly](https://github.com/huggingface/model-integration-contracts/actions/runs/38106887222) passed 90 checks with the same 5 declared `umt5-small` exclusions (`config.json` lacks `model_type`). Both tested Transformers `536ecc0073`, the same commit as 2026-10-10. Target sets and statuses matched the previous day, with no `environment_differs`: another repeatability sample, no new commit coverage.

`find_misses.py` on the six shadow runs of 2026-10-08..11 found no candidates across five distinct commits, with or without `--shared`. The daily samples omit intervening commits, so the [weekly backfill](https://github.com/huggingface/model-integration-contracts/actions/runs/38134921721) checked every first-parent commit in `[2026-10-05, 2026-10-12)`: **75 commits × 9 fixtures = 675 passes**, no missing commits or runner errors, no dependency overrides or `environment_differs`. Result platforms: 636 `cpu-x86_64`, 39 `cpu-x86_64-bf16`. Records: `shadow-results:runs/backfill/2026-10-05_2026-10-12-38134921721.jsonl`.

Miss triage: one model-tier candidate and 35 shared-tier rows (five fixes matched to all seven contracts). Read the diffs and the contracts' calls:

| Fix | Verdict | Reason |
| --- | --- | --- |
| [#49419](https://github.com/huggingface/transformers/pull/49419), `9167f73f23` | Known bug, workaround masks the transition | Whisper timestamp batch decoding is already called; `whisper-decode-tensor-batch` fixes it on the older builds. This is not a new missing call. |
| [#48700](https://github.com/huggingface/transformers/pull/48700), `492f240432` | Outside current usage | Quantized cache configuration and beam reordering; contracts use default caches. |
| [#49276](https://github.com/huggingface/transformers/pull/49276), `9d68278db7` | Outside current usage | Multi-image feature ordering during generation expansion; no current contract uses that VLM path. CLIP does not generate. |
| [#49319](https://github.com/huggingface/transformers/pull/49319), `2c8526daa3` | Outside current usage | Chunked prefill with multi-axis position IDs; no current contract exercises it. |
| [#49315](https://github.com/huggingface/transformers/pull/49315), `3758eba846` | Outside current usage | `WatermarkDetector` repeated-ngram counting; no contract calls the detector. |
| [#49149](https://github.com/huggingface/transformers/pull/49149), `59150f0435` | Outside current usage | Shared AutoTokenizer changes are documentation; runtime change is the MistralCommon backend, absent from this fleet. |

**Limits:** all backfill records declare the three current `local_patches`, including the workarounds for #49403 and #49456. Green patched history does not establish that the unpatched bugs were absent; their original scheduled failures remain the detection evidence. No new regression or baseline update from this weekly check. The five shared fixes identify paths outside current coverage, not bugs caught by this fleet.

### SDPA kernel transition: 2026-10-15

The nightlies of October 15–19 report the same 10 CUDA failures: T5's three
supported targets and MusicGen Melody's two targets, each in default and bfloat16.
CPU and CUDA eager checks pass. Replays in the same current environment isolate
the change to [#49517](https://github.com/huggingface/transformers/pull/49517),
which makes position-bias masks contiguous so fused SDPA kernels can run:

| Transformers build | Result | Run |
| --- | --- | --- |
| Parent `ca35879024` | 95 pass, 10 declared unsupported | [parent replay](https://github.com/huggingface/transformers-mic/actions/runs/39216898928) |
| Merge `457293986e` | 85 pass, 10 fail, 10 declared unsupported | [commit replay](https://github.com/huggingface/transformers-mic/actions/runs/39216908189) |

The image also moved from Python 3.10 to 3.11 on October 15. Both replays use
Python 3.11.17, torch 2.11.0+cu128, and the same A10G/driver; the parent still passes.

The [selective math control](../experiments/sdpa_kernel_control.py) keeps the new
contiguous masks and asserts that their values match the parent's construction.
It forces MATH only for attention calls whose old mask stride blocked fused
kernels, leaving other attention calls on their normal dispatch. All **10 old
references pass under the control**, while native dispatch reproduces all 10
nightly failures. Both modes run twice in fresh processes: **20 repeat pairs are
exact**. [GPU run](https://github.com/huggingface/transformers-mic/actions/runs/39221846237),
[validation](../experiments/results/sdpa_kernel_control.json).

This establishes the kernel transition as the cause of the alerts. Default
failures are training losses; bfloat16 also changes T5 label log-probabilities and
the real MusicGen Melody checkpoint's audio codes and waveform probe. The control
does not establish unchanged task quality. Changed references still require review;
tolerances and contract assertions remain fixed.

## Historical regression replay

Regressions fixed since May 2026 ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)): fixtures and real-checkpoint baselines frozen on a good build (`4d536e1f`), run unchanged on the bad build (the fix's parent) and the fixed one (caught = bad flagged, fixed clean); separately, the affected public call on both. Builds before 2026-08-17 need older `tokenizers` (before mid-June, also `kernels`): the fleet lock must pin dependencies, not just models.

| Regression (fix PR) | Real checkpoint | Generic slim checks | Slim fixture, affected public call |
| --- | --- | --- | --- |
| CLIP processor drops `return_tensors` with `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | crash | missed | crash, caught |
| Whisper batch `decode_with_timestamps` ([#47997](https://github.com/huggingface/transformers/pull/47997)) | crash | missed | crash, caught |
| T5 batch `decode` ignores `clean_up_tokenization_spaces` ([#47793](https://github.com/huggingface/transformers/pull/47793)) | caught | missed | caught only with punctuated inputs |
| `additional_special_tokens` lost with legacy tokenizer config ([#47848](https://github.com/huggingface/transformers/pull/47848)) | caught | missed | missed: re-saved tokenizer hides it |
| OLMo tokenizer appends a stray EOS ([#47988](https://github.com/huggingface/transformers/pull/47988)) | caught | missed: re-saved tokenizer hides it | — |
| UMT5 decoder self-attention not causal under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135)) | caught under SDPA only | missed under eager; caught under SDPA on all 6 fixtures (logit delta 1.1–15) | caught under SDPA: future-token leak 2.9–15 vs 2e-6 |
| Processor config helpers return `{}` for `preprocessor_config.json`-only repos ([#47628](https://github.com/huggingface/transformers/pull/47628)) | caught | — | caught with copied processor files; missed with re-saved |
| BigBird/RemBERT/RoFormer causal LM attends to future tokens ([#48785](https://github.com/huggingface/transformers/pull/48785)) | caught: leak 16.8 vs 5e-5 | — | caught: leak 0.0098 vs 2e-7 |
| T5 tied-weights order ([#48238](https://github.com/huggingface/transformers/pull/48238)) | not reproduced with `t5-small` | clean | — |
| MMS `set_target_lang` stale added-token entry ([#47440](https://github.com/huggingface/transformers/pull/47440)) | not reproduced: internal state only | — | not reproduced |
| CLIP conversion mapping at the wrong nesting level ([#46406](https://github.com/huggingface/transformers/pull/46406)) | not reproduced with `CLIPTextModelWithProjection` | — | not reproduced |

- **Generic checks miss most bugs:** forward + greedy generate caught only UMT5, only under default attention; the rest live in processors, tokenizers, and decoding helpers, mostly shared code. Organize contracts around public usage paths.
- **In this original eight-bug sample**, copied-processor fixtures caught every reproduced checkpoint bug once the affected call was made. Later Whisper #48108 shows this does not hold universally ([replay](#whisper-assisted-decoding-suppression)).
- **Copy processor files:** re-saved ones hid three loading regressions.
- **Run the defaults, chosen at load time:** the UMT5 bug exists only under SDPA; `set_attn_implementation("sdpa")` after loading silently leaves T5 on eager ([#49372](https://github.com/huggingface/transformers/issues/49372)).
- **No false alarms:** 10 benign changes on the same paths (RoPE, decoding, image processing, masking, attention refactors, Wav2Vec2's SDPA dispatch), 20 builds × 42 checks, eager and SDPA.

Original sample: CPU only, no GPU-specific case. Later targeted GPU replays follow.

## Overfit or random weights on real regressions

Does a real regression need an overfit fixture, or would random weights catch it too? 2026-10-09, CPU: random twins built with `build_fixtures.py --keep-random --out` (same seed and recipe); each fix replayed against its parent, baselines recorded on the fix.

| Regression | Path | Overfit fixture | Random fixture |
| --- | --- | --- | --- |
| MusicgenMelody drops all conditioning at generation step 0 ([#48679](https://github.com/huggingface/transformers/pull/48679); from #43679, 2026-02-04, to 2026-09-10) | `generate` with text + melody, with text only | caught (codes) | caught (codes) |
| Same, at its introducing commit (`1687954f5a` vs its parent) | same | caught | caught |
| Whisper long-form position ids with left-padded previous-segment prompts ([#48028](https://github.com/huggingface/transformers/pull/48028)) | batched long-form, `condition_on_prev_tokens=True` | missed | missed |
| Whisper assisted decoding re-applies `begin_suppress_tokens` ([#48108](https://github.com/huggingface/transformers/pull/48108)) | `generate(assistant_model=...)` | missed | missed |

- **Still no overfit-only catch.** The backfill had none either: every Whisper failure there was an exception, and the SmolLM fixtures never failed. A history search (2025-10..2026-10) found no silent bug on Whisper's input side (feature extraction, encoder, cross-attention); the silent candidates are in generation.
- **Random weights ignore conditioning, yet see it vanish.** Random Whisper gives every batch row the same tokens, random MusicGen Melody the same codes for every prompt, melody or not. Dropping the conditioning still changes their outputs: the prompt's presence matters even when its content doesn't. Melody swaps and drops show extra overfit sensitivity on selected inputs, but no real overfit-only catch ([controls](#musicgen-melody-conditioning-controls)).
- **An overfit fixture covers only what it was trained on.** The Whisper fixture learned short-form, no-timestamp answers: in long-form timestamp mode it repeats one word to the token cap, and assisted it answers one token and stops, so neither trigger occurred in the initial replay. Retrained on timestamped targets (`<|0.00|> red<|1.00|>`, then 1–4 words), segments close on timestamps and rows differ in length, yet #48028 still did not fire: instrumented, no call in nine input variants, with either fixture, produced a left-padded decoder mask. Controlled unequal first-segment lengths subsequently reached that trigger with both weight kinds ([follow-up](#whisper-left-padded-previous-segment-prompts)).
- **What the MusicGen Melody contract adds** ([contract](../example/hub/MusicgenMelody-integration/integrations/transformers/usage.md), [fixture](https://huggingface.co/hf-internal-testing/mic-musicgen-melody-slim)): a composite model (T5 text encoder, EnCodec, decoder) and classifier-free guidance. The generic shrink rule left EnCodec at 56M of 59M parameters (`num_filters` doubles per stage) and `codebook_dim` coupled to the old width; two reviewed `config_overrides` bring the fixture to 3.5M (17.5 MB). Random weights pass the gate (weakest 0.018) but generate the same codes for every prompt, so the fixture is overfit by reviewed choice (60 steps). On the M5, the fixture runs in ~1 s; the real `facebook/musicgen-melody` (1.5B) takes ~430 s per variant, training step included.

### MusicGen Melody conditioning controls

2026-10-11: [A10G run](https://github.com/huggingface/model-integration-contracts/actions/runs/38136857327), float32, pinned torch 2.11.0+cu128. Replayed #48679 on fix `5bed62964c` and parent `93ebf6b111`, using the real checkpoint, published overfit fixture, and a random twin built on the pinned `8625785e` with seed 17. Six input cases × guidance 1 and 3 × two builds × three weight sets = 72 generation calls, plus two fresh-process fixture repeats. The four fixture runs were also checked on M5 CPU (48 calls). Random twins share the recipe and fixture key but differ in bytes across build platforms; each is reused unchanged within its bad/fix pair. [Raw records, comparisons, and build metadata](../experiments/results/melody_conditioning.json); [harness](../experiments/melody_conditioning.py).

The model-history and issue search found no known MusicGen Melody fix that swaps conditioning rows or drops only the melody. Those changes are **input-perturbation controls**, not historical bugs. Cases use the contract's two prompts and melodies, crossed text/melody pairs, or identical text in both rows. We observe both the contract's score argmaxes and the codes actually sent to EnCodec: generation removes some score positions with its delay mask.

| Comparison | Real checkpoint (A10G) | Overfit fixture (M5 and A10G) | Random twin (M5 and A10G) |
| --- | --- | --- | --- |
| Real #48679, original/crossed prompts, guidance 1 or 3 | caught | caught | caught |
| Real #48679, identical text across rows, guidance 1 or 3 | caught | **missed** | caught |
| Swap melodies, original prompts, guidance 3 | 39/40 decoded code IDs change | 1/40 changes | no change |
| Drop melody, original prompts, guidance 3 | 40/40 decoded code IDs change | 1/40 changes | no change |
| Swap or drop melody, original prompts, guidance 1 | codes change | no change | no change |
| Swap melodies, identical text, guidance 1 or 3 | codes change | no change | no change |

**No real overfit-only catch.** The random twin catches the historical bug in all 12 input/guidance cases; the overfit fixture catches 8 and misses the 4 identical-text cases, on both platforms. Overfitting adds selected conditioning sensitivity but does not improve detection uniformly. The two default CUDA fixture repeats match exactly in scores' code IDs, decoded code IDs, audio heads, and teacher-forced eval loss.

The eval loss is unchanged across the historical fix because its code changes generation only; this is not the contract's training probe. In the controls, original-prompt melody swaps/drops keep both fixtures' eval-loss deltas within `atol=rtol=1e-4`. Their recorded audio heads are all zero throughout the experiment, so they do not establish waveform sensitivity. The observed extra signal is in decoded code IDs; the real checkpoint's guided melody controls also change its waveform head.

To re-run, use the [harness](../experiments/melody_conditioning.py) on a GPU host; its GPU dispatch job is not part of `gpu-checks.yml`.

### Whisper left-padded previous-segment prompts

2026-10-11: [A10G run](https://github.com/huggingface/model-integration-contracts/actions/runs/38138130043), float32, eager attention, torch 2.11.0+cu128. Replayed [#48028](https://github.com/huggingface/transformers/pull/48028) on fix `e12c79c582` and parent `f9b76f2dfc`, with pinned `openai/whisper-tiny`, the published overfit fixture, and a seed-17 random twin built on the pinned `8625785e`. Four input cases × two builds × three weight sets = **24 generation calls**, plus **12 fixed-build calls in fresh processes**. Each weight file is reused unchanged across its pair and repeat. [Raw records and comparisons](../experiments/results/whisper_left_padding.json); [harness](../experiments/whisper_left_padding.py); [analysis](../experiments/summarize_whisper_left_padding.py).

The controlled long-form case uses two 36-second tone inputs, `condition_on_prev_tokens=True`, and timestamps. A public `logits_processor` constrains only the first segment to `<|0.00|> red<|30.00|>` in one row and `<|0.00|> green blue yellow<|30.00|>` in the other. The second segment generates freely. These identical first-segment outputs create previous-token prompts of different lengths without requiring the weights to learn timestamp closure. Observers capture the actual decoder inputs, attention masks, position IDs, and a subset of raw prefill logits; they do not alter the model computation.

| Input case | Real checkpoint: tokens change | Overfit fixture: tokens change | Random twin: tokens change |
| --- | --- | --- | --- |
| Short-form, ordinary unpadded prefix | no | no | no |
| Short-form, caller-supplied left-padded decoder prompts | yes | **no** | yes |
| Long-form, unforced repeated tones | no | no | no |
| Long-form, controlled unequal first segments | **yes** | **yes** | **yes** |

Both padded cases reach one left-padded decoder prefill per model on each build. In controlled long-form, the first segments match exactly across builds. Only the shorter-prompt row's next segment changes; the longer row is unchanged. Its sampled prefill-logit delta is **12.923 checkpoint, 2.468 overfit, 0.418 random**, all outside `atol=rtol=1e-4`. The fixed build supplies position IDs equal to the decoder mask's cumulative token count minus one, with zero at padding positions; the bad build supplies none. The unpadded short-form control has exactly zero logit delta. Unforced repeated tones reach no left-padding in any weight set.

**No overfit-only catch:** both fixtures catch this real bug in freely generated long-form tokens once the input setup reaches its trigger. The shorter caller-supplied prompt probe changes overfit logits (delta 1.271) but leaves its generated tokens unchanged, while checkpoint and random tokens change. Token and logit detection are reported separately. All 12 fixed-build repeats match exactly in tokens, segment metadata, decoder traces, and sampled logits (excluding timing).

**Limits:** this is a targeted historical replay with a constrained first segment, not evidence that natural audio reliably reaches the trigger or that random weights transcribe usefully. Caller-supplied `decoder_input_ids` is supported but deprecated by Whisper's public `generate`; the controlled long-form case exercises its internally constructed previous-segment prompts. No fixture retraining, published contract change, or baseline update was needed for the experiment.

To re-run, use the [harness](../experiments/whisper_left_padding.py) on a GPU host, then the [analysis](../experiments/summarize_whisper_left_padding.py) on its output; the GPU dispatch job is not part of `gpu-checks.yml`.

The controlled continuation is now in the [Whisper contract](../example/hub/Whisper-integration/integrations/transformers/usage.md), as `continued_longform_tokens`. Published contract revision: `a4ed88605df499c8d655eff2838001279c3fe5e5`; fixture weights are unchanged. **14 baselines** (two targets × CPU ARM default/eager, CPU x86 default/eager, CUDA default/eager/bfloat16) preserve all old outputs and add the new one. Each recording includes a fresh-process stability check; additional isolated checks pass on ARM and the [A10G worker](https://github.com/huggingface/model-integration-contracts/actions/runs/38140685813). The contract itself, through `run.py`, passes on historical fix `e12c79c582` and fails only `continued_longform_tokens` on parent `f9b76f2dfc` with the published fixture. [Validation records](../experiments/results/whisper_contract_validation.json). Baselines honestly retain `contract_revision: local` where recorded before publication; the fleet pins the identical published source.

### Whisper assisted-decoding suppression

2026-10-11: replayed [#48108](https://github.com/huggingface/transformers/pull/48108), fix `b6688043ff` and parent `2b296305d7`. The earlier contract probe uses `WhisperForCausalLM`; the upstream test uses `WhisperForConditionalGeneration` as its assistant. The latter gives the assistant its own encoder, which also permits an assistant of a different hidden width.

The [first A10G replay](https://github.com/huggingface/model-integration-contracts/actions/runs/38140685813) used the pinned real tiny checkpoint, published overfit fixture, and a random twin, with each main model's own weights as the assistant. Five inputs (tones, silence, two text-prompt controls, and speech) × greedy/causal-assistant/encoder-decoder-assistant × two builds × three weight kinds = **90 generation calls**, float32, eager, torch 2.11.0+cu128. Speech is the upstream test's first LibriSpeech dummy sample, `1272-128104-0000`, from dataset revision `5be91486e11a2d616f4ec5db8d3fd248585ac07a`; its original FLAC and [provenance](../experiments/audio/1272-128104-0000.json) are frozen separately from the synthetic contract inputs.

| Main / assistant | Bad versus fixed | Verdict |
| --- | --- | --- |
| Real tiny / causal tiny | All five inputs unchanged | The current contract's assistant type misses this bug on these inputs. |
| Real tiny / encoder-decoder tiny | Speech adds `Thank you.` only on the bad build; the three-word text-prompt control also changes | Caught in generated tokens. Fixed outputs match greedy; the two bad assisted outputs do not. |
| Overfit / causal or encoder-decoder twin | All five inputs unchanged | Missed. The short tone-label answers do not expose the token regression, even on speech. |
| Random / causal or encoder-decoder twin | All five inputs unchanged | Missed. |
| Real large-v2 / encoder-decoder tiny, float16 | Four of five inputs change; speech again adds `Thank you.` only on the bad build | Reproduces the upstream model pairing and dtype. Fixed outputs match greedy on all five inputs. |

The [large-v2 replay](https://github.com/huggingface/model-integration-contracts/actions/runs/38141596107) pins `openai/whisper-large-v2@ae4642769ce2ad8fc292556ccea8e901f1530655`, with the fleet's tiny checkpoint as assistant: five inputs × greedy/assisted × two builds = **20 more calls**. Greedy outputs are identical across both builds for every weight set and input. Each weight file is reused unchanged across its pair. Observers return the original logits processors unchanged; on the bad build, cleared suppression settings are restored and extra beginning-token suppression reaches assisted generation. These observations are diagnostic; the catch is the actual generated token/text difference. [Harness](../experiments/whisper_assisted_suppression.py); [raw records and comparisons](../experiments/results/whisper_assisted_suppression.json).

The routine contract is published at `77fb2ffa32645b4e02243c7a149edf218602684b`, with **seven reviewed baselines**: ARM default/eager, x86 default/eager, CUDA default/eager/bfloat16. Every recording includes a fresh-process stability check. The [A10G worker](https://github.com/huggingface/model-integration-contracts/actions/runs/38145323837) passed all five supported worker variants and caught the bad/fix pair through `run.py` on CPU and CUDA eager; the published contract also caught it on ARM. Input bytes and provenance are part of the contract pin. Worker isolated checks took 15–16 s on CPU and 9–10 s on CUDA. Baselines retain their pre-publication `unpublished` provenance. [Validation records](../experiments/results/whisper_speech_contract_validation.json).

**Limits and decision:** no overfit-only catch. The checkpoint-only [WhisperSpeech contract](../example/hub/WhisperSpeech-integration/integrations/transformers/usage.md) now adds frozen speech with an encoder-decoder assistant to routine nightly coverage. Readable short answers do not establish longer-speech or assisted EOS coverage. Cross-width causal assistants are unsupported, so large-v2 uses the upstream encoder-decoder pairing. Fixture-only runs explicitly exclude this path; no fixture was retrained. Stop broad weight-pair searches; reserve controls for specific detection/repair questions ([comparison](fixtures.md#overfit-versus-random-what-can-we-conclude)).

Re-run the targeted GPU jobs:

```sh
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=whisper-contract
gh workflow run gpu-checks.yml --repo huggingface/model-integration-contracts -f matrix=whisper-assist-large
```
