# Problems and opened questions

The experiments support a CI pilot; removing tests or requiring contracts needs more evidence. A slim pass never means the full model passes ([CI integration](slim-models-in-transformers-ci.md)). Six problems stand in between:

| # | Problem | Kind |
| --- | --- | --- |
| 1 | [Contracts only catch what they call](#1-contracts-only-catch-what-they-call) | Limit of the approach |
| 2 | [A fixture does not cover every path of its checkpoint](#2-a-fixture-does-not-cover-every-path-of-its-checkpoint) | Limit of the approach |
| 3 | [The sensitivity gate checks the wrong thing](#3-the-sensitivity-gate-checks-the-wrong-thing) | Prototype gap |
| 4 | [Baselines multiply and need review rules](#4-baselines-multiply-and-need-review-rules) | Open cost |
| 5 | [Cross-repository coordination and total cost](#5-cross-repository-coordination-and-total-cost) | Open cost |
| 6 | [Runner validation gaps](#6-runner-validation-gaps) | Prototype gap, closed |

## 1. Contracts only catch what they call

**Problem.** Fixtures catch real regressions only when the contract makes the affected call. The detection rate on unseen bugs is unknown.

**What we know.**

- [Replay](evidence.md#historical-regression-replay): all eight reproducible regressions caught once the affected call was made; generic forward/generate missed most.
- [Backfill](evidence.md#shadow-backfill), 1,468 commits: frozen contracts caught 1 of 5 in-family regressions, missed 4 they didn't call; with those calls added, 5 of 5, but post-hoc.
- One catch was not post-hoc: a call added for #48000 caught #48981 from its introducing commit ([third round](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)). Scheduled runs caught #48282 the night it merged ([#49456](https://github.com/huggingface/transformers/issues/49456)).

**Next.** Freeze everything and measure on regressions not used to design it (the shadow trial). Report detections, misses, unsupported, and false alarms separately, post-hoc catches apart ([improvement loop](improving-contracts.md)).

**Open questions.**

- How are representative inputs and API options chosen before a bug is known?
- What must a contract demonstrate to replace a given fast test?

## 2. A fixture does not cover every path of its checkpoint

**Problem.** Same API, different code paths: smaller dimensions, short inputs, and CPU each skip paths the checkpoint takes.

**What we know.**

- The fixture key guarantees identical processor files, retained config, and dtype, not behavior that depends on shrunk dimensions.
- CPU excludes GPU paths; short inputs leave long context untested.
- Whisper #48108 changes real-checkpoint assisted speech outputs but neither slim weight kind. The checkpoint-only WhisperSpeech contract covers frozen speech with an encoder-decoder assistant in nightly; fast fixture checks still miss this bug ([replay](evidence.md#whisper-assisted-decoding-suppression)).
- Fixtures load in their source's dtype, but at hidden size 64 bfloat16 may not reach the native CPU kernels the real checkpoint hits: mic-SmolLM-slim agrees across native and emulated bfloat16 while SmolLM-135M does not ([evidence](evidence.md#bfloat16-on-cpu)).

**Next.** Per contract, record what the fixture covers and which full-model or hardware job covers the rest. Choose PR variants explicitly, for cost versus signal.

**Open questions.**

- Which cheap dtype, attention, or input variants add signal?
- When should a PR also trigger a full-model `run-slow` before merge?

## 3. The sensitivity gate checks the wrong thing

**Problem.** The [gate](../example/perturb.py) shows a perturbed input moves *some* logit, not that a broken mask, detached module, or bad preprocessing moves the outputs the contract *compares*.

**What we know.**

- It measures max movement over all logits; contracts may compare only selected logits or label log-probs.
- The [builder](../example/build_fixtures.py)'s threshold uses only absolute tolerances, not relative ones or platform overrides.
- Overfitting helps on synthetic Whisper faults; random fixtures caught all eight historical regressions too ([weights](fixtures.md#choosing-the-weights)).
- Still no real overfit-only catch ([replays](evidence.md#overfit-or-random-weights-on-real-regressions)): both catch Melody #48679 and controlled Whisper #48028; both miss #48108. On identical-text Melody inputs, random catches #48679 while overfit misses it. Readable outputs do not guarantee broader detection.
- The gate can pass while generation ignores the input: random SmolLM and MusicGen Melody fixtures move their logits yet generate one answer for every prompt, so both are overfit by reviewed choice.

**Next.** Measure perturbations through the contract's own outputs and rules, with a margin above normal variation; keep a few injected faults to check sensitivity becomes detection.

**Open questions.**

- What margin suffices for given outputs and tolerances?
- Which injected faults add detection, and when is overfitting worth it? Conditioning-row swaps are a candidate; current controls show selected overfit sensitivity, not a real overfit-only catch.

## 4. Baselines multiply and need review rules

**Problem.** Baselines grow with targets × frameworks × hardware × dtype × attention. Re-recording is easy, so without review criteria a regression can be approved.

**What we know.**

- Whisper already needs per-CPU-architecture baselines.
- [A10G results](../example/README.md#on-gpu) hold only for that GPU and stack.
- Five months of `main`: 8 `cpu-x86_64` fixture baselines needed 2 updates (benign RNG shift, intended T5 SDPA).
- bfloat16 real checkpoints differ on native-bfloat16 CPUs, now a platform of their own (`cpu-x86_64-bf16`). Intel `amx_bf16` (the PR CI runner) matched the AMD `avx512_bf16` baseline on the bfloat16 fixture; real bfloat16 checkpoints on Intel are untested.
- Each new contract adds baselines on four platforms and three variants: MusicGen Melody's real checkpoint needs a large runner even for `cpu-x86_64`.
- Fresh-process reproduction proves repeatability, not correctness.

The [GPU upgrade matrix](evidence.md#gpu-dependency-upgrades) repeats exactly within environments but flags NumPy-driven Wav2Vec2 preprocessing changes and CLIP/Wav2Vec2 drift under a newer CUDA bundle. The [full GPU normalization control](evidence.md#wav2vec2-normalization-control) restores all Wav2Vec2 NumPy outputs exactly; an upstream normalization fix and CUDA-bundle investigation remain. Repeatability alone does not justify new baselines or wider tolerances.

**Next.** Define environment classes, calibrate on representative runners, track churn and review time, budget the slim PR matrix apart from the full-model one. Each baseline update needs evidence beyond repeatability ([review process](design.md#revisions-and-baselines)).

**Open questions.**

- How narrowly must hardware and dependency environments be grouped?
- What evidence approves an output change, and how much upkeep is acceptable?

## 5. Cross-repository coordination and total cost

**Problem.** Contracts live on the Hub, expectations in framework repos: one change can need a contract revision, a fleet-lock bump, and baselines, tested together and reproducible for old revisions.

**What we know.**

- [Cost](evidence.md#stability-and-cost): fixtures are far smaller and batching matters; fleet-scale transfer isn't measured.
- Human cost (authoring, shrink rules, fixture repair, baseline review, coordination) isn't measured; moving work doesn't reduce it.

**Next.**

- Run one contract + framework update end to end, with review and rollback.
- Track maintainer time alongside execution, setup, and download cost.
- Before retiring a test, map its inputs, assertions, variants, hardware, and frequency to the replacement; a migrated full-model test keeps its full-model run.

**Open questions.**

- How are contract revisions and framework changes tested and reviewed together?
- Does the pilot reduce total maintainer effort, and which tests can go?

## 6. Runner validation gaps

**Closed 2026-10-08.** The [runner](../example/run.py) silently dropped checks in five ways; each now fails.

| Gap | Before | Now |
| --- | --- | --- |
| Missing output | Skipped if absent from result and baseline. | Every manifest output required (training ones only where the framework declares `training`); `--record` refuses incomplete candidates. |
| Empty gradient checks | `{}` passed `all_true`. | Non-empty dict, names matching the baseline, each `true`. |
| Changed shape | Flattened: (1, 4) matched (2, 2). | `allclose` outputs must match the baseline's shape, ragged lists included. |
| NaN gradients | NaN counted as nonzero. | Every training contract also requires `torch.isfinite(grad).all()`. |
| Nothing ran | All-`unsupported` runs exited 0. | Skips record why (`declared`, `hardware`, `variant`). No target run or no `--contract`/`--target` match fails. `--require-all` (shadow, backfill, nightly) fails on undeclared skips. |

**Still open.**

- GitHub runners (7 GB) skip SmolLM-1.7B and MusicGen Melody's checkpoint when recording, so only `machine=gpu-host` enforces `REQUIRE_ALL`; picking runner classes from [`resources`](design.md#minimum-hardware) (not implemented) fixes it.
- No per-framework exception list for non-training outputs; add one when needed.
