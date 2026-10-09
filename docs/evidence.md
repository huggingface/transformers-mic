# MICs: evidence so far

Results from local experiments, the regression replay, and the shadow backfill over Transformers `main`. Fixture experiments: [Small fixtures](fixtures.md); mechanics: [technical design](design.md).

## Summary

**Detection**

- **Contracts must call what users call:** generic checks missed every processor and tokenizer bug; the affected call caught all of them ([replay](#historical-regression-replay)).
- **Five months of `main`: 1 caught, 4 missed, 1 false alarm**; the misses were calls no contract made ([backfill](#shadow-backfill)). With those calls added, all five are caught, with no new false alarm, and the new Whisper call found a live bug, [#49403](https://github.com/huggingface/transformers/issues/49403) ([details](#after-adding-the-missed-calls)). **A call added for one bug caught another nobody targeted:** Whisper's assisted `generate` raised for 41 commits after [#48981](https://github.com/huggingface/transformers/pull/48981), from its introducing commit to the fix [#49075](https://github.com/huggingface/transformers/pull/49075) ([third round](#third-round-second-round-calls-and-bfloat16-smollm)).
- **Scheduled runs caught a regression the day it merged:** #48282 (2026-10-08) made `generate()` raise on every original Whisper checkpoint; the next nightly and shadow runs failed Whisper everywhere ([#49456](https://github.com/huggingface/transformers/issues/49456), worked around in `local_patches`). The PR's CI could not see it ([why](https://github.com/huggingface/transformers/issues/49456#issuecomment-6076843277)): model tests build configs without `suppress_tokens`, pipeline tests passed their own `generation_config`, and slow `generate()` tests were not in its `run-slow` list. The contract calls `model.generate()` on the checkpoint's config, as users do.
- **The runner found a real bug:** seven audio models leave `masked_spec_embed` uninitialized ([#49385](https://github.com/huggingface/transformers/issues/49385), fix [#49386](https://github.com/huggingface/transformers/pull/49386)); the slim Wav2Vec2 fixture catches it every run ([runner](../example/README.md#runner-prototype)).
- **Golden outputs catch silent backend bugs:** Transformers.js on WebGPU returned all-zero fp16/q4f16 logits with no error ([repro](../experiments/webgpu_browser/README.md)).

**Fixtures, baselines, cost**

- **Fixtures:** synthetic-fault detection, weight choice, and the architecture sweep are in [Small fixtures](fixtures.md#tldr). Publish them, don't rebuild: x86 rebuilds differed from the M5 bytes.
- **Baselines are per platform and dtype; fixtures are shared** ([GPU](#on-gpu), [bfloat16](#bfloat16-on-cpu)). So split, 16 targets × 3 variants reproduced at `1e-4` on A10G ([numbers](../experiments/results/runner.json)); ten benign changes raised no false alarm.
- **Batching is the cost lever:** eight fixture checks take 4.9 s in one process vs 27.4 s one per process (Mac); 13.1 s vs 81.1 s (GPU runner).

**Scope:** ten families, eleven candidate regressions (eight reproducible), ten benign changes, five months on six contracts; replay and backfill on CPU, one GPU for stability and cost ([cases.json](../experiments/cases.json)).

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

**Cost per check:** fresh process, three runs, warm cache ([cost.json](../experiments/results/cost.json)). "Work" = input prep, load, first inference, comparison, (slim) training probe; imports add ~2.4 s per process.

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

- **Process start dominates**, so run many contracts per process. On these small cached checkpoints, little local time is saved.
- **Savings are transfer and size:** slim weights are 10–1,300× smaller. Slow tests for these families load ~144 GB, mostly not CPU-runnable.
- **Memory:** 0.5–1.3 GB peak RSS slim (mostly framework), 0.7–2 GB real.

### bfloat16 on CPU

SmolLM-135M and -360M ship bfloat16, which users get by default, but fixtures used to be float32. [`build_fixtures.py`](../example/build_fixtures.py) now saves each fixture in its source's dtype; only proto-SmolLM-slim changed (float32 fixtures rebuild byte-identical).

- **Random weights pass the gate but are useless for generation:** logits moved ≥ 0.24 under every perturbation (gate: 1e-3), yet all four prompts generated " is is is is". Overfit in float32, then cast: " Paris…", " Berlin…", " Rome…", " London…" (80 steps; 3.2M parameters, 6.1 MB).
- **The training batch has an irreducible loss:** rows share "The capital of" then name a country, so ln 4 at that position can't be learned; training plateaus at 0.25, so the SmolLM recipe sets `target_loss: 0.3`.

**Native bfloat16 changes real-checkpoint results, not the fixture's.** GitHub runners at `8625785e`, EPYC 7763 (`avx2` only; run 37916317747) vs EPYC 9V45 (`avx512_bf16`; 37916321632):

| Target (bfloat16) | Max logit delta | Loss delta | Generated tokens |
| --- | ---: | ---: | --- |
| proto-SmolLM-slim, SDPA / eager | 0 / 0 | 1.5e-5 / 1.9e-5 | same |
| SmolLM-135M, SDPA / eager | 0.25 / 0.25 | 0.0074 / 0.011 | same / **differ** |
| SmolLM-360M, SDPA / eager | 0.125 / 0.25 | 0.010 / 0.013 | same |

- bfloat16 runs on x86 CPUs with `avx512_bf16` or `amx_bf16` use the `cpu-x86_64-bf16` platform, which the [runner](../example/run.py) picks from `/proc/cpuinfo`. GitHub's x86 runners are mixed (one of three shadow runs on 2026-10-08/09 had `avx512_bf16`).
- The fixture agrees across both CPUs (hidden size 64 likely never reaches the native kernels) but follows the same rule.
- Emulated bfloat16 on x86 and the M5 generate the same tokens on every SmolLM target. The 24 shared float32 targets agree across the 7763, 9V45, and 7R32.
- Not tested: Intel (`amx_bf16`) vs AMD (`avx512_bf16`), needed before PR CI uses Intel runners.

### On the Transformers PR CI runner

Phase 1 of the [CI proof](transformers-ci-proof.md): the eight fixture contracts on `aws-m8i-l-cache` in `huggingface/transformers-torch-light`, the runner and image of Transformers' PR test jobs, from the `contracts-proof` branch of `huggingface/transformers` ([run 37932247464](https://github.com/huggingface/transformers/actions/runs/37932247464), 2026-10-09). Transformers `8625785e`, the baselines' build.

| Packages | Differ from the lock | Default | Eager | Run (8 fixtures, one process) |
| --- | --- | ---: | ---: | ---: |
| Image as is | Python 3.11.17, torch 2.14.0, numpy 2.3.5, tokenizers 0.23.2, + 8 more | 8/8 | 8/8 | 37–39 s |
| Lock's pins installed | Python 3.11.17 only | 8/8 | 8/8 | 36–57 s |

- **The `cpu-x86_64` baselines hold on Intel**, recorded on AMD EPYC 7R32/7763/9V45; no pin install needed (16 s if it were).
- **Intel `amx_bf16` matches AMD `avx512_bf16` on the bfloat16 fixture** (Xeon 6975P-C against the 9V45's `cpu-x86_64-bf16` baseline). Weak evidence: the fixture also matches emulated bfloat16. Real bfloat16 checkpoints on Intel are untested.
- **Slower than a 2-core GitHub runner** (~13 s) on 128 cores, presumably thread oversubscription on tiny tensors; `OMP_NUM_THREADS` untested.
- Installing Transformers and the runner's dependencies over the image: 14 s.

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

Imports take ~4.9 s (2× the Mac), so batching matters more; slim fixtures need < 200 MB, so many share a GPU. Not measured: CPU-only CI runners, setup, downloads, cold cache, other GPUs, multi-GPU.

## Shadow backfill

[`shadow-backfill.yml`](../.github/workflows/shadow-backfill.yml) ran the frozen fixture contracts on every first-parent `main` commit from 2026-05-02 to 2026-10-07: 1,468 × 8 = 11,744 results, no runner error, ~15 s per commit on GitHub x86 with `cpu-x86_64` baselines. Commits that reject a pin use their newest accepted version (`tokenizers` 0.22 before 2026-08-17; no effect). Results, with `selected_by`, are on `shadow-results` (`runs/backfill/`).

Every change in outcome (all else passed on every commit):

| Fixture | Commits | Differs from today's baseline | Starts | Ends | Verdict |
| --- | --- | --- | --- | --- | --- |
| `proto-t5-small-slim`, `proto-umt5-small-slim` | 761 | `runtime`: eager, not SDPA | window start | [#47014](https://github.com/huggingface/transformers/pull/47014) (`560f36cc56`, 2026-07-30), "Enable SDPA for T5" | Intended |
| `proto-umt5-small-slim` | 597 | `label_logprobs` off by 1.22, training loss by 0.20 | #47014 | [#49135](https://github.com/huggingface/transformers/pull/49135) (`d7ba937671`, 2026-09-28) | **Regression caught at its introducing commit** (UMT5 decoder not causal under SDPA) |
| `proto-wav2vec2-base-960h-slim` | 513 | training loss off by 0.39 | window start | [#46259](https://github.com/huggingface/transformers/pull/46259) (`f196737b0c`, 2026-07-03) | **Benign change flagged:** load-time RNG stream shifted |

- **Caught at introduction, two months before the fix:** #47014 touched shared code, so it selected every contract; a reviewer would have seen UMT5 move by 1.22 beside T5's intended runtime change.
- **One false alarm:** the fixture omits `masked_spec_embed`, as the checkpoint does, to catch [#49385](https://github.com/huggingface/transformers/issues/49385), so its loss follows the RNG stream. Cost: one baseline update.
- **Four misses, one cause: no contract made the call.** CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)), processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)), Whisper timestamp batch decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)), T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)).
- **Selection missed nothing:** all three outcome-changing commits selected the affected contract. It selected none on 768 commits (52%), all on 679 (46%), some on 21.
- **No other drift** across three CPUs (EPYC 7763, 9V45, 9V74) and two `tokenizers` versions.

### After adding the missed calls

Calls added via the [improvement loop](improving-contracts.md#what-happened-the-first-time); no existing baseline moved on any platform (98 candidates). Re-backfilled over 1,470 commits (2026-05-01 to 2026-10-08, [run](https://github.com/huggingface/model-integration-contracts/actions/runs/37664740099)):

| Regression | Fixture result while present | Ends at (fix) |
| --- | --- | --- |
| CLIP `text_kwargs` ([#46195](https://github.com/huggingface/transformers/pull/46195)) | `proto-clip-vit-base-patch32-slim` errors, 206 commits | `420c80aad0` |
| Processor config helpers ([#47628](https://github.com/huggingface/transformers/pull/47628)) | CLIP and Wav2Vec2 config keys empty, 595 and 801 commits | `c62820a73c` |
| Whisper timestamp decoding ([#47997](https://github.com/huggingface/transformers/pull/47997)) | `proto-whisper-tiny-slim` errors, 890 commits | `64b6595f0c` |
| T5 batch decoding ([#47793](https://github.com/huggingface/transformers/pull/47793)) | T5 and UMT5 `decoded_roundtrip` keeps spaces before punctuation, 1,067 commits | `4ff98bf236` |
| UMT5 causal mask ([#49135](https://github.com/huggingface/transformers/pull/49135)) | caught from its introducing commit #47014 | `d7ba937671` |

Each fails exactly until its fix (four predate the window); no new false alarm, no added cost (~13 s per commit). **Post-hoc:** the calls were written knowing the bugs, so this shows the loop closes gaps, not the rate on unknown bugs.

### Third round: second-round calls and bfloat16 SmolLM

After the [second round](improving-contracts.md#the-second-round-triaging-the-model-tier-misses) added Whisper `max_length_tokens`, `assisted_tokens`, `longform_tokens`, and proto-SmolLM-slim moved to bfloat16 ([why](#bfloat16-on-cpu)). 1,480 commits, 2026-05-01 to 2026-10-08 ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37930845944)); every other segment as in the second round.

| Window | Commits | Fixture result | Verdict |
| --- | ---: | --- | --- |
| #47997 fix (`64b6595f0c`) to [#48000](https://github.com/huggingface/transformers/pull/48000) (`a5f41e095a`), 2026-08-16/17 | 2 | `proto-whisper-tiny-slim` errors in the assisted call: `UnboundLocalError: is_updated` | Caught; the call was added for this bug (post-hoc) |
| [#48981](https://github.com/huggingface/transformers/pull/48981) (`87d34bfa80`, 2026-09-22, "Always correctly restrict assisted decoding with max length/eos token") to [#49075](https://github.com/huggingface/transformers/pull/49075) (`010f0ca2ec`, 2026-09-24, "Fix assisted eos token condition") | 41 | same call errors: `RuntimeError: a Tensor with 2 elements cannot be converted to Scalar` in `_assisted_decoding` | **New catch, not targeted: caught at its introducing commit** |

- **Not post-hoc for #48981:** `assisted_tokens` was added for #48000; #49075 touches only shared generation code, so `find_misses.py`'s model tier never listed it. #48981 changes `generation/` (shared), so selection would have run Whisper on that PR; it also deleted 9 lines from `tests/generation/test_utils.py`.
- **bfloat16 SmolLM fixture: no false alarm**, all 1,480 commits, on `cpu-x86_64` (888: EPYC 7763, 9V74 without native bfloat16, Xeon 8370C/8573C) and `cpu-x86_64-bf16` (592: EPYC 9V45, 9V74 with `avx512_bf16`).
- **Cost:** 6.7 s per commit summed over targets, vs 6.2 s (the three added Whisper calls).

## Historical regression replay

Regressions fixed since May 2026 ([freeze/check](../experiments/replay.py), [per-case driver](../experiments/replay_case.py), [usage repros](../experiments/usage_repros.py)): freeze fixtures and real-checkpoint baselines on a good build (`4d536e1f`); run them unchanged on the bad build (fix's parent) and the fixed build (caught = bad flagged, fixed clean); separately run the affected public call on both. Builds before 2026-08-17 need older `tokenizers` (and before mid-June, `kernels`): the fleet lock must pin dependencies, not just models.

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
- **Making the affected call is enough:** the fixture then caught every bug the checkpoint did.
- **Copy processor files:** re-saved ones hid three loading regressions.
- **Run the defaults, chosen at load time:** the UMT5 bug exists only under SDPA; `set_attn_implementation("sdpa")` after loading silently leaves T5 on eager ([#49372](https://github.com/huggingface/transformers/issues/49372)).
- **No false alarms:** 10 benign changes on the same paths (RoPE, decoding, image processing, masking, attention refactors, Wav2Vec2's SDPA dispatch), 20 builds × 42 checks, eager and SDPA.

Sample: CPU only, no GPU-specific case.
