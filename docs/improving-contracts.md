# Improving contracts continuously

A contract catches only what it calls: over five months of `main`, frozen contracts caught one regression and missed four whose calls they didn't make ([evidence](evidence.md#shadow-backfill)). Adding those calls found a live bug ([#49403](https://github.com/huggingface/transformers/issues/49403)).

The loop: find misses, add the user-facing call, prove it on history, keep false alarms out. Tools: [shadow](../.github/workflows/shadow.yml), [backfill](../.github/workflows/shadow-backfill.yml), [nightly](../.github/workflows/nightly.yml), [`find_misses.py`](../example/find_misses.py), [`replay.py`](../example/replay.py), [`review_candidates.py`](../example/review_candidates.py).

## Where new checks come from

| Source | How to find it | Example |
| --- | --- | --- |
| A fix no contract noticed | `find_misses.py`: fixes in a contract's code with no result change | #47997 touched `models/whisper`; no Whisper result moved |
| A bug or fix in a contract's family | Issues/PRs in its model dirs or shared processing, tokenization, generation, loading | #46195 (CLIP `text_kwargs`), in `processing_utils.py` |
| Documented usage the contract skips | Docs, card snippets, task guides | Whisper `generate(..., return_timestamps=True)` |
| A false alarm | A failure classified benign | #46259: load-time RNG shift moved Wav2Vec2's loss |
| A silent bug in a family no contract covers | Fix PRs whose bug changed outputs without raising | #48679: MusicGen Melody dropped all conditioning for seven months; became the MusicGen Melody contract |

`find_misses.py` tiers:

- **Model:** touches the contract's own model dir. Review all (17 on the first five-month backfill, incl. #47997, #47440).
- **Shared** (`--shared`): touches shared code, matches every contract. Review by subject; three of four known misses (#46195, #47628, #47793) are here.

A candidate is a question, not a verdict.

## Adding a check

1. **Reproduce on the real checkpoint first**, through the public call, on bad and fixed builds ([replay](evidence.md#historical-regression-replay)). No checkpoint failure → stop.
2. **Write it as user code:** documented API, default settings, short, deterministic, in its own `usage.md` section.
3. **Don't move existing outputs.** Place it before training if it draws no random numbers (tokenization, decoding, config reads, greedy generation); otherwise after the outputs it could affect, and re-review them.
4. **Declare its output** in the manifest: `exact` for tokens, text, labels, keys; `allclose` only for floats.
5. **Prove it on the bug** with [`replay.py`](../example/replay.py): fixture fails on bad, passes on fixed. If only the checkpoint differs, keep it as full-model coverage (nightly, `run-slow`), not PR coverage.
6. **Record baselines on every platform** (each rerun in a fresh process):
   - `cpu-arm64`: `run.py --record`
   - `cpu-x86_64`: [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) `machine=gpu-host` (EPYC 7R32, no native bfloat16)
   - `cpu-x86_64-bf16` (bfloat16 targets only): `cpu-baselines.yml machine=github`. The CPU kind isn't selectable: rerun until the job lands on one with `avx512_bf16`/`amx_bf16` (the job logs CPU flags; `run.py` picks the platform from `/proc/cpuinfo`).
   - `cuda`: [`nightly.yml`](../.github/workflows/nightly.yml) `mode=record`
   - For a change not yet on `main`, run both workflows with `--ref <branch>` and merge only once the baselines are approved: scheduled runs fail on a missing baseline.
7. **Review.** [`review_candidates.py`](../example/review_candidates.py) classifies each candidate **unchanged** (new outputs only), **changed** (needs a reason), or **new**. `--approve` renames only unchanged and new; changed is a person's call.
8. **Measure on history.** Re-run the backfill; record catches, new false alarms, and per-commit cost in [evidence](evidence.md), post-hoc detections apart.
9. **Fails on today's `main`?** Live bug: report upstream and keep the check with a `local_patches` workaround (recorded per result) here. In the framework's own CI, use a narrowed `known_failures` entry instead (`match` on the error, or `skip_outputs`), never the whole-target form, which hides the contract's other checks. Drop the patch and re-record once fixed.

## Adding a contract

Same loop, one level up: a new usage path, as for MusicGen Melody (2026-10-09, after #48679).

1. **Write the contract** (`example/hub/<Name>-integration/`: manifest, `usage.md`, README) and an `unpublished` lock entry with `path`.
2. **Build the fixture** with [`build_fixtures.py`](../example/build_fixtures.py). Check its size and what generation does: the gate passed for MusicGen Melody, but random weights gave every prompt the same codes, so the recipe overfits by reviewed choice; the generic shrink rule left 59M parameters, so `config_overrides` set two EnCodec fields (3.5M).
3. **Prove it on the bug** that motivated it, on the fix and its parent, and at the introducing commit when the build still runs.
4. **Record `cpu-arm64`** on the pinned build, approve, then publish: `publish_fixtures.py` (refuses files that don't match the lock's hash), then `publish_contracts.py`; both pin revision and hash in the lock.
5. **Record the other platforms from a branch** (step 6 above) and merge once approved.

## Keeping false alarms down

Log one decision per benign failure:

- **Keep and update the baseline** if rare and the check guards something real: Wav2Vec2's loss follows load-time randomness because the fixture omits `masked_spec_embed`, which is how it catches [#49385](https://github.com/huggingface/transformers/issues/49385). One update in five months.
- **Make it robust:** seed closer to use, compare a discrete outcome, or avoid unrelated randomness.
- **Drop it** if it keeps failing benignly and never caught anything.

## Cadence

| When | What |
| --- | --- |
| Each shadow/nightly failure | Classify: regression, intended (update baseline with reason), or benign. |
| Weekly | `find_misses.py` on the week's shadow results; triage model tier; open contract changes. |
| After a contract change | Re-run the backfill; update evidence. |
| After an upstream fix of a found bug | Drop the `local_patches`/`known_failures` entry; re-record. |

## What happened the first time

Calls added for the four misses: CLIP per-modality `text_kwargs` and `get_image_processor_config`; Whisper timestamp generation, `batch_decode(..., decode_with_timestamps=True)`, and `get_feature_extractor_config`; the same config read for Wav2Vec2; T5 `decode` on a punctuated batch.

- **No existing output moved:** `review_candidates.py` confirmed all 98 re-recorded baselines (18 `cpu-arm64` ones only gained outputs).
- **Live Whisper bug:** batch decoding the tensor `generate` returns merges rows or raises ([#49403](https://github.com/huggingface/transformers/issues/49403); `local_patches` `whisper-decode-tensor-batch`).
- **`find_misses.py` lists all four misses** from the first backfill's results.
- **The backfill re-run caught all four** until their fixes, no new false alarm or cost ([details](evidence.md#after-adding-the-missed-calls)). Post-hoc: the loop closes gaps; it doesn't measure the unknown-bug rate.

## The second round: triaging the model-tier misses

On the post-round-one backfill (2026-10-08), `find_misses.py --ref origin/main` lists 16 model-tier candidates. Each was read; those on a usage path were reproduced on the real checkpoint before and after the fix.

| Verdict | Fix | What it fixed | Seen by |
| --- | --- | --- | --- |
| Covered | #47150, #46795 (July) | Whisper `generate(max_length=N)` miscounted the forced prefix; `max_length=1` raised | fixture and checkpoint |
| Covered | #48000 (2026-08-17) | `WhisperForCausalLM` assistant raised `UnboundLocalError` | fixture and checkpoint |
| Initially checkpoint only; fixture follow-up caught it | #48028 (2026-08-19) | Wrong positions for left-padded previous-segment prompts | both fixtures with controlled unequal first segments, added 2026-10-11 ([follow-up](evidence.md#whisper-left-padded-previous-segment-prompts)) |
| Not visible in the contract | #48108 | Assisted decoding re-applies beginning-token suppression | real speech + encoder-decoder assistant catches it; neither slim fixture does ([follow-up](evidence.md#whisper-assisted-decoding-suppression)) |
| Parked | #48238 | T5 `lm_head` tying without `shared.weight` (madlad400): needs such a target | — |
| Out of path | #47682, #47684 (spelling), #45806 (decorator order), #47459, #47946 (GPT-2 modeling; SmolLM shares only the tokenizer), #49006 (MPS), #47921 (quantized CLIP), #47440 (multilingual `set_target_lang`) | — | — |

Whisper gained three sections drawing no random numbers; its 4 `cpu-arm64` baselines only added `max_length_tokens`, `assisted_tokens`, `longform_tokens`. Through the runner:

- Before #48000: fails on the assisted call.
- Before #48028: the initial `longform_tokens` probe fails only on `openai/whisper-tiny`; fixture rows never need left-padding, even after timestamp retraining. The later `continued_longform_tokens` probe controls this trigger and catches it on the fixture too ([follow-up](evidence.md#whisper-left-padded-previous-segment-prompts)).
- Before #47150/#46795 (July): already fails on the #47997 timestamp bug (fixed 2026-08-16), so `max_length` adds no detection in the window; it guards the next regression.

**Measured on history** (third backfill, [details](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)): the assisted call fails the 2 commits before #48000, as intended, and **caught a regression no triage listed**: #48981 broke Whisper's assisted `generate` for 41 commits until #49075 (2026-09-22 to 09-24), from its introducing commit. No new false alarm. A call added for one bug guarding the next one is the loop's point; it is still a single case.
