# Improving contracts continuously

A contract catches only what its calls exercise. Over five months of `main`, the frozen contracts caught the one covered regression and missed four whose calls they did not make ([evidence](evidence.md#shadow-backfill)). Adding those calls then found a fifth bug, still on `main` ([#49403](https://github.com/huggingface/transformers/issues/49403)).

So contracts need a maintenance loop:

1. Find what they miss.
2. Add the user-facing call that would have caught it.
3. Prove it on history.
4. Keep false alarms out.

Tools: the [shadow](../.github/workflows/shadow.yml), [backfill](../.github/workflows/shadow-backfill.yml) and [nightly](../.github/workflows/nightly.yml) jobs, [`find_misses.py`](../example/find_misses.py), [`replay.py`](../example/replay.py), and [`review_candidates.py`](../example/review_candidates.py).

## Where new checks come from

| Source | How to find it | Example |
| --- | --- | --- |
| A fix nobody's contract noticed | `find_misses.py` on shadow, backfill, or nightly results: fix commits in a contract's code where none of its results changed | #47997 (Whisper timestamps) touched `models/whisper`; no Whisper result moved |
| A bug report or fix PR in a contract's family | Issues and PRs touching the contract's model directories or shared processing, tokenization, generation, and loading code | #46195 (CLIP `text_kwargs`), fixed in `processing_utils.py` |
| A documented usage the contract skips | Compare the model's docs, card snippets, and task guides with the contract's calls | Whisper's documented `generate(..., return_timestamps=True)` was not called |
| A false alarm | A shadow or backfill failure classified as benign | #46259 moved the Wav2Vec2 training loss via a load-time RNG shift |

`find_misses.py` reports two tiers:

- **Model** candidates touch the contract's own model directory. Review all of them (17 over five months for six contracts, including #47997 and #47440).
- **Shared** candidates (`--shared`) touch only shared code and match every contract. Review them by subject; three of the four known misses (#46195, #47628, #47793) are here.

A candidate is a question, not a verdict. A typo fix or a speculative-decoding fix may be off the contract's usage path on purpose.

## Adding a check

1. **Reproduce on the real checkpoint first.** Take the bad and fixed builds (fix commit and its parent, or the introducing commit) and confirm the failure through the public call, as in the [replay](evidence.md#historical-regression-replay). If the checkpoint does not show it, stop: the fixture won't either.
2. **Write the call as user code.** The documented public API with default settings, not a probe of internals. Short and deterministic. Give it its own section in `usage.md`, so it renders as documentation too.
3. **Do not move existing outputs.** Place the section where it draws no random numbers before the training step (tokenization, decoding, config reads, greedy generation). Existing outputs then stay byte-identical, which review can prove. If the call needs randomness, put it after the outputs it could affect and expect to re-review them.
4. **Declare what it returns.** Add it to the manifest's `outputs`: `exact` for tokens, text, labels, keys; `allclose` only for floats.
5. **Prove it on the bug.** Run [`replay.py`](../example/replay.py) on the bad and fixed builds: the fixture must fail on bad, pass on fixed. If only the real checkpoint differs, keep the check and note it as covered by full-model paths (nightly, `run-slow`), not the PR check.
6. **Record baselines on every platform**, each rerun in a fresh process before writing:
   - `cpu-arm64`: `run.py --record`
   - `cpu-x86_64`: [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) `machine=gpu-host`
   - `cuda`: [`nightly.yml`](../.github/workflows/nightly.yml) `mode=record`
7. **Review and approve.** [`review_candidates.py`](../example/review_candidates.py) classifies each candidate against the approved baseline, using the contract's rules:
   - **unchanged:** existing outputs match; new ones listed as added.
   - **changed:** an existing output moved; needs a reason.
   - **new:** no baseline yet.

   `--approve` renames only unchanged and new candidates. A changed one is a person's decision.
8. **Measure it on history.** Re-run the backfill over the same window. Count separately: regressions now caught, new false alarms, added cost per commit. Record them in the [evidence](evidence.md), keeping frozen-suite detections apart from post-hoc ones: a check written for a known bug proves little about the next.
9. **If it fails on today's `main`, it found a live bug.** Reproduce on the real checkpoint and report upstream. Keep the check running with a `local_patches` workaround in the fleet lock (recorded in every result). Don't use `known_failures`: it would hide the contract's other checks. Remove the patch and re-record when the fix lands.

## Keeping false alarms down

Each benign failure costs review time on every PR that trips it. For each one, decide:

- **Keep it, update the baseline** when the change is rare and the check guards something real. The Wav2Vec2 training loss depends on load-time randomness because the fixture omits `masked_spec_embed`, like the real checkpoint; that is how it catches [#49385](https://github.com/huggingface/transformers/issues/49385). One update in five months is cheap.
- **Make it robust:** seed closer to the use, compare a discrete outcome instead of a number, or move the call away from unrelated randomness.
- **Drop it** if it keeps failing on benign changes and has never caught anything.

Log each decision with the failure that prompted it, so each check's false-alarm rate and cost stay visible.

## Cadence

| When | What |
| --- | --- |
| Every shadow or nightly failure | Classify: regression, intended change (update the baseline with the reason), or benign (see above). |
| Weekly | Run `find_misses.py` on the week's shadow results; triage model-tier candidates; open a contract change for each one worth covering. |
| After each contract change | Re-run the backfill over the reference window; update the evidence with detections, false alarms, and cost. |
| After an upstream fix of a bug a contract found | Drop the `local_patches` or `known_failures` entry; re-record the affected baselines. |

## What happened the first time

The five-month backfill flagged four misses. Their calls were added (CLIP per-modality `text_kwargs` and `get_image_processor_config`; Whisper timestamp generation and `batch_decode(..., decode_with_timestamps=True)` and `get_feature_extractor_config`; the same config read for Wav2Vec2; T5 `decode` on a punctuated batch).

- **No existing output moved.** All 18 affected `cpu-arm64` baselines only gained outputs.
- **The Whisper call found a live bug.** `WhisperTokenizer.decode` batches only a list of lists, so decoding the tensor `generate` returns merges rows or raises; #47997 fixed lists only. Reported as [#49403](https://github.com/huggingface/transformers/issues/49403); the contract keeps the call with the `whisper-decode-tensor-batch` workaround in `local_patches`.
- **The re-run backfill caught all four,** each failing until its fix, with no new false alarm and no added cost ([details](evidence.md#after-adding-the-missed-calls)).

This is post-hoc: it shows the loop closes gaps, not how many unknown bugs it will catch. Two tools came out of it: `find_misses.py`, which lists all four misses from the first backfill, and `review_candidates.py`, which confirmed that all 98 re-recorded baselines kept their outputs.

## The second round: triaging the model-tier misses

`find_misses.py --ref origin/main` on the post-round-one backfill lists 16 model-tier candidates (2026-10-08). Each was read; those on a contract's usage path were reproduced on the real checkpoint before and after the fix.

| Verdict | Fix | What it fixed | Seen by |
| --- | --- | --- | --- |
| Covered | #47150 (2026-07-17), #46795 (2026-07-23) | Whisper `generate(max_length=N)`: the forced prefix was not added back to `max_length`, then added one short; `max_length=1` raised `ValueError` | fixture and checkpoint |
| Covered | #48000 (2026-08-17) | Speculative decoding with a `WhisperForCausalLM` assistant raised `UnboundLocalError` | fixture and checkpoint |
| Covered | #48028 (2026-08-19) | Long-form batch with `condition_on_prev_tokens=True`: wrong decoder positions for left-padded prompts | checkpoint only (nightly) |
| Not visible | #48108 | Speculative decoding reset the cleared suppress tokens: the contract's audio never produces them | — |
| Parked | #48238 | T5 `lm_head` tying for checkpoints with a separate `lm_head` and no `shared.weight` (madlad400): needs such a target | — |
| Out of path | #47682, #47684 (spelling), #45806 (decorator order), #47459, #47946 (GPT-2 modeling: SmolLM shares only its tokenizer), #49006 (MPS autocast; no MPS platform), #47921 (CLIP with quantized children), #47440 (Wav2Vec2 `set_target_lang` on multilingual vocabularies) | — | — |

The Whisper contract gained three sections, all before the training step and drawing no random numbers (checked against the RNG state). The 4 `cpu-arm64` baselines kept their outputs and only added `max_length_tokens`, `assisted_tokens`, and `longform_tokens`.

Through the runner:

- The build before #48000 fails on the assisted call.
- The build before #48028 fails `longform_tokens` on `openai/whisper-tiny` but passes on the fixture, whose output does not depend on decoder positions.
- The July builds before #47150 and #46795 already fail earlier, on the timestamp bug fixed by #47997 (2026-08-16). So the `max_length` check adds no detection in the backfill window; it guards against the next regression of that call.
