# Improving contracts continuously

A contract catches only what its calls exercise. Over five months of Transformers `main`, the frozen contracts caught the one covered regression on its introducing commit and missed four others in the same model families, because no contract made the affected call ([evidence](evidence.md#shadow-backfill)). Adding those four calls found a fifth bug, still on `main`. So contracts need a maintenance loop: find what they miss, add the user-facing call that would have caught it, prove it on history, and keep false alarms from creeping in.

This page is that loop. It uses the tools already in this repository: the [shadow](../.github/workflows/shadow.yml), [backfill](../.github/workflows/shadow-backfill.yml) and [nightly](../.github/workflows/nightly.yml) jobs, [`find_misses.py`](../example/find_misses.py), [`replay.py`](../example/replay.py), and [`review_candidates.py`](../example/review_candidates.py).

## Where new checks come from

| Source | How to find it | Example |
| --- | --- | --- |
| A fix nobody's contract noticed | [`find_misses.py`](../example/find_misses.py) on shadow, backfill, or nightly results: fix commits in a contract's code where none of its results changed | #47997 (Whisper timestamps): its fix touched `models/whisper`, and no Whisper result moved |
| A bug report or fix PR in a contract's family | Issues and PRs touching the contract's model directories or the shared processing, tokenization, generation, and loading code | #46195 (CLIP `text_kwargs`), fixed in `processing_utils.py` |
| A documented usage the contract skips | The model's docs page, model card snippets, and task guides compared with the contract's calls | `generate(..., return_timestamps=True)` is documented Whisper usage; the contract did not call it |
| A false alarm | A shadow or backfill failure classified as benign | #46259 moved the Wav2Vec2 fixture's training loss through a load-time RNG shift |

`find_misses.py` reports two tiers. **Model** candidates touch the contract's own model directory (17 commits over five months for the six contracts, including #47997 and #47440); review all of them. **Shared** candidates (`--shared`) only touch shared code and match every contract, so they are many; review them by subject, as three of the four known misses (#46195, #47628, #47793) are in this tier. Each candidate is a question, not a verdict: a typo fix or a speculative-decoding fix may be outside the contract's usage path on purpose.

## Adding a check

1. **Reproduce on the real checkpoint first.** Find the bad and fixed builds (the fix commit and its parent, or the introducing commit) and confirm the failure through the public call, as in the [replay](evidence.md#historical-regression-replay). If the real checkpoint does not show it, stop: the fixture will not either.
2. **Write the call as user code.** Use the documented public API, the call users or downstream libraries make, with the defaults they get; not a probe of internals. Keep it short and deterministic. Put it in its own section of the contract (`usage.md`), so it renders as documentation too.
3. **Do not move existing outputs.** Place the new section where it draws no random numbers before the training step (tokenization, decoding, config reads, greedy generation). Then every existing output stays byte-identical, and the review below can prove it. If the new call must consume randomness, put it after the outputs it could affect and expect to re-review those.
4. **Declare what it returns.** Add the output to the manifest's `outputs`, `exact` for tokens, text, labels, and keys, `allclose` only for floating-point values.
5. **Prove it on the bug.** Run the contract on the bad and fixed builds with [`replay.py`](../example/replay.py): the fixture must fail on the bad build and pass on the fixed one. If only the real checkpoint shows the difference, keep the check and note it as covered by the full-model paths (nightly, `run-slow`), not by the PR check.
6. **Record baselines on every platform.** `run.py --record` on `cpu-arm64`; [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) `machine=gpu-host` for `cpu-x86_64`; [`nightly.yml`](../.github/workflows/nightly.yml) `mode=record` for `cuda`. Each candidate is rerun in a fresh process before it is written.
7. **Review and approve.** [`review_candidates.py`](../example/review_candidates.py) compares each candidate with the approved baseline under the contract's own rules: **unchanged** (all existing outputs match, new ones listed as added), **changed** (an existing output moved: needs a reason), or **new**. `--approve` renames only unchanged and new candidates; a changed one is a person's decision.
8. **Measure it on history.** Re-run the [backfill](../.github/workflows/shadow-backfill.yml) over the same window. Count, separately: regressions now caught, new false alarms, and the added cost per commit. Record them in the [evidence](evidence.md), distinguishing detections by the frozen suite from detections after adding the call: a check written for a known bug proves little about the next one.
9. **If the new call fails on today's `main`,** it found a live bug. Reproduce it on the real checkpoint, report it upstream, and keep the check running with a `local_patches` workaround in the fleet lock (recorded in every result) rather than a `known_failures` entry, which would hide the contract's other checks. Remove the patch and re-record when the fix lands.

## Keeping false alarms down

A check that fails on benign changes costs review time on every PR that trips it. For each failure that turns out benign, decide one of:

- **Keep it, update the baseline.** When the change is rare and the check guards something real. The Wav2Vec2 fixture's training loss depends on load-time randomness because the fixture omits `masked_spec_embed`, like the real checkpoint, which is how it catches [#49385](https://github.com/huggingface/transformers/issues/49385). One baseline update in five months is cheap.
- **Make it robust.** Seed closer to the use, compare a discrete outcome instead of a number, or move the call so unrelated randomness does not reach it.
- **Drop it** if it keeps failing on benign changes and has never caught anything.

Track each decision with the failure that prompted it, so the false-alarm rate and the cost of each check stay visible.

## Cadence

| When | What |
| --- | --- |
| Every shadow or nightly failure | Classify it: regression, intended change (update the baseline with the reason), or benign (see above). |
| Weekly | Run `find_misses.py` on the week's shadow results; triage the model-tier candidates; open a contract change for each one worth covering. |
| After each contract change | Re-run the backfill over the reference window; update the evidence with the added detections, false alarms, and cost. |
| After each upstream fix of a bug a contract found | Drop the `local_patches` or `known_failures` entry and re-record the affected baselines. |

## What happened the first time

The five-month backfill flagged four misses. Adding their calls (CLIP per-modality `text_kwargs` and `get_image_processor_config`; Whisper `generate(return_timestamps=True)` with `batch_decode(..., decode_with_timestamps=True)` and `get_feature_extractor_config`; the same config read for Wav2Vec2; T5 `decode` on a punctuated batch) left all 18 affected `cpu-arm64` baselines unchanged and only added outputs. The Whisper call failed on today's `main`: `WhisperTokenizer.decode` batches only a list of lists, so decoding the tensor `generate` returns merges its rows or raises; #47997 had fixed lists only. The contract keeps the call with the `whisper-decode-tensor-batch` workaround in `local_patches`. The re-run backfill measures what the four calls add over the same five months.
