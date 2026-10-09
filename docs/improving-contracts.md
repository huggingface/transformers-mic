# Improving contracts continuously

A contract catches only what it calls. Over five months of `main`, the frozen contracts caught the one covered regression and missed four whose calls they didn't make ([evidence](evidence.md#shadow-backfill)); adding those calls found a fifth bug still on `main` ([#49403](https://github.com/huggingface/transformers/issues/49403)).

The loop: find misses, add the user-facing call, prove it on history, keep false alarms out. Tools: [shadow](../.github/workflows/shadow.yml), [backfill](../.github/workflows/shadow-backfill.yml), [nightly](../.github/workflows/nightly.yml), [`find_misses.py`](../example/find_misses.py), [`replay.py`](../example/replay.py), [`review_candidates.py`](../example/review_candidates.py).

## Where new checks come from

| Source | How to find it | Example |
| --- | --- | --- |
| A fix no contract noticed | `find_misses.py` on shadow/backfill/nightly results: fixes in a contract's code with no result change | #47997 (Whisper timestamps) touched `models/whisper`; no Whisper result moved |
| A bug or fix PR in a contract's family | Issues/PRs in its model dirs or shared processing, tokenization, generation, loading code | #46195 (CLIP `text_kwargs`), fixed in `processing_utils.py` |
| Documented usage the contract skips | Docs, card snippets, task guides vs. the contract's calls | Whisper `generate(..., return_timestamps=True)` was not called |
| A false alarm | A shadow/backfill failure classified benign | #46259 moved Wav2Vec2's training loss via a load-time RNG shift |

`find_misses.py` tiers:

- **Model:** touch the contract's own model dir. Review all (17 over five months for six contracts, incl. #47997, #47440).
- **Shared** (`--shared`): touch shared code, match every contract. Review by subject; three of four known misses (#46195, #47628, #47793) are here.

A candidate is a question, not a verdict: a typo or speculative-decoding fix may be off-path on purpose.

## Adding a check

1. **Reproduce on the real checkpoint first**, through the public call, on the bad and fixed builds (fix commit's parent and fix, or the introducing commit, as in the [replay](evidence.md#historical-regression-replay)). No checkpoint failure → stop; the fixture won't show it either.
2. **Write it as user code:** documented API, default settings, short, deterministic, in its own `usage.md` section (renders as docs).
3. **Don't move existing outputs.** Place it before training where it draws no random numbers (tokenization, decoding, config reads, greedy generation), so existing outputs stay byte-identical. If it needs randomness, put it after the outputs it could affect and re-review them.
4. **Declare its output** in the manifest: `exact` for tokens, text, labels, keys; `allclose` only for floats.
5. **Prove it on the bug** with [`replay.py`](../example/replay.py): fixture fails on bad, passes on fixed. If only the checkpoint differs, keep it as full-model coverage (nightly, `run-slow`), not PR coverage.
6. **Record baselines on every platform** (each rerun in a fresh process):
   - `cpu-arm64`: `run.py --record`
   - `cpu-x86_64`: [`cpu-baselines.yml`](../.github/workflows/cpu-baselines.yml) `machine=gpu-host`
   - `cuda`: [`nightly.yml`](../.github/workflows/nightly.yml) `mode=record`
7. **Review.** [`review_candidates.py`](../example/review_candidates.py) classifies each candidate: **unchanged** (new outputs only), **changed** (needs a reason), **new**. `--approve` renames only unchanged and new; changed is a person's call.
8. **Measure on history.** Re-run the backfill; record caught regressions, new false alarms, and per-commit cost separately in [evidence](evidence.md), frozen-suite detections apart from post-hoc ones.
9. **Fails on today's `main`?** It found a live bug: reproduce, report upstream, and keep the check with a `local_patches` workaround in the fleet lock (recorded in every result), not `known_failures`, which hides the contract's other checks. Drop the patch and re-record once fixed.

## Keeping false alarms down

For each benign failure, log one decision with it:

- **Keep and update the baseline** if rare and the check guards something real. Wav2Vec2's training loss depends on load-time randomness because the fixture, like the checkpoint, omits `masked_spec_embed`; that's how it catches [#49385](https://github.com/huggingface/transformers/issues/49385). One update in five months.
- **Make it robust:** seed closer to use, compare a discrete outcome, or move away from unrelated randomness.
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

- **No existing output moved:** the 18 affected `cpu-arm64` baselines only gained outputs; `review_candidates.py` confirmed all 98 re-recorded baselines.
- **Live Whisper bug:** `WhisperTokenizer.decode` batches only lists of lists, so the tensor `generate` returns merges rows or raises (#47997 fixed lists only). Reported as [#49403](https://github.com/huggingface/transformers/issues/49403); worked around with `whisper-decode-tensor-batch` in `local_patches`.
- **`find_misses.py` lists all four misses** from the first backfill's results.
- **The backfill re-run caught all four**, each until its fix, with no new false alarm or added cost ([details](evidence.md#after-adding-the-missed-calls)). Post-hoc: it shows the loop closes gaps, not the unknown-bug rate.

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

Whisper gained three pre-training sections drawing no random numbers (checked against RNG state); its 4 `cpu-arm64` baselines only added `max_length_tokens`, `assisted_tokens`, `longform_tokens`. Through the runner:

- Before #48000: fails on the assisted call.
- Before #48028: fails `longform_tokens` on `openai/whisper-tiny`; the fixture passes (its output ignores decoder positions).
- Before #47150/#46795 (July): already fails earlier, on the #47997 timestamp bug (fixed 2026-08-16). So `max_length` adds no detection in the window; it guards the next regression.
