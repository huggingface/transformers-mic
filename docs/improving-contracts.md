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
7. **Review.** [`review_candidates.py`](../example/review_candidates.py) classifies each candidate **unchanged** (new outputs only), **changed** (needs a reason), or **new**. `--approve` renames only unchanged and new; changed is a person's call.
8. **Measure on history.** Re-run the backfill; record catches, new false alarms, and per-commit cost in [evidence](evidence.md), post-hoc detections apart.
9. **Fails on today's `main`?** Live bug: report upstream and keep the check with a `local_patches` workaround (recorded per result), not `known_failures`, which hides the contract's other checks. Drop the patch and re-record once fixed.

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
| Covered | #48028 (2026-08-19) | Long-form `condition_on_prev_tokens=True`: wrong positions for left-padded prompts | checkpoint only (nightly) |
| Not visible | #48108 | Assisted decoding reset suppress tokens the contract's audio never produces | — |
| Parked | #48238 | T5 `lm_head` tying without `shared.weight` (madlad400): needs such a target | — |
| Out of path | #47682, #47684 (spelling), #45806 (decorator order), #47459, #47946 (GPT-2 modeling; SmolLM shares only the tokenizer), #49006 (MPS), #47921 (quantized CLIP), #47440 (multilingual `set_target_lang`) | — | — |

Whisper gained three sections drawing no random numbers; its 4 `cpu-arm64` baselines only added `max_length_tokens`, `assisted_tokens`, `longform_tokens`. Through the runner:

- Before #48000: fails on the assisted call.
- Before #48028: fails `longform_tokens` on `openai/whisper-tiny`; the fixture passes (its output ignores decoder positions).
- Before #47150/#46795 (July): already fails on the #47997 timestamp bug (fixed 2026-08-16), so `max_length` adds no detection in the window; it guards the next regression.

**Measured on history** (third backfill, [details](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)): the assisted call fails the 2 commits before #48000, as intended, and **caught a regression no triage listed**: #48981 broke Whisper's assisted `generate` for 41 commits until #49075 (2026-09-22 to 09-24), from its introducing commit. No new false alarm. A call added for one bug guarding the next one is the loop's point; it is still a single case.
