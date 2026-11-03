# Pilot

The proposed adoption pilot is 3–5 usage paths run alongside existing tests in
Transformers CI, measuring regressions caught, fleet cost and maintenance.
Remaining work: [technical design](design.md#pilot); open questions:
[problems and open questions](problems-and-opened-questions.md). Agent reuse is
measured separately
([recipe](agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic)).
Remote-code models are out of initial scope.

As of 2026-11-02 the contracts job runs on Transformers PRs and reports failures
without blocking ([status](evidence.md#transformers-ci-proof)).

Whether a red contract should block a merge is measured daily by
[`mic-impact.yml`](../.github/workflows/mic-impact.yml): `summary.md` on the
`impact-results` branch replays blocking on every merged PR and tracks real
catches, false alarms, baseline review work, real-weights-only failures and
overfit versus random fixtures, against a proposed bar for a decision after a
month. Verdicts on PR-only failures go in that branch's `labels.jsonl`.

## Results so far

From replaying the first fleet's frozen contracts over 1,468 Transformers `main`
commits, five months ([shadow backfill](evidence.md#shadow-backfill)):

- They caught the UMT5 regression ([#49135](https://github.com/huggingface/transformers/pull/49135))
  at its introducing commit, flagged one benign change, and missed four
  regressions whose calls no contract made.
- With those calls added ([how](improving-contracts.md)), all five are
  caught with no new false alarm, at about 13 s per commit on CPU. The new Whisper
  call found a live bug ([#49403](https://github.com/huggingface/transformers/issues/49403)).
- A call added for one bug caught another nobody targeted: Whisper's assisted
  `generate`, broken by [#48981](https://github.com/huggingface/transformers/pull/48981)
  for 41 commits ([third round](evidence.md#third-round-second-round-calls-and-bfloat16-smollm)).
- Scheduled runs caught a Whisper regression the night it merged, after its PR's
  CI had passed ([#49456](https://github.com/huggingface/transformers/issues/49456)).
- In Transformers' PR CI runner and image, the job ran 8 fixtures in 17 s,
  selected from changed files. Reverting the UMT5 fix fails it with the backfill's
  exact deltas, and the summary's repro command gives the same
  failure on a Mac ([proof](transformers-ci-proof.md)).
- No real regression has needed overfit weights yet: random fixtures also caught
  MusicGen Melody's dropped conditioning (#48679) and Whisper's left-padding bug
  (#48028). Whisper assisted decoding (#48108) is missed by both
  ([replays](evidence.md#overfit-or-random-weights-on-real-regressions)).

