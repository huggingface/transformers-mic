# MIC * Model Integration Contract

**A proposed standard**: next to any Hub model, an executable check that it works in a given framework, run by that framework's CI.

## Summary

For each usage path (an architecture and a task, across the checkpoints that share it), publish a small integration contract on the Hub and run it from framework CI.

A contract is ordinary user code: load the model, call the public API, report the result. Transformers CI runs it on small fixtures on every PR (CPU, seconds) and on the real checkpoints after merge, nightly, and before release. Nothing is Transformers-specific: Transformers.js can use the same contracts.

**A slim pass is a fast regression signal, not a pass for the real model.** Run on full checkpoints, contracts can replace model-specific integration tests in `run-slow`; other slow tests stay ([CI integration](docs/slim-models-in-transformers-ci.md)).

![A proposed link lets agents discover a model's MIC on the Hub. Transformers checks slim fixtures on PRs and real checkpoints in run-slow or nightly jobs; Transformers.js has its own contract entry point and baseline, with CI triggers still to define.](docs/assets/mic-overview.svg)

**[Full example](example/README.md):** a runnable MIC for SmolLM text generation in Transformers and Transformers.js, plus six more contracts (CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2, MusicGen Melody) on the same runner.

The first goal is CI, not a Hub feature. The same contracts could later back compatibility status, tested snippets, Colabs, or agent usage.

## Problem

- **Compatibility is expensive to verify.** Integration tests need real checkpoints, large downloads, and GPUs, so each PR runs only a small part of the fleet. Regressions surface late: nightly, at release, or from users.
- **Copies drift.** Test code, documented usage, and Hub snippets diverge.
- **Agents guess.** In two [recorded runs](docs/agent-walkthrough/README.md) on Florence-2, both agents reconstructed a working path and neither could check output against a reviewed baseline. This shows friction, not a measured MIC gain ([assessment](docs/evidence.md#how-agents-use-the-hub-today)).

## The proposed standard

A MIC covers one usage path and every checkpoint on it. Each part links to the [full example](example/README.md).

1. **Repositories by owner.** Creators' model repos are unchanged. Hugging Face owns an [integration repo](https://huggingface.co/hf-internal-testing/SmolLM-integration) (the contract; [source](example/hub/SmolLM-integration)) and [fixture repos](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) (slim models). Each [framework repo](example/transformers/tests/contracts) owns its fleet lock and expected results.
2. **Fixed layouts:**

   ```text
   hf-internal-testing/SmolLM-integration      # Hub: what to run
     integration.json                          # manifest
     integrations/transformers/usage.md        # runnable markdown: documentation and test
     integrations/transformers.js/integration.js
   transformers/tests/contracts/               # framework repo: what to expect
     fleet.lock
     expectations/SmolLM/cpu-arm64/proto-SmolLM-slim.json  # one baseline per target, platform, and variant
     expectations/SmolLM/cpu-arm64/SmolLM-135M.json
     expectations/SmolLM/cpu-arm64/SmolLM-135M.eager.json
   ```

3. **A static manifest.** [`integration.json`](example/hub/SmolLM-integration/integration.json) declares checkpoints, each one's fixture, and per framework: entry point, variants, unsupported targets, resources, and a comparison rule per output.
4. **User code as the contract**: the public API with default settings, printing one JSON line. As [doc-builder runnable markdown](example/hub/SmolLM-integration/integrations/transformers/usage.md), it is both documentation and test.
5. **Checks beyond "it ran"**: generated outputs, logits within tolerance, and where training applies, finite loss and gradients and a parameter update.
6. **Reviewed baselines** ([`expectations/`](example/transformers/tests/contracts/expectations/SmolLM/cpu-arm64)) per target, variant, and platform class. A behavior-changing PR updates them; nothing updates them automatically.
7. **Shared [fixtures](example/hub/proto-SmolLM-slim).** Processor files copied byte for byte; only weights and config shrink. Checkpoints share a fixture only if those match.
8. **Everything pinned** in the framework's [fleet lock](example/transformers/tests/contracts/fleet.lock): contract, fixture, checkpoint, and dependency revisions.

[`run.py`](example/run.py) sketches the runner; the pilot builds the real one and the schemas. See the [technical design](docs/design.md).

## Why on the Hub, owned by Hugging Face

Contract upkeep shouldn't depend on creators' repos or on creators maintaining framework CI: Hugging Face owns contracts and fixtures, creators change nothing. Expected results change mostly with the framework, so each framework owns its own and the behavior-changing PR shows the diff. On the Hub, one versioned contract per usage path serves every framework and can later show per-checkpoint results.

## Pilot run

A CI-only pilot on 3–5 models, alongside existing tests, measuring regressions caught, cost at fleet scale, and maintenance load.

Early results on 1,468 Transformers `main` commits over five months ([shadow backfill](docs/evidence.md#shadow-backfill)):

- Frozen contracts caught the UMT5 regression ([#49135](https://github.com/huggingface/transformers/pull/49135)) at its introducing commit, flagged one benign change, and missed four regressions whose calls they didn't make.
- With those calls added ([how](docs/improving-contracts.md)), all five are caught, no new false alarm, ~13 s per commit on CPU. The new Whisper call found a live bug ([#49403](https://github.com/huggingface/transformers/issues/49403)).
- Scheduled runs on `main` caught a Whisper regression the night it merged, after its PR's CI had passed ([#49456](https://github.com/huggingface/transformers/issues/49456)).
- A call added for one bug caught another nobody targeted: Whisper's assisted `generate` broken by [#48981](https://github.com/huggingface/transformers/pull/48981) for 41 commits, from its introducing commit ([third round](docs/evidence.md#third-round-second-round-calls-and-bfloat16-smollm)).
- A proof job runs in Transformers' own PR CI runner and image, on a branch of `huggingface/transformers`: 8 fixtures in 17 s, selected from changed files, open bugs as narrowed known failures. A reverted UMT5 fix fails it with the backfill's exact deltas, and the repro command in the job summary gives the same failure on a Mac ([proof](docs/transformers-ci-proof.md)).
- No real regression has needed an overfit fixture yet: random-weight twins also caught MusicGen Melody's dropped conditioning ([#48679](https://github.com/huggingface/transformers/pull/48679)) from its introducing commit, and two silent Whisper generation bugs reached neither ([overfit or random](docs/evidence.md#overfit-or-random-weights-on-real-regressions)).

Agent reuse is measured separately ([recipe](docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic)). Remote-code models are out of initial scope.

## Documents

- [Full example](example/README.md): runnable SmolLM MIC, every part of the standard; the runner on seven contracts.
- [Evidence](docs/evidence.md): experiments, regression replay, shadow backfill, overfit vs random.
- [Problems and opened questions](docs/problems-and-opened-questions.md): what the pilot must settle.
- [Slim models in Transformers CI](docs/slim-models-in-transformers-ci.md): PR and `run-slow` flows.
- [Transformers CI proof](docs/transformers-ci-proof.md): the fleet in Transformers PR CI, with a one-page write-up for maintainers.
- [Technical design](docs/design.md): layout, CI, checks, baselines, runner.
- [Improving contracts](docs/improving-contracts.md): finding and closing coverage gaps.
- [End-to-end tests](docs/end-to-end-tests.md): every check in this repo, with commands.
- [Small fixtures](docs/fixtures.md): the fixture recipe and its experiments.
- [Agent walkthroughs](docs/agent-walkthrough/README.md): recorded Claude and Codex runs.
