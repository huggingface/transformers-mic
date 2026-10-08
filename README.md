# MIC * Model Integration Contract

**A proposed standard** for publishing, next to any model on the Hub, an executable check that the model works in a given framework, run by that framework's CI.

## Summary

For each usage path we support (an architecture and a task, across the checkpoints that share it), publish a small integration contract on the Hub and run it from framework CI.

A contract is ordinary user code: load the model, call the public API, report the result. Transformers CI runs it against small fixtures on every PR (CPU, seconds) and against the real checkpoints after merge, nightly, and before release. Transformers.js has its own entry point and baselines; its CI triggers are still to define.

**A slim pass is a fast regression signal, not a pass for the real model.** Contracts can replace model-specific integration tests in `run-slow` by running on full checkpoints; other slow tests stay ([CI integration](docs/slim-models-in-transformers-ci.md)).

![A proposed link lets agents discover a model's MIC on the Hub. Transformers checks slim fixtures on PRs and real checkpoints in run-slow or nightly jobs; Transformers.js has its own contract entry point and baseline, with CI triggers still to define.](docs/assets/mic-overview.svg)

**[Full example](example/README.md):** a runnable MIC for SmolLM text generation in Transformers and Transformers.js.

The first goal is CI, not a Hub feature. The same contracts could later back compatibility status, tested snippets, Colabs, or agent usage.

## Problem

- **Compatibility is expensive to verify.** Integration tests need real checkpoints, large downloads, and GPUs, so only a small part of the fleet runs on each PR. Regressions surface late: nightly, at release, or from users.
- **Copies drift.** Test code, documented usage, and Hub snippets are separate and diverge.
- **Agents guess.** In two [recorded runs](docs/agent-walkthrough/README.md) on Florence-2, both agents had to reconstruct a working path, and neither could check its output against a reviewed baseline. This shows friction, not a measured gain from MICs ([assessment](docs/evidence.md#how-agents-use-the-hub-today)).

## The proposed standard

A MIC covers one usage path and every checkpoint that shares it. Each part links to the [full example](example/README.md).

1. **Repositories by owner.** Creators' model repos are unchanged. Hugging Face owns an [integration repo](example/hub/SmolLM-integration) (the contract) and [fixture repos](example/hub/proto-SmolLM-slim) (slim models). Each [framework repo](example/transformers/tests/contracts) owns its fleet lock and expected results.
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

3. **A static manifest.** [`integration.json`](example/hub/SmolLM-integration/integration.json) declares checkpoints, the fixture for each, and per framework: entry point, variants, unsupported targets, resources, and a comparison rule per output.
4. **User code as the contract.** It calls the public API with default settings and prints one JSON line. Written as [doc-builder runnable markdown](example/hub/SmolLM-integration/integrations/transformers/usage.md), it is both documentation and test.
5. **Checks beyond "it ran".** Generated outputs, logits within tolerance, and where training applies, finite loss and gradients and a parameter update.
6. **Reviewed baselines** ([`expectations/`](example/transformers/tests/contracts/expectations/SmolLM/cpu-arm64)) per target, variant, and platform class. A PR that changes behavior updates them; nothing updates them automatically.
7. **Shared [fixtures](example/hub/proto-SmolLM-slim).** Processor files are copied byte for byte; only weights and config shrink. Checkpoints share a fixture only if those match.
8. **Everything pinned** in the framework's [fleet lock](example/transformers/tests/contracts/fleet.lock): contract, fixture, checkpoint, and dependency revisions.

[`run.py`](example/run.py) sketches the runner; the pilot builds the real one and the schemas. See the [technical design](docs/design.md).

## Why on the Hub, owned by Hugging Face

Contracts need upkeep, which shouldn't depend on creators' repos or creators maintaining framework CI. Hugging Face owns contracts and fixtures; creators change nothing. Expected results change mostly with the framework, so each framework owns its own, and the behavior-changing PR shows the diff. On the Hub, one versioned contract per usage path serves every framework, and can later display per-checkpoint results.

## Pilot run

A CI-only pilot on 3–5 models, existing tests kept running. It measures regressions caught, cost at fleet scale, and maintenance load.

Early results on 1,468 Transformers `main` commits over five months ([shadow backfill](docs/evidence.md#shadow-backfill)):

- Frozen contracts caught the UMT5 regression ([#49135](https://github.com/huggingface/transformers/pull/49135)) at its introducing commit, flagged one benign change, and missed four regressions whose calls they didn't make.
- With those calls added ([how](docs/improving-contracts.md)), all five are caught, no new false alarm, ~13 s per commit on CPU. The new Whisper call found a live bug ([#49403](https://github.com/huggingface/transformers/issues/49403)).

Agent reuse is measured separately ([recipe](docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic)). Remote-code models are out of initial scope.

## Documents

- [Full example](example/README.md): runnable SmolLM MIC, every part of the standard.
- [Evidence](docs/evidence.md): experiments, regression replay, shadow backfill.
- [Problems and opened questions](docs/problems-and-opened-questions.md): what the pilot must settle.
- [Slim models in Transformers CI](docs/slim-models-in-transformers-ci.md): PR and `run-slow` flows.
- [Technical design](docs/design.md): layout, CI, checks, baselines, runner.
- [Improving contracts](docs/improving-contracts.md): finding and closing coverage gaps.
- [End-to-end tests](docs/end-to-end-tests.md): every check in this repo, with commands.
- [Small fixtures](docs/fixtures.md): the fixture recipe and its experiments.
- [Agent walkthroughs](docs/agent-walkthrough/README.md): recorded Claude and Codex runs.
