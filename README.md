# MICs * Model integration contracts

**A proposed standard** for publishing, next to any model on the Hub, an executable check that the model works in a given framework and can be run by that framework's CI.

## Summary

We should publish a small, executable integration contract on the Hub for each usage path we want to support (an architecture and a task, across the checkpoints that share it), and run it from framework CI.

A contract is ordinary user code: load the model, call the public API users call, report the result. In the proposed Transformers flow, it runs against small fixtures on every PR and against the real checkpoints later. The contract and fixtures live in Hugging Face-owned companion repositories on the Hub, not in the model repositories; each framework keeps its expected results next to its own tests:

- `HuggingFaceTB/SmolLM-135M`, `-360M`, and `-1.7B`: the real models, owned by the creator, unchanged;
- [`hf-internal-testing/SmolLM-integration`](example/hub/SmolLM-integration): one contract for SmolLM text generation covering all three checkpoints, owned by Hugging Face;
- [`hf-internal-testing/proto-SmolLM-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-slim) and [`proto-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/proto-SmolLM-1.7B-slim) (published; sources in [`example/hub/`](example/hub)): small fixtures, one per distinct tokenizer and config, owned by Hugging Face;
- [`transformers/tests/contracts/`](example/transformers/tests/contracts): the fleet lock and Transformers' expected results, owned by Transformers maintainers.

Transformers CI would fetch the contract and fixtures at pinned revisions and run them: against each small fixture on every PR, where it runs on CPU in seconds, and against the real checkpoints after merge, nightly, and before release. Transformers.js has its own contract entry point and baselines; its CI triggers remain to be defined.

**A passing slim check is only a fast regression signal. It does not mean that the real model passes.** Contracts can replace model-specific integration tests in `run-slow` by running against full checkpoints; the other slow framework tests remain. [Slim models in Transformers development and CI](docs/slim-models-in-transformers-ci.md) explains the impact on PRs, contract maintenance, and the fast and slow validation paths.

![A proposed link lets agents discover a model's MIC on the Hub. Transformers checks slim fixtures on PRs and real checkpoints in run-slow or nightly jobs; Transformers.js has its own contract entry point and baseline, with CI triggers still to define.](docs/assets/mic-overview.svg)

**[Full example](example/README.md):** a complete, runnable MIC for SmolLM text generation in Transformers and Transformers.js: manifest, contract as doc-builder runnable markdown, shared fixtures, fleet locks, reviewed baselines in each framework, and a runner sketch.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## Problem

Model compatibility is expensive to verify. Integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks, so only a small part of the model fleet runs on each PR. We find regressions late: after merge, in nightly jobs, during a release, or from users.

The code that validates a model is also separate from the code we show users and from Hub snippets. Each copy drifts on its own.

Agents feel this. In two [recorded runs](docs/agent-walkthrough/README.md) on Florence-2, Claude fell back to a community checkpoint and Codex rebuilt an old environment with a workaround. Both had to reconstruct a working path; neither could check its result against a reviewed baseline. These runs show the friction, not a measured gain from MICs ([assessment](docs/evidence.md#how-agents-use-the-hub-today)). Agents would also need a discoverable link from a model to its contract.

## The proposed standard

A MIC is defined per usage path and covers every checkpoint that shares it. A conforming MIC has the parts below; each links to the [full example](example/README.md).

1. **Repositories by owner.** Creators' model repositories are unchanged. A Hugging Face-owned [integration repository](example/hub/SmolLM-integration) holds the contract for one usage path and lists the checkpoints it covers. Hugging Face-owned [fixture repositories](example/hub/proto-SmolLM-slim) hold only slim models: weights, the shrunk config, and copied processor files. Each [framework repository](example/transformers/tests/contracts) holds its own fleet lock and expected results.
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

3. **A manifest readable without running code.** [`integration.json`](example/hub/SmolLM-integration/integration.json), validated by a schema (written during the pilot), declares the checkpoints and the fixture that covers each, and per framework the entry point, artifacts, supported variants, unsupported targets with a reason, resource requirements, and the comparison rule for each output.
4. **Ordinary user code as the contract.** The contract ([runnable markdown](example/hub/SmolLM-integration/integrations/transformers/usage.md), [JavaScript](example/hub/SmolLM-integration/integrations/transformers.js/integration.js)) calls the public API users call (pipeline, processor, generate) in the configuration they get by default (attention implementation, dtype), and prints one JSON result line on stdout. Where a framework's docs support [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md), the contract is written as one, so the same code is rendered as documentation and run as a test.
5. **Checks beyond "it ran".** Generated tokens or task outputs, selected logits within calibrated tolerances, and, where training is supported, a finite loss, gradients on the expected modules, and a parameter update. Missing fields, non-finite values, and mismatches fail.
6. **Reviewed baselines in each framework repository** ([`expectations/`](example/transformers/tests/contracts/expectations/SmolLM/cpu-arm64)): one per fixture and one per checkpoint, for each variant and platform class (CPU, GPU) where they differ. Each records the revision, environment, device, and dtype it came from. A framework PR that changes behavior updates its baselines in the same PR. Outputs never update a baseline automatically ([`run.py --record`](example/run.py) only writes a candidate), and a build under test never defines its own expected answer.
7. **[Fixtures](example/hub/proto-SmolLM-slim) shared by checkpoints that preprocess the same way.** Processor, tokenizer, and preprocessing files are copied byte for byte; only the weights and model config shrink. Checkpoints share a fixture when those files and every config field that shrinking keeps are identical, and the fixture build fails otherwise.
8. **Everything pinned.** Each framework runs contracts from its [fleet lock](example/transformers/tests/contracts/fleet.lock), listing the contract, fixture, checkpoint, converted artifact, and dependency revisions; its baselines are versioned with it.

The example's [`run.py`](example/run.py) sketches the shared runner. The manifest and result schemas and the real runner are what the pilot builds. Details are in the [technical design](docs/design.md).

Matching a reviewed baseline establishes agreement on fixed inputs under the recorded runtime. It does not establish semantic correctness on arbitrary new inputs, and a passing slim fixture is not a passing result for the real checkpoint.

## Why on the Hub, owned by Hugging Face

Contracts need maintenance: inputs change, fixtures get regenerated, checkpoints get added. That should not depend on write access to a creator's repository or on the creator maintaining framework CI. Hugging Face owns the contracts and fixtures; creators keep owning their models and change nothing. Expected results change mostly when a framework changes, so each framework owns its own, and the PR that changes behavior also shows the expectation diff.

On the Hub, a contract is one versioned place per usage path that every framework can run, and the Hub can later show the per-checkpoint results that framework CI publishes.

## Pilot run

Run a CI-only pilot on 3–5 models, keeping existing tests running in parallel. Measure three things: real regressions caught, cost at fleet scale, and maintenance load. If the pilot confirms it, expand the fleet, then build the Hub-facing pieces.

Early results ([shadow backfill](docs/evidence.md#shadow-backfill)), on all 1,468 Transformers `main` commits of the last five months:

- The frozen fixture contracts caught the UMT5 regression ([#49135](https://github.com/huggingface/transformers/pull/49135)) on its introducing commit, flagged one benign change, and missed four regressions whose calls no contract made.
- With those four calls added ([how](docs/improving-contracts.md)), all five are caught, with no new false alarm, at about 13 s per commit on a CPU runner. The new Whisper call also found a bug still on `main` ([#49403](https://github.com/huggingface/transformers/issues/49403)).

Agent reuse is measured separately, with the [controlled walkthrough recipe](docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic). Remote-code models such as Microsoft's Florence are out of the initial scope.

## Documents

- [Full example](example/README.md): a runnable MIC for SmolLM text generation (three checkpoints, two fixtures) in Transformers and Transformers.js, with every part of the standard.
- [Evidence](docs/evidence.md): what the local experiments and the historical regression replay show.
- [Problems and opened questions](docs/problems-and-opened-questions.md): limits of the fast checks, fixture sensitivity, baseline reliability, maintenance costs, and validation gaps to resolve during the pilot.
- [Slim models in Transformers development and CI](docs/slim-models-in-transformers-ci.md): developer and maintainer flows, fast slim checks on PRs, full-model checks in `run-slow`, and the CI integration decisions still to make.
- [Technical design](docs/design.md): contract layout, CI triggers, where code lives, checks, baselines, runner, pilot plan, open questions.
- [Improving contracts](docs/improving-contracts.md): the maintenance loop: find what contracts miss, add the user-facing call, prove it on history, keep false alarms down.
- [End-to-end tests](docs/end-to-end-tests.md): every check this repository runs (contracts, shadow and GPU jobs, regression replays, synthetic faults, stability, cost, sweep, WebGPU), with commands.
- [Small fixtures](docs/fixtures.md): the fixture recipe (real config shrunk by layer pattern, copied processor files, overfit and frozen) and the experiments behind it.
- [Agent walkthroughs](docs/agent-walkthrough/README.md): the shared recipe and prompts, with separate Claude Opus and Codex results recording local Hub inference, tool calls, outputs, and failures.
