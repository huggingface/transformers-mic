# MICs * Model integration contracts

**A proposed standard** for publishing, next to any model on the Hub, an executable check that the model works in a given framework, run by that framework's CI on every change.

## Summary

We should publish a small, executable integration contract on the Hub for each usage path we want to support (an architecture and a task, across the checkpoints that share it), and run it from framework CI.

A contract is ordinary user code: load the model, call the public API users call, report the result. It runs against small fixtures on every PR and against the real checkpoints later. The contract and fixtures live in Hugging Face-owned companion repositories on the Hub, not in the model repositories; each framework keeps its expected results next to its own tests:

- `HuggingFaceTB/SmolLM-135M`, `-360M`, and `-1.7B`: the real models, owned by the creator, unchanged;
- [`hf-internal-testing/SmolLM-integration`](example/hub/SmolLM-integration): one contract for SmolLM text generation covering all three checkpoints, owned by Hugging Face;
- [`hf-internal-testing/SmolLM-slim`](example/hub/SmolLM-slim) and [`SmolLM-1.7B-slim`](example/hub/SmolLM-1.7B-slim): small fixtures, one per distinct tokenizer and config, owned by Hugging Face;
- [`transformers/tests/contracts/`](example/transformers/tests/contracts): the fleet lock and Transformers' expected results, owned by Transformers maintainers.

Framework CI fetches the contract and fixtures at pinned revisions and runs them: against each small fixture on every PR, where it runs on CPU in seconds, and against the real checkpoints after merge, nightly, and before release.

**[Full example](example/README.md):** a complete, runnable MIC for SmolLM text generation in Transformers and Transformers.js: manifest, contract as doc-builder runnable markdown, shared fixtures, fleet locks, reviewed baselines in each framework, and a runner sketch.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## Problem

Model compatibility is expensive to verify. Integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks, so only a small part of the model fleet runs on each PR. We find regressions late: after merge, in nightly jobs, during a release, or from users.

The code that validates a model is also separate from the code we show users, which is separate again from Hub snippets and other examples. Each copy drifts on its own: in a [recorded agent run](docs/agent-walkthrough.md), the `microsoft/Florence-2-base` card snippet failed as written (a missing import, unlisted dependencies, and remote code that breaks on every `transformers` version tried), and neither model tried came with a way to check that its output was right.

## The proposed standard

A MIC is defined per usage path and covers every checkpoint that shares it. A conforming MIC has the parts below; each links to the [full example](example/README.md).

1. **Repositories by owner.** Creators' model repositories are unchanged. A Hugging Face-owned [integration repository](example/hub/SmolLM-integration) holds the contract for one usage path and lists the checkpoints it covers. Hugging Face-owned [fixture repositories](example/hub/SmolLM-slim) hold only slim models: weights, the shrunk config, and copied processor files. Each [framework repository](example/transformers/tests/contracts) holds its own fleet lock and expected results.
2. **Fixed layouts:**

   ```text
   hf-internal-testing/SmolLM-integration      # Hub: what to run
     integration.json                          # manifest
     integrations/transformers/usage.md        # runnable markdown: documentation and test
     integrations/transformers.js/integration.js
   transformers/tests/contracts/               # framework repo: what to expect
     fleet.lock
     expectations/SmolLM/cpu/SmolLM-slim.json  # one baseline per target, platform, and variant
     expectations/SmolLM/cpu/SmolLM-135M.json
     expectations/SmolLM/cpu/SmolLM-135M.eager.json
   ```

3. **A manifest readable without running code.** [`integration.json`](example/hub/SmolLM-integration/integration.json), validated by a schema (written during the pilot), declares the checkpoints and the fixture that covers each, and per framework the entry point, artifacts, supported variants, unsupported targets with a reason, resource requirements, and the comparison rule for each output.
4. **Ordinary user code as the contract.** The contract ([runnable markdown](example/hub/SmolLM-integration/integrations/transformers/usage.md), [JavaScript](example/hub/SmolLM-integration/integrations/transformers.js/integration.js)) calls the public API users call (pipeline, processor, generate) in the configuration they get by default (attention implementation, dtype), and prints one JSON result line on stdout. Where a framework's docs support [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md), the contract is written as one, so the same code is rendered as documentation and run as a test.
5. **Checks beyond "it ran".** Generated tokens or task outputs, selected logits within calibrated tolerances, and, where training is supported, a finite loss, gradients on the expected modules, and a parameter update. Missing fields, non-finite values, and mismatches fail.
6. **Reviewed baselines in each framework repository** ([`expectations/`](example/transformers/tests/contracts/expectations/SmolLM/cpu)): one per fixture and one per checkpoint, for each variant and platform class (CPU, GPU) where they differ. Each records the revision, environment, device, and dtype it came from. A framework PR that changes behavior updates its baselines in the same PR. Outputs never update a baseline automatically ([`run.py --record`](example/run.py) only writes a candidate), and a build under test never defines its own expected answer.
7. **[Fixtures](example/hub/SmolLM-slim) shared by checkpoints that preprocess the same way.** Processor, tokenizer, and preprocessing files are copied byte for byte; only the weights and model config shrink. Checkpoints share a fixture when those files and every config field that shrinking keeps are identical, and the fixture build fails otherwise.
8. **Everything pinned.** Each framework runs contracts from its [fleet lock](example/transformers/tests/contracts/fleet.lock), listing the contract, fixture, checkpoint, converted artifact, and dependency revisions; its baselines are versioned with it.

The example's [`run.py`](example/run.py) sketches the shared runner. The manifest and result schemas and the real runner are what the pilot builds ([next steps](docs/next-steps.md), step 3). Details are in the [technical design](docs/design.md).

## Why on the Hub, owned by Hugging Face

Contracts need maintenance: inputs change, fixtures get regenerated, checkpoints get added. That should not depend on write access to a creator's repository or on the creator maintaining framework CI. Hugging Face owns the contracts and fixtures; creators keep owning their models and change nothing. Expected results change mostly when a framework changes, so each framework owns its own, and the PR that changes behavior also shows the expectation diff.

On the Hub, a contract is one versioned place per usage path that every framework can run, and the Hub can later show the per-checkpoint results that framework CI publishes.

## Pilot run

Run a CI-only pilot on 3–5 models, with existing tests kept running in parallel, and measure whether contracts catch real regressions, what they cost at fleet scale, and how much maintenance they create. Early local experiments are encouraging ([evidence](docs/evidence.md)). If the pilot confirms it, expand the fleet and then build the Hub-facing pieces.

## Documents

- [Full example](example/README.md): a runnable MIC for SmolLM text generation (three checkpoints, two fixtures) in Transformers and Transformers.js, with every part of the standard.
- [Evidence](docs/evidence.md): what the local experiments and the historical regression replay show.
- [Technical design](docs/design.md): contract layout, CI triggers, where code lives, checks, baselines, runner, pilot plan, open questions.
- [Small fixtures](docs/fixtures.md): the fixture project, from today's tiny models to frozen random and overfit fixtures.
- [Next steps](docs/next-steps.md): the validation plan and what is done.
- [Agent walkthrough](docs/agent-walkthrough.md): a recorded run of an agent taking two Hub models to working code, every tool call included, with a prompt to repeat it in other agents.
