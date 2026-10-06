# MICs * Model integration contracts

**A proposed standard** for publishing, next to any model on the Hub, an executable check that the model works in a given framework, run by that framework's CI on every change.

## Summary

We should publish a small, executable integration contract on the Hub for each model/framework combination we want to support, and run it from framework CI.

A contract is ordinary user code — load the model, call the public API users call, check the result — plus a small fixture and the expected results. It lives in a Hugging Face-owned companion repository next to the model, not in the model repository and not in the framework repository:

- `meta-llama/Llama-3.2-1B`: the real model, owned by the creator, unchanged;
- `hf-internal-testing/Llama-3.2-1B-integration`: the contract and expected results, owned by Hugging Face;
- `hf-internal-testing/Llama-3.2-1B-slim`: a small fixture of the same model, owned by Hugging Face.

Framework CI fetches these at pinned revisions and runs them: against the small fixture on every PR, where it runs on CPU in milliseconds, and against the real checkpoint after merge, nightly, and before release.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## The proposed standard

A MIC is defined per model and framework. A conforming MIC has:

1. **Three repositories, three owners.** The creator's model repository is unchanged. A Hugging Face-owned integration repository holds the contract and expected results; a Hugging Face-owned fixture repository holds a slim copy of the model.
2. **A fixed layout** in the integration repository:

   ```text
   integration.json              # manifest
   integrations/
     transformers/integration.py
     transformers.js/integration.js
   expectations/                 # slim and real baselines
   ```

3. **A manifest readable without running code.** `integration.json`, validated by a schema, declares the framework and runtime, artifact repositories, entry point, supported variants, expected outputs, and resource requirements.
4. **Ordinary user code as the contract.** The integration module calls the public API users call (pipeline, processor, generate) in the configuration they get by default (attention implementation, dtype), and prints one JSON result line on stdout.
5. **Checks beyond "it ran".** Generated tokens or task outputs, selected logits within calibrated tolerances, and, where training is supported, a finite loss, gradients on the expected modules, and a parameter update. Missing fields, non-finite values, and mismatches fail.
6. **Two reviewed baselines**, one for the slim fixture and one for the real checkpoint. Outputs never update a baseline automatically, and a build under test never defines its own expected answer.
7. **Fixtures that keep the real preprocessing.** Processor, tokenizer, and preprocessing files are copied byte for byte; only the weights and model config shrink.
8. **Everything pinned.** Framework CI runs contracts from a fleet lock listing the model, processor, converted artifact, contract, fixture, baseline, and dependency revisions.

The manifest and result schemas and the shared runner are what the pilot builds ([next steps](docs/next-steps.md), step 3). Details are in the [technical design](docs/design.md).

## Problem

Model compatibility is expensive to verify. Integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks, so only a small part of the model fleet runs on each PR. We find regressions late: after merge, in nightly jobs, during a release, or from users.

The code that validates a model is also separate from the code we show users, which is separate again from Hub snippets and other examples. Each copy drifts on its own: in a [recorded agent run](docs/agent-walkthrough.md), the `microsoft/Florence-2-base` card snippet failed as written (a missing import, unlisted dependencies, and remote code that breaks on every `transformers` version tried), and neither model tried came with a way to check that its output was right.

## Why on the Hub, owned by Hugging Face

Contracts need maintenance: tests change with framework behavior, fixtures get regenerated, expectations get reviewed and updated. That should not depend on write access to a creator's repository or on the creator maintaining framework CI. Hugging Face owns the contracts; creators keep owning their models and change nothing.

On the Hub, a contract is one versioned place per model that every framework can use, and that the Hub can later read to show what was actually tested.

## Pilot run

Run a CI-only pilot on 3–5 models, with existing tests kept running in parallel, and measure whether contracts catch real regressions, what they cost at fleet scale, and how much maintenance they create. Early local experiments are encouraging ([evidence](docs/evidence.md)). If the pilot confirms it, expand the fleet and then build the Hub-facing pieces.

## Documents

- [Evidence](docs/evidence.md): what the local experiments and the historical regression replay show.
- [Technical design](docs/design.md): contract layout, CI triggers, where code lives, checks, baselines, runner, pilot plan, open questions.
- [Small fixtures](docs/fixtures.md): the fixture project, from today's tiny models to frozen random and overfit fixtures.
- [Next steps](docs/next-steps.md): the validation plan and what is done.
- [Agent walkthrough](docs/agent-walkthrough.md): a recorded run of an agent taking two Hub models to working code, every tool call included, with a prompt to repeat it in other agents.
