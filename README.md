# Proposal: Model integration contracts — CI first, Hub later

## Summary

We should publish a small, executable integration contract on the Hub for each model/framework combination we want to support, and run it from framework CI.

A contract is ordinary user code — load the model, call the public API users call, check the result — plus a small fixture and the expected results. It lives in a Hugging Face-owned companion repository next to the model, not in the model repository and not in the framework repository:

- `meta-llama/Llama-3.2-1B`: the real model, owned by the creator, unchanged;
- `hf-internal-testing/Llama-3.2-1B-integration`: the contract and expected results, owned by Hugging Face;
- `hf-internal-testing/Llama-3.2-1B-slim`: a small fixture of the same model, owned by Hugging Face.

Framework CI fetches these at pinned revisions and runs them: against the small fixture on every PR, where it runs on CPU in milliseconds, and against the real checkpoint after merge, nightly, and before release.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## Problem

Model compatibility is expensive to verify. Integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks, so only a small part of the model fleet runs on each PR. We find regressions late: after merge, in nightly jobs, during a release, or from users.

The code that validates a model is also separate from the code we show users, which is separate again from Hub snippets and other examples. Each copy drifts on its own: in a [recorded agent run](docs/agent-walkthrough.md), the `microsoft/Florence-2-base` card snippet failed as written (a missing import, unlisted dependencies, and remote code that breaks on every `transformers` version tried), and neither model tried came with a way to check that its output was right.

## Why on the Hub, owned by Hugging Face

Contracts need maintenance: tests change with framework behavior, fixtures get regenerated, expectations get reviewed and updated. That should not depend on write access to a creator's repository or on the creator maintaining framework CI. Hugging Face owns the contracts; creators keep owning their models and change nothing.

On the Hub, a contract is one versioned place per model that every framework can use, and that the Hub can later read to show what was actually tested.

## What we propose

Run a CI-only pilot on 3–5 models, with existing tests kept running in parallel, and measure whether contracts catch real regressions, what they cost at fleet scale, and how much maintenance they create. Early local experiments are encouraging ([evidence](docs/evidence.md)). If the pilot confirms it, expand the fleet and then build the Hub-facing pieces.

## Documents

- [Evidence](docs/evidence.md): what the local experiments and the historical regression replay show.
- [Technical design](docs/design.md): contract layout, CI triggers, where code lives, checks, baselines, runner, pilot plan, open questions.
- [Small fixtures](docs/fixtures.md): the fixture project, from today's tiny models to frozen random and overfit fixtures.
- [Next steps](docs/next-steps.md): the validation plan and what is done.
- [Agent walkthrough](docs/agent-walkthrough.md): a recorded run of an agent taking two Hub models to working code, every tool call included, with a prompt to repeat it in other agents.
