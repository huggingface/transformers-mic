# Proposal: Model integration contracts — CI first, Hub later

## Summary

We should publish a small, executable integration contract on the Hub for each model/framework combination we want to support, and run it from framework CI.

A contract is ordinary user code — load the model, call the public API users call, check the result — plus a small fixture and the expected results. It lives in a Hugging Face-owned companion repository next to the model, not in the model repository and not in the framework repository:

- `meta-llama/Llama-3.2-1B`: the real model, owned by the creator, unchanged;
- `hf-internal-testing/Llama-3.2-1B-integration`: the contract and expected results, owned by Hugging Face;
- `hf-internal-testing/Llama-3.2-1B-slim`: a small fixture of the same model, owned by Hugging Face.

Framework CI fetches these at pinned revisions and runs them: against the small fixture on every PR, and against the real checkpoint after merge, nightly, and before release.

The first goal is CI, not a new Hub feature. If this works, the same contracts can later back compatibility status, tested snippets, Colabs, APIs, or agent usage.

## Problem

Model compatibility is expensive to verify. Integration coverage depends on real checkpoints, large downloads, GPUs, framework-specific test code, or manual checks, so only a small part of the model fleet runs on each PR. We find regressions late: after merge, in nightly jobs, during a release, or from users.

The code that validates a model is also separate from the code we show users, which is separate again from Hub snippets and other examples. Each copy drifts on its own.

## Why on the Hub, owned by Hugging Face

Contracts need maintenance: a test changes when framework behavior changes, a fixture gets regenerated, an expectation is updated after review, obsolete coverage is retired. That should not depend on write access to a creator's repository or on the creator maintaining framework CI. So Hugging Face owns the contracts, and creators keep owning their models and do not have to change anything.

Putting contracts on the Hub rather than inside a framework repository gives one place per model that every framework can use (Transformers, Transformers.js, others), that is versioned like the model itself, and that the Hub can later read to show what was actually tested.

## Why it is cheap enough to run on every PR

The small fixture is the real model's config shrunk to a few million parameters for most models, with the real tokenizer and processor, frozen weights, and recorded outputs. It runs on CPU in milliseconds to a few hundred milliseconds, so PR checks can cover far more of the fleet than real checkpoints can. Real checkpoints stay in the slower merge, nightly, and release runs.

## What we know so far

- **Small fixtures detect what matters, if the checks are numerical.** On synthetic faults across six model families, frozen random fixtures with recorded logits, loss, and gradient checks caught 120 of 126 fault cases; a passing forward pass caught none. The misses were one model (Whisper) whose random fixture ignores its inputs; overfitting that fixture fixed it.
- **The contract must call what users call.** Replaying seven real Transformers regressions, generic forward/generate checks missed every processor and tokenizer bug; the same small fixtures called through the affected public API caught them.
- **No false alarms so far.** Fixtures frozen on today's `main` stayed clean across ten benign changes from the past two months.

These are small local experiments, not fleet-level evidence. Details are in [Small fixtures](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_fixtures-md) and the [technical design](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_design-md).

## What we propose

Run a CI-only pilot on 3–5 models (a text model, a multimodal model, a GPU-heavy model, one that is painful to maintain today, and one Transformers.js integration), with existing tests kept running in parallel. Measure whether contracts catch real regressions, what they cost at fleet scale, and how much maintenance they create.

If the pilot shows good regression coverage at much lower PR cost without large ongoing maintenance, expand the fleet and then build the Hub-facing pieces.

## Documents

- [Technical design](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_design-md): contract layout, CI triggers, where code lives, checks, baselines, runner, historical regression replay, pilot plan, open questions.
- [Small fixtures](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_fixtures-md): the fixture project, from today's tiny models to frozen random and overfit fixtures, with the local experiments.
- [Next steps](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b#file-model_integration_contracts_next_steps-md): the validation plan and what is done.
