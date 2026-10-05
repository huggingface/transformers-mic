# Proposal: Hub-native integration contracts for Transformers models

## Summary

Transformers currently maintains many checkpoint-specific integration tests in the library, while the checkpoints themselves live on the Hub. This creates maintenance overhead, and very large models are too expensive to test on every PR.

The proposal is to move **model-specific integration knowledge next to the model**, while keeping **orchestration centralized**.

Each participating model exposes:

1. **A canonical executable integration test**: a real Python program showing how to load and use the model with Transformers.
2. **A canonical slim integration fixture**: a reduced version of the architecture, intentionally overfit on a tiny deterministic example, so training and inference can be tested cheaply with meaningful assertions.
3. **Machine-readable metadata**: where the test and slim fixture live, what hardware is required, and what Transformers artifact should be tested.

The Hub validation service can then run the same contract against:

- a private Transformers clone during development;
- an upstream PR or commit;
- `main`;
- releases;
- the real production checkpoint during nightly validation.

The result is a **centrally orchestrated, distributed compatibility test fleet**.


![architecture](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b/raw/425cea148d0c3bd2831b50c03afb1099be90888a/pipeline.png)


## Prototype Model Card

Below is an example of a possible UI integration in the model card.


![card](https://gist.github.com/tarekziade/c3f203edbb8b97e40b86a97c5909a81b/raw/c5ce7d2ccff67535927d3a803e76048b1dfb18db/model-card.png)



## Why change

Today, checkpoint-specific tests are centrally owned by Transformers even though the model, processor, tokenizer, weights, and usage conventions live on the Hub and evolve independently.

The new system should preserve the strongest property of the current setup: **current Transformers is regularly tested against the real model fleet**.

The change is mainly about ownership and execution:

```text
Today
Transformers repo owns test + CI runs it

Proposed
Model repo owns contract + Hub/CI runs it
```

## Model integration contract

A participating model repository could expose something like:

```text
model repo
├── config.json
├── weights
├── processor/tokenizer files
├── transformers_integration.json
└── integration_test.py
```

The integration script is normal, executable Transformers code. It is simultaneously:

- an integration test;
- executable documentation;
- a canonical usage example;
- a machine-discoverable entry point for agents and tooling.

A separate versioned manifest is preferable to overloading `config.json`, because model configuration and validation metadata have different responsibilities.

Example:

```json
{
  "schema_version": 1,
  "integration_test": "integration_test.py",
  "slim_fixture": "org/model-slim",
  "resources": {
    "accelerator": "gpu",
    "gpu_count": 1,
    "min_gpu_memory_gb": 24,
    "timeout_seconds": 600
  }
}
```

The manifest should have a JSON Schema so the Hub, CI, agents, and other tooling can validate it without executing Python.

## Slim integration fixtures

A slim fixture is not just a randomly initialized tiny model. It should preserve the architecture and processor-facing contract closely enough to model real behavior, while reducing expensive dimensions such as depth and width.

The intended workflow is:

```text
real config + processor
        ↓
create slim config
        ↓
instantiate reduced model
        ↓
train on tiny deterministic data
        ↓
intentionally overfit
        ↓
save checkpoint
        ↓
run deterministic integration test
        ↓
assert expected behavior
```

This tests two things cheaply:

- **training works**: forward, backward, optimizer/update path;
- **inference works**: save/load, generation or task-specific inference, processor/tokenizer path, cache/attention behavior, decoding, etc.

The important principle is:

> **Slim means computationally small, not minimal in every config field.**

Processor-coupled dimensions such as `vocab_size`, special token IDs, image/patch relationships, audio dimensions, or multimodal token IDs should usually stay compatible with the real processor.

### Generation strategy

Automatic generation should be the default:

1. start from the real model config and processor;
2. shrink compute-heavy architectural dimensions;
3. preserve processor-coupled dimensions;
4. train/overfit the fixture;
5. validate it with the canonical integration test.

Generic generation will not work for every architecture, so the system should allow **small architecture-specific strategies or model-author overrides**. If automatic generation is not reliable, the model creator can provide the canonical slim fixture directly.

A fixture is valid only if its training check and deterministic integration test pass.

## Validation cadence

The slim fixture and the real checkpoint serve different purposes.

| Trigger | Validation |
| --- | --- |
| Transformers PR | Relevant slim fixtures |
| Merge to `main` | **Relevant real checkpoints (run-slow)** |
| Nightly | **All registered real checkpoints** |
| Release candidate | All real checkpoints, with stricter checks if useful |
| Model update | That model's slim fixture + real checkpoint |

A slim-fixture pass must never be presented as equivalent to real-checkpoint validation.

If a real checkpoint cannot run nightly because of exceptional hardware, privacy, licensing, or access constraints, that should be explicit in its status.

## Validation across Transformers revisions

A validation result belongs to immutable inputs:

```text
model revision × Transformers revision/build → result
```

The Transformers side must not be limited to public releases. The validator should accept an arbitrary build artifact such as a wheel, commit, container, PR build, or private fork.

This is required for new-model development, where maintainers often collaborate with companies using a private Transformers clone before the patch is upstreamed.

The same contract can therefore follow the whole lifecycle:

```text
private Transformers fork
        ↓
validate model + slim fixture
        ↓
move patch upstream
        ↓
validate same contract against upstream candidate
        ↓
merge
        ↓
validate against main/releases/nightly fleet
```

Private validation should remain distinct from public support. A model passing against a private fork is not yet "validated with Transformers" publicly.

## What remains in Transformers

This proposal does not replace ordinary library tests.

Transformers still owns unit tests for implementation details, edge cases, and invariants.

The separation is:

- **unit tests**: implementation correctness inside Transformers;
- **slim-fixture tests**: cheap end-to-end training and deterministic inference;
- **real-checkpoint tests**: actual compatibility with the published model.

## Attestation and discoverability

When a validation succeeds, the Hub can record an attestation containing at least:

- model revision;
- integration-contract revision;
- Transformers build/revision;
- relevant dependency versions;
- hardware used;
- timestamp;
- result;
- private/public visibility.

This can power a human-visible and machine-readable compatibility status while avoiding a permanent, unqualified "works with Transformers" badge.

## Security

`integration_test.py` is arbitrary repository code, so execution needs isolation, resource limits, secret protection, immutable revisions, and access control for private models/builds.

The manifest must remain safely inspectable without executing the Python entry point.

## Pilot implementation

The pilot should prove the workflow with minimal infrastructure rather than solve every edge case.

### Scope

Pick **3-5 representative models**:

- one standard text model;
- one multimodal or processor-heavy model;
- one model that needs non-trivial GPU resources;
- ideally one model currently painful to maintain in integration tests.

### Step 1: Define the contract

For each pilot model, add:

- `integration_test.py`;
- a small `transformers_integration.json` manifest;
- a canonical slim fixture.

Keep the manifest intentionally small: entry point, slim fixture, resource requirements, and schema version.

### Step 2: Build the slim-fixture generator


**Work in progress by Yih-Dar**

Start from the existing tiny-model tooling, but change the target from "small random model" to "slim deterministic fixture":

- derive from the real config + processor;
- shrink only compute-heavy dimensions;
- support per-architecture overrides;
- train on a tiny example until overfit;
- save the resulting checkpoint;
- run the deterministic integration assertion.

### Step 3: Add a generic runner

Create one runner that accepts:

```text
model revision
Transformers build artifact
integration manifest
```

and executes the contract on the requested hardware.

For the pilot, the Transformers artifact can simply be a wheel or source checkout.

### Step 4: Wire CI

Use the pilot models in two paths:

- PR / `main`: run their slim fixtures;
- nightly: run their real checkpoints.

Keep the current integration tests in parallel during the pilot so results can be compared.

### Step 5: Test private-to-upstream handoff

For one new-model contribution, run the same contract first against a private/candidate Transformers build, then against the upstreamed patch. This validates that the design works before a PR exists.

### Step 6: Evaluate

After a few weeks, compare:

- maintenance effort versus existing integration tests;
- runtime and cost;
- failure signal quality;
- how often slim fixtures catch regressions before nightly real-checkpoint tests;
- how much architecture-specific override logic is required;
- whether the same integration script is useful for docs/agents.

If the pilot works, the next step is to move more existing integration tests to the contract model and add Hub-native scheduling/attestation.

## Core idea

> **Model compatibility becomes an executable, versioned, discoverable contract owned alongside the model.**

The model owner provides the contract and slim fixture. Transformers keeps implementation-level tests. The Hub provides discovery, execution, scheduling, and attestation. Real checkpoints still run nightly; slim fixtures provide fast feedback earlier in the development cycle.
