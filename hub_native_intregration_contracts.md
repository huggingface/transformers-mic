# Proposal: Hub-native executable integration contracts for Transformers models

## Status

Draft proposal for discussion.

## Summary

Transformers currently owns and maintains many integration tests for
model checkpoints hosted on the Hugging Face Hub. This creates an
increasing maintenance burden: both the library and model repositories
evolve over time, while some checkpoints are too large to test
continuously in ordinary CI environments.

This proposal moves checkpoint-specific integration knowledge closer to
the model itself while keeping orchestration centralized.

A participating model would expose two standardized artifacts:

1.  **A canonical executable integration test**: a complete, valid
    Python program showing how the model is loaded and exercised with
    Transformers.
2.  **A canonical slim integration fixture**: a reduced checkpoint that preserves the overall architecture and processor-facing contract, is intentionally trained/overfit on a tiny deterministic example, and can be used for fast and meaningful continuous testing.

The Hub would understand these artifacts through standardized,
machine-readable metadata. It could execute the canonical integration
test on appropriate hardware and record an attestation that a particular
model revision was successfully validated against a particular
Transformers build.

The same executable integration contract would therefore serve several
purposes at once:

-   full-checkpoint integration validation;
-   executable documentation;
-   a canonical usage example for humans;
-   a machine-readable execution entry point for agents and tooling;
-   continuous low-cost training and generation testing through the slim integration fixture;
-   validation during private model-development workflows before
    upstreaming code to Transformers.

The core architectural idea is:

> **Distributed ownership and execution, centralized orchestration.**

Transformers should still be able to answer, every day, "does current
Transformers work with the model fleet?" The difference is that
Transformers no longer needs to own every checkpoint-specific test
implementation.

## Motivation

### Integration tests are currently centrally maintained

Today, many integration tests live in the Transformers repository even
though the checkpoints they exercise live on the Hub.

This creates an ownership mismatch. A test may depend on:

-   a particular model revision;
-   Transformers behavior;
-   tokenizer or processor behavior;
-   external dependencies;
-   hardware capabilities;
-   model-specific loading or generation logic.

Any of these can change independently.

As the number and diversity of models increase, centrally maintaining
checkpoint-specific integration behavior becomes increasingly expensive.

### The full integration fleet is re-run as Transformers evolves

A useful property of the current system is that when Transformers
changes, the integration fleet is run again regularly, typically at
least daily. This provides a broad regression signal across many real
checkpoints.

The proposed system should preserve that property.

The goal is not to stop broad continuous validation. The goal is to
change where integration definitions live, how they are discovered, and
where they execute.

### Large checkpoints are difficult to test continuously

Some models require substantial GPU memory, multiple GPUs, or
specialized hardware. Running the real checkpoint for every relevant CI
change can therefore be prohibitively expensive or simply unavailable.

As a result, the strongest integration test is often exactly the test
that cannot be run continuously.

### Usage examples are not necessarily executable contracts

Model cards commonly contain example code, but examples embedded in
prose are not necessarily continuously executed or machine-readable.

This creates duplication between:

-   documentation;
-   Transformers integration tests;
-   examples in model repositories;
-   code inferred or constructed by downstream tools and agents.

A standardized executable example could collapse much of this
duplication into one canonical artifact.

## Proposed model integration contract

A participating model repository exposes a standardized integration
contract.

Conceptually:

```text
model repository
├── config.json
├── model weights
├── ...
├── transformers_integration.json
└── integration_test.py

slim integration fixture repository (or related revision/artifact)
├── config.json
├── slim checkpoint weights
├── tokenizer / processor files
├── tiny training fixture
└── ...
```

The exact filenames and whether the slim fixture is stored in the same
repository or referenced as a related repository are open design
questions.

### Canonical executable integration test

`integration_test.py` is a complete, valid Python program rather than a
test fragment or prose snippet.

For example:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "org/model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

inputs = tokenizer("Hello", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=8)

print(tokenizer.decode(outputs[0]))
```

The exact validation semantics still need to be defined. Depending on
the model, the contract might verify successful execution, output
shapes/types, deterministic properties, or selected expected outputs.

The important property is that this is **real Transformers usage code
that can actually be executed**.

### Executable documentation

Because the canonical integration test is itself a normal Transformers
program, it can also become a canonical usage example.

This provides a useful property: the documentation example stays honest
because the platform can actually run it.

Model cards, documentation generators, IDE tooling, and other consumers
could link to or derive examples from the same artifact.

### Agent and tooling discoverability

Agents and automated tools often need to determine how a model should be
loaded and invoked.

Instead of inferring usage only from free-form README content, an agent
could discover a standardized integration contract and inspect a known
executable entry point.

This creates a machine-readable answer to:

> "How do I actually run this model with Transformers?"

The executable contract therefore acts as both a test interface and a
discoverable usage interface.

## Discoverability and metadata

The integration contract should be discoverable without requiring
consumers to import arbitrary model code.

One possibility is to add fields to `config.json`, but model
configuration and integration metadata have different responsibilities.

`config.json` primarily represents model configuration consumed by
Transformers configuration classes. The proposed integration contract
instead describes testing, execution requirements, validation and
related artifacts.

A cleaner design may therefore be a separate versioned manifest, for
example:

```json
{
  "schema_version": 1,
  "framework": "transformers",
  "integration_test": {
    "entrypoint": "integration_test.py"
  },
  "slim_fixture": {
    "model_id": "org/model-slim",
    "training_fixture": "fixtures/overfit.jsonl"
  },
  "resources": {
    "accelerator": "gpu",
    "gpu_count": 1,
    "min_gpu_memory_gb": 24,
    "min_system_memory_gb": 32,
    "timeout_seconds": 600
  }
}
```

The exact structure is illustrative.

### JSON Schema

The integration manifest is a good candidate for a formal, versioned
JSON Schema.

A schema would allow the Hub, Transformers CI, agents and third-party
tooling to validate and understand the contract without executing
Python.

It could define:

-   supported manifest versions;
-   required and optional fields;
-   integration entry points;
-   slim-fixture references and generation metadata;
-   dependency constraints;
-   hardware requirements;
-   timeout constraints;
-   validation modes.

The schema itself should be versioned so the contract can evolve without
silently breaking older tooling.

## Resource requirements

Resource requirements should be first-class metadata.

Before scheduling an integration test, infrastructure should be able to
determine what is needed without attempting execution and waiting for an
out-of-memory failure.

Potential declarative fields include:

-   CPU-only vs accelerator required;
-   accelerator type;
-   number of GPUs;
-   minimum VRAM per GPU;
-   minimum system RAM;
-   expected disk requirements;
-   timeout;
-   optional hardware capabilities or compute dtype requirements.

There is an important distinction between **declared requirements** and
**observed validation resources**.

### Declared requirements

Provided by the model author and used for scheduling.

### Observed resources

Recorded by the validation system when the test actually succeeds.

Keeping both provides traceability and allows infrastructure to improve
scheduling decisions over time.

## Slim integration fixtures

For large models, we need a reduced artifact that is cheap enough to run in PR and merge-time CI while still testing more than "the model does not crash."

The proposal therefore introduces a canonical **slim integration fixture**. The fixture is a reduced version of the architecture that preserves the overall model structure and processor-facing contract, but reduces expensive dimensions such as depth and width. It is then intentionally trained to overfit a tiny deterministic example so that the resulting checkpoint exhibits known behavior that can be asserted.

The intended workflow is:

```text
real model architecture + processor contract
                  ↓
          create slim configuration
                  ↓
      instantiate reduced checkpoint
                  ↓
  train on a tiny deterministic fixture
                  ↓
        intentionally overfit
                  ↓
       save canonical slim checkpoint
                  ↓
 run deterministic generation/inference test
                  ↓
           assert expected behavior
```

This gives us two valuable signals from the same fixture:

1. **Training works**: the reduced model can execute forward/backward/update steps and successfully overfit the tiny fixture.
2. **Generation/inference works**: the saved checkpoint can be loaded through the public Transformers API and produces a deterministic expected result.

This is stronger than only checking that a tiny randomly initialized model executes. A regression that silently changes generation, weight loading, cache behavior, decoding, or another part of the end-to-end path can be detected even if tensor shapes remain valid.

### Slim, not maximally tiny

A key principle is:

> **The fixture should be slim enough to be cheap, while preserving the architecture and external model contract closely enough to remain representative.**

The goal is not to minimize every config field or every byte. Some dimensions are safe to reduce aggressively, while others are coupled to the tokenizer, processor, or overall architecture.

Examples of generally shrinkable dimensions include:

- number of layers;
- hidden size;
- intermediate size;
- number of attention heads, subject to divisibility constraints;
- encoder/decoder depth;
- other dimensions that mainly determine compute.

Examples of interface-coupled dimensions that often need to remain compatible with the real processor include:

- `vocab_size`;
- BOS/EOS/PAD and other special token IDs;
- image size and patch-size relationships;
- number of input channels;
- audio feature dimensions;
- multimodal token IDs;
- other config values that constrain legal processor outputs.

For example, shrinking `vocab_size` to 99 while retaining a production tokenizer can make the fixture unusable as soon as the tokenizer emits token ID 100 or higher. Keeping the real tokenizer vocabulary may increase the embedding table somewhat, but it is often a better tradeoff than introducing model-specific tokenizer shrinking logic.

### Automatic generation by default, explicit override when needed

The default path should be **automatic slim-fixture generation from the real model package**.

Conceptually:

```text
real config + tokenizer / processor
              ↓
      generic slim generator
              ↓
shrink compute-heavy architectural dimensions
              +
preserve / infer processor-coupled dimensions
              ↓
       slim configuration
              ↓
      overfit tiny fixture
              ↓
 validate training + deterministic inference
```

Starting from the real model package is preferable to starting from model-tester values. Model testers are useful unit-test fixtures, but their arbitrary small values may not be compatible with the model's real tokenizer or processor. The integration fixture should instead be a reduced version of a real end-to-end model package.

Fully generic generation will not cover every architecture. Some models have special topology constraints, tied dimensions, multimodal relationships, or training requirements. Therefore the generator should support **explicit architecture-specific or model-author overrides**.

The intended policy is:

1. use generic automatic generation whenever possible;
2. preserve processor-facing dimensions unless there is a reliable joint transformation;
3. allow a small architecture-specific generation strategy when generic slimming is insufficient;
4. allow the model creator/maintainer to provide a canonical slim fixture when automatic generation cannot reliably produce one;
5. require the resulting fixture to pass both its overfit-training check and its deterministic integration test.

This keeps the common path automated while recognizing that the creator of a model is often best positioned to define a valid reduced representative.

### Deterministic behavioral assertions

The canonical test should assert meaningful expected behavior rather than only successful execution. For a generative model, the fixture may intentionally overfit a tiny prompt/target pair and then assert that greedy generation reproduces the target.

For example:

```python
inputs = processor(KNOWN_INPUT, return_tensors="pt")
outputs = model.generate(**inputs, do_sample=False, max_new_tokens=EXPECTED_LENGTH)
result = processor.decode(outputs[0])

assert result == KNOWN_EXPECTED_OUTPUT
```

Exact generated text will not be the right assertion for every architecture or execution environment. Depending on the model, a stable assertion may instead target class IDs, a short deterministic token sequence, logits within a tolerance, a known reconstruction, or another task-specific output.

The important requirement is:

> **A slim integration fixture should be intentionally trained so that its canonical integration test can assert deterministic expected behavior.**

The fixture and test should be designed for stability: deterministic decoding, short outputs, controlled seeds where relevant, and enough overfitting margin to avoid borderline behavior across supported kernels and dtypes.

### Same contract, different model IDs

Where practical, the same canonical integration program should run against either the slim fixture or the real checkpoint:

```text
integration_test.py --model-id org/model-slim
integration_test.py --model-id org/model
```

The slim execution uses deterministic assertions tailored to the overfit fixture. The real-checkpoint execution validates the published checkpoint against the same public API path, although its assertions may differ when real-model outputs are not stable enough for exact matching.

### Slim-fixture validation

Slim-fixture validation is fast and inexpensive enough for PR and merge-time CI. It should exercise both:

```text
slim config → tiny overfit training → PASS
slim checkpoint → canonical generation/inference test → PASS
```

It answers roughly:

> Does this architecture still train, serialize/load, and produce the expected deterministic end-to-end behavior through Transformers?

### Full-checkpoint validation

Full-checkpoint validation uses the real published checkpoint and its required hardware.

The registered real-checkpoint fleet should run nightly so that the distributed system preserves the strong compatibility guarantee of the current integration-test system.

It answers roughly:

> Does this exact published checkpoint execute successfully with this Transformers environment?

The slim fixture is therefore a **fast behavioral integration fixture**, never a substitute for validation of the real checkpoint.

## Central orchestration over a distributed model fleet

The new system should preserve the current ability to re-run broad
integration coverage whenever Transformers changes.

Conceptually:

```text
Transformers commit / nightly build
            ↓
Publish or identify candidate Transformers artifact
            ↓
Hub validation service
            ↓
Discover registered model integration contracts
            ↓
Schedule each contract on declared hardware
            ↓
Aggregate results for that Transformers revision
```

The Hub can maintain a registry or index of participating model
contracts. It should not need to crawl the entire Hub for every
Transformers change.

For each registered model, the scheduler would know at least:

```json
{
  "model": "org/model",
  "revision": "abc123",
  "integration_test": "integration_test.py",
  "slim_fixture": "org/model-slim",
  "resources": {
    "accelerator": "gpu",
    "gpu_count": 8,
    "min_gpu_memory_gb": 80
  }
}
```

A Transformers nightly then becomes conceptually:

> Validate `transformers@<sha>` against the registered model fleet.

### Suggested validation cadence

| Trigger | Validation |
| --- | --- |
| Transformers PR | Relevant slim integration fixtures when selection is available; otherwise a conservative subset |
| Merge to `main` | Broad or full slim-fixture fleet |
| Nightly | **All registered real checkpoints** |
| Release candidate | All registered real checkpoints, potentially with stricter or more exhaustive validation |
| Model update | That model's slim fixture and real checkpoint |

The nightly run should preserve the strongest guarantee of the current
system: current Transformers is tested against the **full registered
real-checkpoint fleet**. Slim integration fixtures serve a different purpose: fast,
low-cost feedback on PRs and merges while exercising meaningful learned behavior. A slim-fixture pass must never be
treated as equivalent to validation of the real checkpoint.

If a real checkpoint cannot run nightly because of exceptional hardware,
runtime, licensing, privacy, or access constraints, that limitation
should be recorded explicitly in the validation status rather than
silently substituting slim-fixture coverage.

### Affected-test selection

Over time, the system could understand which models are likely affected
by a change.

For example:

```text
change in modeling_llama.py
        ↓
models declaring architecture = llama
        ↓
run those contracts
```

This is an optimization, not a prerequisite. The initial system can
simply run all registered slim-fixture contracts on a regular cadence.

## Validation is a relation between revisions

A compatibility result should belong to a pair of immutable revisions
rather than to a model in the abstract.

Conceptually:

```text
model_revision × transformers_revision → result
```

For example:

```text
org/model @ 812bd4
transformers @ 98ac21
integration contract @ 812bd4

PASS
hardware: 8x H100
duration: 4m32s
```

This naturally produces a compatibility matrix over time:

| Model | Monday Transformers | Tuesday Transformers |
| --- | --- | --- |
| Model A | pass | pass |
| Model B | pass | fail |
| Model C | pass | pass |

A regression becomes immediately attributable to a concrete Transformers
revision.

## Bidirectional validation

The same infrastructure should work in both directions.

A Transformers change can trigger model revalidation:

```text
Transformers changed → revalidate model contracts
```

A model change can trigger validation against Transformers:

```text
Model changed → revalidate against current/released Transformers
```

This makes the Hub integration contract a stable interface between two
independently evolving systems.

## Introducing a new model architecture

Adding a new model is different from validating an already-supported
model because it introduces new Python code into Transformers.

The validation system must therefore support **unreleased Transformers
builds**, not only published versions.

A new architecture contribution should conceptually include three
pieces:

```text
new Transformers implementation
+
canonical slim integration fixture
+
canonical executable integration contract
```

These can be reviewed and validated together.

### Testing layers for a new architecture

The proposal does not replace ordinary Transformers unit tests.

A useful separation is:

| Layer | Owner | Purpose |
| --- | --- | --- |
| Unit tests | Transformers | Test implementation details, edge cases and invariants |
| Slim-fixture integration | Model contract + Transformers CI | Test training plus deterministic end-to-end inference cheaply |
| Full-checkpoint integration | Hub validation | Test the actual published checkpoint |

The canonical slim integration fixture is especially valuable for new architectures because it can exercise both training and deterministic end-to-end inference without requiring production-scale hardware.

## Private development before upstreaming

In practice, a new model architecture may be developed collaboratively
before there is an upstream Transformers PR.

A maintainer may work with a company using a private clone or fork of
Transformers. Once the implementation is ready, the patch is moved into
the upstream Transformers repository and committed or proposed for
merge.

The validation architecture should support this workflow directly.

Conceptually:

```text
Company model repository
        +
Maintainer/private Transformers fork
        ↓
private development and iteration
        ↓
run canonical integration contract
against arbitrary Transformers artifact
        ↓
slim fixture training and generation tests pass
full checkpoint passes when appropriate
        ↓
implementation considered ready
        ↓
patch moved to upstream Transformers
        ↓
run the same integration contract
against upstream candidate
        ↓
merge
        ↓
model joins normal nightly/release fleet
```

This leads to an important design requirement:

> **The unit of validation should be an arbitrary Transformers build
> artifact, not a Transformers PR or release.**

The validator should be able to consume something equivalent to:

```text
Transformers source:
    private-fork@4e8a21f

Model:
    company/new-model@98bc11

Contract:
    integration_test.py

Slim fixture:
    company/new-model-slim@01ca77
```

The Transformers artifact could be represented as a wheel, container
image, commit from an accessible repository, or another build artifact.
The validation service should not fundamentally care whether it came
from upstream Transformers, a maintainer's private fork, a pull request
build, or a release.

## Private and public attestations

Private development introduces an important distinction between
**successful validation** and **public Transformers support**.

During development:

```text
company/new-model
    × private-transformers-fork@abc123
    → PASS
```

This is useful to the company and maintainer, but it should not become a
public claim that the model works with released or upstream
Transformers.

After upstreaming:

```text
company/new-model
    × huggingface/transformers@def456
    → PASS
```

This result can become the basis for public compatibility metadata.

Possible lifecycle states include:

```text
contract present
    ↓
validated against private candidate
    ↓
validated against upstream candidate
    ↓
validated against main
    ↓
validated against release
```

The Hub could expose only the appropriate public states while keeping
private development results access-controlled.

## Upstream handoff as a validation gate

The same external contract should survive the transition from private
implementation to upstream Transformers.

```text
private implementation                upstream implementation
        │                                      │
        └──────── same integration contract ───┘
                         +
                    same slim integration fixture
```

This creates a strong handoff check:

> Does the exact model contract that passed against the private
> implementation also pass against the upstreamed implementation?

If it does not, the upstreaming process introduced a regression or
changed a behavior that needs to be understood.

This avoids reconstructing integration context when moving code
upstream.

## Implementation source and remote code

The canonical integration test should be a **consumer of the model
implementation**, not the implementation itself.

For an upstream Transformers model:

```text
Transformers repository
    └── modeling_newmodel.py    ← implementation

Hub model repository
    └── integration_test.py     ← canonical usage
```

This keeps the proposal conceptually separate from `trust_remote_code`.

Remote-code models could still participate, but the manifest should
distinguish implementation source, for example:

```json
{
  "implementation": {
    "source": "transformers"
  }
}
```

versus:

```json
{
  "implementation": {
    "source": "repository",
    "requires_remote_code": true
  }
}
```

A stronger "validated Transformers model" status may choose to require
that the implementation source is upstream Transformers.

## Hub validation and attestation

The Hub could execute the canonical integration test when a model is
submitted for validation, when a Transformers build changes, when a
model changes, or when a stronger compatibility guarantee is requested.

A successful execution would produce an attestation tied to immutable
inputs, for example:

-   model repository;
-   model revision / commit SHA;
-   integration contract version;
-   Transformers artifact identity;
-   Transformers version or commit;
-   Python version;
-   PyTorch version;
-   relevant dependency versions;
-   execution hardware;
-   execution timestamp;
-   test result;
-   whether the validation result is private or public.

The Hub could then expose a machine-readable and human-visible status
such as:

> Validated with Transformers

Critically, this should not be an unqualified permanent badge.
Validation is meaningful only relative to a particular model revision
and software environment.

## Security and access control

A Python integration test stored in a model repository is arbitrary code
and must not be treated as trusted merely because it follows a naming
convention.

Hub execution therefore needs an explicit security model, potentially
including:

-   isolated/sandboxed execution;
-   network restrictions;
-   secret isolation;
-   resource quotas;
-   immutable revisions;
-   explicit trust or approval policies;
-   provenance for validation results.

Private development adds further requirements:

-   access-controlled model repositories;
-   access-controlled Transformers artifacts;
-   private logs and results;
-   no accidental publication of private validation metadata;
-   a clear transition from private validation to public attestation.

The manifest itself should remain safely inspectable without executing
the Python entry point.

## Design principles

**Distributed ownership, centralized orchestration**\
Model-specific test definitions live with the model, while a central
service can still execute a broad compatibility fleet after Transformers
changes.

**Configuration vs integration contract**\
Model architecture configuration remains separate from testing and
validation metadata.

**Slim-fixture validation vs full validation**\
A slim integration fixture verifies that reduced training and deterministic end-to-end inference still behave as expected; it does not prove that the production checkpoint works.

**Unit tests vs integration contracts**\
Transformers still owns implementation-level unit tests. The Hub
contract captures checkpoint-specific end-to-end behavior.

**Declared resources vs observed resources**\
Authors describe what should be required; infrastructure records what
was actually used.

**Executable code vs attestation**\
The repository supplies the executable contract; the Hub supplies
evidence that a particular immutable version of that contract
successfully ran in a particular environment.

**Arbitrary build artifacts, not only releases**\
The validation system must work against private forks, candidate wheels,
upstream commits, `main`, release candidates and published releases.

**Private validation is not public support**\
Passing against a private Transformers fork should remain distinct from
passing against upstream or released Transformers.

## Expected benefits

### Reduced central maintenance

Checkpoint-specific knowledge moves closer to the checkpoint and its
maintainers.

### Preserved regression coverage

Transformers can still run a broad model fleet daily or on every
significant change, preserving the strongest property of today's
integration system.

### Scalable testing

Slim integration fixtures allow training and deterministic behavior to be tested without requiring production-scale hardware on every CI run.

### Better new-model development workflow

The same contract can be used during private implementation, upstream
handoff and long-term maintenance.

### Stronger documentation

Canonical examples can be continuously executed rather than copied
between tests and README snippets.

### Better agent interoperability

Agents gain a standardized, machine-discoverable entry point instead of
relying primarily on heuristics over free-form documentation.

### Explicit compatibility evidence

Hub validation can provide evidence tied to concrete model and library
revisions instead of an ambiguous notion that a model "supports
Transformers."

### Ecosystem extensibility

A versioned manifest and schema can be consumed by the Hub,
Transformers, agents, CI systems and third-party tools without each
inventing its own convention.

## Open questions

Several parts require further design work:

-   Should the manifest live in the model repository, Hub metadata, or
    both?
-   What should the canonical filename be?
-   Should the integration program follow a strict API or simply be an
    executable Python program?
-   What constitutes a successful integration test?
-   How should expected outputs be represented when outputs are
    nondeterministic?
-   How should dependencies beyond Transformers be declared?
-   How should multiple tasks or execution modes be represented?
-   Can a model expose multiple integration tests?
-   How should multimodal inputs and test assets be packaged?
-   Should slim integration fixtures live in the same repository or separate repositories?
-   What architectural and processor-facing invariants must a slim fixture preserve to qualify as canonical?
-   Who is allowed to mark a slim fixture as canonical?
-   When should full-checkpoint validation be triggered?
-   Which models should run on every PR, merge, nightly and release
    candidate?
-   How should affected-test selection map Transformers source files to
    model contracts?
-   When does an attestation expire or become stale?
-   What sandboxing/trust model is required to execute
    repository-provided Python?
-   How should resource declarations map to Hub hardware SKUs?
-   How should private Transformers artifacts be supplied securely to
    the validator?
-   What is the exact transition from a private validation result to a
    public compatibility attestation?
-   Should a strong public Transformers validation badge require
    implementation source = upstream Transformers?

## Possible rollout

A staged rollout could reduce risk.

### Phase 1 - Convention

Define an experimental manifest, JSON Schema and canonical integration
script convention for a small set of models.

### Phase 2 - Slim-fixture fleet

Register canonical slim integration fixtures and run their training and deterministic generation contracts centrally against Transformers `main` and selected PRs.

### Phase 3 - Hub execution and full-checkpoint validation

Allow the Hub to schedule real checkpoints according to declared
hardware requirements and record structured validation results.

### Phase 4 - Arbitrary Transformers artifacts

Support candidate wheels, private forks and other unreleased artifacts
so the same system can be used during new-model development and upstream
handoff.

### Phase 5 - Attestation and discoverability

Expose public validation status through Hub metadata/API and make the
integration contract discoverable to agents and tooling.

### Phase 6 - Smarter scheduling

Add affected-test selection, cost-aware scheduling and broader ecosystem
adoption.

## Core idea

The central idea is to make **model compatibility an executable,
versioned and discoverable contract owned alongside the model**.

The Hub provides discovery, scheduling, execution and attestation.
Transformers provides the runtime, implementation-level tests and
compatibility target. Model authors provide the canonical way to
exercise their model and, where necessary, a slim, intentionally overfit integration fixture.

For new architectures, the same contract can be validated first against
a private Transformers implementation, then against the upstreamed
implementation, and finally against `main` and releases.

The result is not the removal of the Transformers integration test
suite. It is its evolution into a **centrally orchestrated, distributed
compatibility test fleet** whose model-specific test definitions live
with the models they validate.