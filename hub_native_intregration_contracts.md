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
2.  **A canonical tiny model**: a very small checkpoint preserving the
    relevant architecture and execution path, intended for fast and
    inexpensive continuous testing.

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
-   continuous low-cost architecture testing through the tiny model;
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

tiny model repository (or related revision/artifact)
├── config.json
├── tiny weights
└── ...
```

The exact filenames and whether the tiny model is stored in the same
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
  "tiny_model": {
    "model_id": "org/model-tiny"
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
-   tiny-model references;
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

## Tiny models

For very large models, a small architectural representative makes it
possible to exercise the Transformers implementation quickly and
cheaply. The proposal therefore standardizes a relationship between a
production checkpoint and a canonical **tiny model**.

The tiny model is not intended to reproduce the quality or functional
behavior of the full checkpoint. Its purpose is to preserve enough of
the architecture and end-to-end execution path to catch integration
regressions at low cost.

A key principle is:

> **Tiny means computationally tiny, not necessarily minimal in every
> tensor or processor dimension.**

Some model configuration attributes are tightly coupled to the tokenizer
or processor. For example, reducing `vocab_size` while keeping the
production tokenizer can make an otherwise valid tiny model fail as soon
as the tokenizer emits an ID outside the reduced embedding table.
Similar coupling can exist for special token IDs, image dimensions,
patch sizes, audio feature dimensions, multimodal tokens, and other
processor-facing attributes.

Tiny-model generation should therefore distinguish between:

-   **shrinkable architectural dimensions**, such as number of layers,
    hidden size, intermediate size, and other dimensions that primarily
    determine compute;
-   **interface-coupled dimensions**, which should normally remain
    compatible with the real tokenizer/processor unless the generator
    has a robust way to transform both sides together.

This favors reducing compute-heavy architectural dimensions while
preserving the real processor-facing contract.

### Automatic generation by default, explicit override when needed

The default path should be **automatic tiny-model generation from the
real model package**.

Conceptually:

```text
real model config + tokenizer / processor
                  ↓
          tiny-model generator
                  ↓
    shrink expensive architecture dimensions
                  +
    preserve / infer interface-coupled dimensions
                  ↓
          generated tiny model
                  ↓
       canonical integration test
                  ↓
                 PASS
```

Starting from the real model package is preferable to starting from
unit-test fixtures or model-tester values: the goal is to create a
reduced version of a real end-to-end model contract, including its
processor/tokenizer relationships.

A generated tiny model should not be considered valid merely because the
model can be instantiated or execute a synthetic forward pass. The
stronger criterion is that the model's **canonical integration program
executes successfully against the generated tiny artifact**.

Ideally, the same integration program can be parameterized with either
model ID:

```text
integration_test.py --model-id org/model-tiny
integration_test.py --model-id org/model
```

This gives a concrete definition:

> A canonical tiny model is a reduced checkpoint against which the
> model's canonical integration contract executes successfully.

Fully generic shrinking will not cover every Transformers architecture.
Some models have architecture-specific constraints or processor
relationships that require specialized handling. Therefore, the
generator should support **explicit model-author or
architecture-specific overrides**.

The intended policy is:

1.  use generic automatic generation whenever possible;
2.  allow a small architecture-specific generation strategy or override
    when generic shrinking is insufficient;
3.  allow the model creator/maintainer to provide a canonical tiny
    artifact when automatic generation cannot reliably produce one;
4.  validate every resulting tiny artifact with the same canonical
    integration contract.

This keeps the common path automated without forcing complex
architectures into fragile generic rules. The creator of a model remains
the final authority on how to produce a valid reduced representative
when specialized knowledge is required.

The tiny artifact can be used for:

-   model initialization;
-   forward passes;
-   generation paths;
-   serialization and loading;
-   processor/tokenizer integration;
-   attention implementations;
-   cache behavior;
-   device placement paths;
-   architecture-specific regressions.

### Tiny validation

Tiny validation is fast, inexpensive, and suitable for PR and merge-time
CI.

It answers roughly:

> Does this architecture and its end-to-end Transformers integration
> still execute correctly?

### Full-checkpoint validation

Full-checkpoint validation uses the real published checkpoint and its
required hardware.

The registered real-checkpoint fleet should run nightly so that the
distributed system preserves the strong compatibility guarantee of the
current integration-test system.

It answers roughly:

> Does this exact published checkpoint execute successfully with this
> Transformers environment?

The tiny model should therefore be treated as an **architectural test
fixture for fast feedback**, never as a substitute for validation of the
real checkpoint.

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
  "tiny_model": "org/model-tiny",
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
| Transformers PR | Relevant tiny models when selection is available; otherwise a conservative subset |
| Merge to `main` | Broad or full tiny-model fleet |
| Nightly | **All registered real checkpoints** |
| Release candidate | All registered real checkpoints, potentially with stricter or more exhaustive validation |
| Model update | That model's tiny model and real checkpoint |

The nightly run should preserve the strongest guarantee of the current
system: current Transformers is tested against the **full registered
real-checkpoint fleet**. Tiny models serve a different purpose: fast,
low-cost feedback on PRs and merges. A tiny-model pass must never be
treated as equivalent to validation of the real checkpoint.

If a real checkpoint cannot run nightly because of exceptional hardware,
runtime, licensing, privacy, or access constraints, that limitation
should be recorded explicitly in the validation status rather than
silently substituting tiny-model coverage.

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
simply run all registered tiny contracts on a regular cadence.

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
canonical tiny checkpoint
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
| Tiny-model integration | Model contract + Transformers CI | Test the complete public API cheaply |
| Full-checkpoint integration | Hub validation | Test the actual published checkpoint |

The canonical tiny model is especially valuable for new architectures
because it can exercise the end-to-end implementation without requiring
production-scale hardware.

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
tiny model passes
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

Tiny model:
    company/new-model-tiny@01ca77
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
                    same tiny model
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

**Tiny validation vs full validation**\
A tiny model verifies architectural integration cheaply; it does not
prove that the production checkpoint works.

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

Tiny models allow architecture-level integration testing without
requiring production-scale hardware on every CI run.

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
-   Should tiny models live in the same repository or separate
    repositories?
-   What invariants must a tiny model preserve to qualify as canonical?
-   Who is allowed to mark a tiny model as canonical?
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

### Phase 2 - Tiny-model fleet

Register canonical tiny models and run their contracts centrally against
Transformers `main` and selected PRs.

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
exercise their model and, where necessary, a tiny architectural
representative.

For new architectures, the same contract can be validated first against
a private Transformers implementation, then against the upstreamed
implementation, and finally against `main` and releases.

The result is not the removal of the Transformers integration test
suite. It is its evolution into a **centrally orchestrated, distributed
compatibility test fleet** whose model-specific test definitions live
with the models they validate.