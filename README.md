# transformers-mic

![logo](/docs/mic-logo.png)

A **Model Integration Contract (MIC)** is a proposed standard: next to a Hub
model, an executable check that it works in a framework, run by that
framework's CI. Each MIC has three parts:

- a **fixture**: a shrunk model built from the real checkpoint's config, with its
  tokenizer and processor files copied unchanged
- a **usage path**: runnable Markdown that does what users do, such as generating
  text or running a training step
- **baselines**: the outputs that usage produced, reviewed by a human

Fixtures run on framework PRs (CPU, seconds); real checkpoints run after merge,
nightly and before release. **A slim pass is a fast regression signal, not a pass
for the real model.**

This repository holds the `tmic` CLI, the fixture builder and the runner. MICs are
published in [`huggingface-mic`](https://huggingface.co/huggingface-mic) and pinned
by the [fleet dataset](https://huggingface.co/datasets/huggingface-mic/transformers-fleet);
coverage is in the [coverage space](https://huggingface.co/spaces/huggingface-mic/mic-coverage).
In Transformers, the contracts job runs on PRs and reports failures without
blocking ([status](docs/evidence.md#transformers-ci-proof)).

![A proposed link lets agents discover a model's MIC on the Hub. Transformers checks slim fixtures on PRs and real checkpoints in run-slow or nightly jobs; Transformers.js has its own contract entry point and baseline, with CI triggers still to define.](docs/assets/mic-overview.svg)

## Try it locally

```sh
pip install "transformers-mic[runtime] @ git+https://github.com/huggingface/transformers-mic.git" \
  "hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git"
hf auth login
tmic run org/model --local-only --fixtures-only --no-history --bill-to ''
```

This drafts a contract for `org/model`, builds its fixture and records baselines
on your machine, asks you to review each step, and uploads nothing. See
[Try it locally](docs/cli.md#try-it-locally) for the flags. `tmic run --mode agent`
gives agents a JSON protocol for the same questions.

Cloud baselines and publication are for maintainers: [CLI guide](docs/cli.md),
[cloud baselines](docs/cloud-baselines.md) (dashboard activation is still pending).

## Why

- **Compatibility is expensive to verify.** Integration tests need real
  checkpoints, large downloads and GPUs, so each PR runs a small part of them.
  Regressions surface late: nightly, at release, or from users.
- **Copies drift.** Test code, documented usage and Hub snippets diverge.
- **Agents guess.** In two [recorded runs](docs/agent-walkthrough/README.md) on
  Florence-2, both agents reconstructed a working path and neither could check
  its output against a reviewed baseline. That shows friction, not a measured MIC
  gain ([assessment](docs/evidence.md#how-agents-use-the-hub-today)).

The first goal is CI, not a Hub feature. The same contracts could later back
compatibility status, tested snippets or agent usage. Run on full checkpoints,
contracts could replace model-specific integration tests in `run-slow`; other
slow tests stay ([CI integration](docs/slim-models-in-transformers-ci.md)).

## The standard

1. **One repository per slim model**, owned by Hugging Face, such as
   [`huggingface-mic/qwen2_5_omni-mic`](https://huggingface.co/huggingface-mic/qwen2_5_omni-mic).
   Creators' checkpoint repos stay unchanged; each framework reviews its own
   expectations and pins a reviewed snapshot.
2. **Fixed layout:**

   ```text
   huggingface-mic/qwen2_5_omni-mic/
     config.json
     model.safetensors
     build.json
     integration.json
     .mic-repo.json                            # separates contract files from weights
     integrations/transformers/
       usage.md
       expectations/cpu-x86_64-bf16/
         mic-qwen2_5_omni-slim.bfloat16.json
     integrations/transformers.js/             # when this framework is supported
       integration.js
       expectations/<platform>/<target>.<variant>.json
   transformers/tests/contracts.lock           # shared fleet dataset commit SHA
   ```

   The fixture pin's `target` names its expectation file ([Hub layout](docs/hub-organization.md)).
3. **A static manifest**, [`integration.json`](example/hub/SmolLM-integration/integration.json):
   checkpoints, each one's fixture, and per framework the entry point, variants,
   unsupported targets, resources and a comparison rule per output.
4. **User code as the contract:** the public API with default settings, printing
   one JSON line. Written as [doc-builder runnable Markdown](example/hub/SmolLM-integration/integrations/transformers/usage.md),
   it is both documentation and test.
5. **Checks beyond "it ran":** generated outputs, logits within tolerance and,
   where training applies, finite loss and gradients and a parameter update.
6. **Reviewed baselines** per platform, target and variant. A behavior-changing
   PR measures candidates, gets them reviewed, publishes a new snapshot and
   updates its lock. Nothing updates approved values automatically.
7. **Shared [fixtures](example/hub/mic-SmolLM-slim):** checkpoints share one only
   with identical processor files, retained config fields and default dtype
   ([recipe](docs/fixtures.md#recipe)).
8. **Everything pinned** in the fleet dataset: contract, fixture, checkpoint and
   dependency versions. Transformers keeps only the dataset commit SHA.

[`run.py`](example/run.py) is the runner prototype; the [full example](example/README.md)
is a runnable SmolLM MIC for Transformers and Transformers.js. The proposed
adoption pilot and the results so far are in [Pilot](docs/pilot.md).

## Documents

- [CLI guide](docs/cli.md): `tmic`, from a checkpoint to a published MIC.
- [Small fixtures](docs/fixtures.md): the shrink recipe and the weight experiments.
- [Pilot](docs/pilot.md): the proposed adoption pilot and the results so far.
- [Evidence](docs/evidence.md): replays, backfills, stability and cost.
- [Full example](example/README.md): the SmolLM MIC and the runner.
- [Technical design](docs/design.md): layout, CI, checks, baselines, runner.
- [Hub layout](docs/hub-organization.md): what lives where on the Hub.
- [Slim models in Transformers CI](docs/slim-models-in-transformers-ci.md): PR and `run-slow` flows.
- [Transformers CI proof](docs/transformers-ci-proof.md): the proof that led to the PR job.
- [Improving contracts](docs/improving-contracts.md): finding and closing coverage gaps.
- [Smoke vs assertions](docs/smoke-vs-assertions.md): which regressions raised, and which only a value check sees.
- [Problems and open questions](docs/problems-and-opened-questions.md): what the pilot must settle.
- [Cloud baselines](docs/cloud-baselines.md): maintainer measurement on CPU and CUDA runners.
- [End-to-end tests](docs/end-to-end-tests.md): every check in this repo, with commands.
- [Agent walkthroughs](docs/agent-walkthrough/README.md): recorded Claude and Codex runs.
- [MIC for Transformers maintainers](docs/slides/index.html): slides on small models, CI usage and dashboard failures.
