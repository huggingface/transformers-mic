# The MIC standard

What a Model Integration Contract is for, and what each one contains. Start with the [README](../README.md) for a summary and a local quick start.

## Why

- **Compatibility is expensive to verify.** Integration tests need real
  checkpoints, large downloads and GPUs, so each PR runs a small part of them.
  Regressions surface late: nightly, at release, or from users.
- **Copies drift.** Test code, documented usage and Hub snippets diverge.
- **Agents guess.** In two [recorded runs](agent-walkthrough/README.md) on
  Florence-2, both agents reconstructed a working path and neither could check
  its output against a reviewed baseline. That shows friction, not a measured MIC
  gain ([assessment](evidence.md#how-agents-use-the-hub-today)).

The first goal is CI, not a Hub feature. The same contracts could later back
compatibility status, tested snippets or agent usage. Run on full checkpoints,
contracts could replace model-specific integration tests in `run-slow`; other
slow tests stay ([CI integration](slim-models-in-transformers-ci.md)).

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

   The fixture pin's `target` names its expectation file ([Hub layout](hub-organization.md)).
3. **A static manifest**, [`integration.json`](../example/hub/SmolLM-integration/integration.json):
   checkpoints, each one's fixture, and per framework the entry point, variants,
   unsupported targets, resources and a comparison rule per output.
4. **User code as the contract:** the public API with default settings, printing
   one JSON line. Written as [doc-builder runnable Markdown](../example/hub/SmolLM-integration/integrations/transformers/usage.md),
   it is both documentation and test.
5. **Checks beyond "it ran":** generated outputs, logits within tolerance and,
   where training applies, finite loss and gradients and a parameter update.
6. **Reviewed baselines** per platform, target and variant. A behavior-changing
   PR measures candidates, gets them reviewed, publishes a new snapshot and
   updates its lock. Nothing updates approved values automatically.
7. **Shared [fixtures](../example/hub/mic-SmolLM-slim):** checkpoints share one only
   with identical processor files, retained config fields and default dtype
   ([recipe](fixtures.md#recipe)).
8. **Everything pinned** in the fleet dataset: contract, fixture, checkpoint and
   dependency versions. Transformers keeps only the dataset commit SHA.

[`run.py`](../example/run.py) is the runner prototype; the [full example](../example/README.md)
is a runnable SmolLM MIC for Transformers and Transformers.js. The proposed
adoption pilot and the results so far are in [Pilot](pilot.md).
