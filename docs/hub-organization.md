# One repository per MIC model

The canonical layout is [`huggingface-mic/<name>-mic`](https://huggingface.co/huggingface-mic),
for example [`qwen2_5_omni-mic`](https://huggingface.co/huggingface-mic/qwen2_5_omni-mic).

```text
qwen2_5_omni-mic/
  config.json
  model.safetensors
  build.json
  integration.json
  .mic-repo.json
  integrations/
    transformers/
      usage.md
      expectations/
        cpu-x86_64-bf16/
          mic-qwen2_5_omni-slim.bfloat16.json
    transformers.js/                         # if supported
      integration.js
      expectations/<platform>/<target>.<variant>.json
```

`tmic create` defaults to the `huggingface-mic` organization; `--namespace` overrides
it. New generated usage and the slim model share a `<name>-mic` Hub identifier.
Catalog reuse creates the same layout, including offline creation from the
bundled templates and reference files.

Each framework's expectations sit beside its usage. `record` writes candidates
there; `review` uses the same directory. Changed values still require manual review.
`publish --execute` assembles usage, the selected slim model and reviewed references
into one Hub commit, then pins both roles. Candidates are excluded. `export` writes
fleet pins; the reviewed reference files remain in the pinned Hub snapshot.
A contract covering several slim models can keep separate fixture pins and publish
the selected model's umbrella snapshot.

`.mic-repo.json` declares the contract file inventory. The runner verifies that
inventory against the contract hash, and verifies weights and processors separately
against `build.json`. Expectations are outside both content inventories. They remain
covered by the immutable Hub commit pin and workspace validation fingerprint.
The fixture pin's `target` field identifies the matching expectation file.

The fixture pin's `target` also appears in `recorded_with.target` inside every
baseline. For `huggingface-mic/bert-mic`, the canonical target is `bert-mic`,
with references such as `bert-mic.bfloat16.json`. Rename the files and their
recorded identities together, then update the fleet's target and immutable model
revision. Numeric expectations, executable contract hashes and fixture hashes
stay unchanged for a target rename. Exports carry scoped known-failure entries
to the matching renamed target.
