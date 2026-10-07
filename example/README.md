# Full example: a MIC for SmolLM text generation

A complete, runnable MIC for one usage path, batched greedy text generation, across three SmolLM checkpoints, in Transformers and Transformers.js. It shows every part of the [proposed standard](../README.md#the-proposed-standard) with real files and real outputs. The Hugging Face repositories are not published; their contents are in [`hub/`](hub).

## Where each piece lives

| Piece | Owner | Here |
| --- | --- | --- |
| `HuggingFaceTB/SmolLM-135M`, `-360M`, `-1.7B` | Creator | not copied: unchanged, pinned in the fleet locks |
| `hf-internal-testing/SmolLM-integration`: contract and manifest | Hugging Face | [`hub/SmolLM-integration/`](hub/SmolLM-integration) |
| `hf-internal-testing/SmolLM-slim`: fixture for 135M and 360M | Hugging Face | [`hub/SmolLM-slim/`](hub/SmolLM-slim) |
| `hf-internal-testing/SmolLM-1.7B-slim`: fixture for 1.7B | Hugging Face | [`hub/SmolLM-1.7B-slim/`](hub/SmolLM-1.7B-slim) |
| `onnx-community/SmolLM-135M-ONNX`, `-360M-ONNX` | Hugging Face | not copied: the conversions Transformers.js loads |
| Transformers: fleet lock and expected results | Transformers maintainers | [`transformers/tests/contracts/`](transformers/tests/contracts) |
| Transformers.js: fleet lock and expected results | Transformers.js maintainers | [`transformers.js/tests/contracts/`](transformers.js/tests/contracts) |

Tooling:

| File | What it is |
| --- | --- |
| [`run.py`](run.py) | Sketch of the shared runner: read a framework's fleet lock, expand targets (each fixture once, then each checkpoint), run the contract in its own process, compare with that framework's reviewed baseline |
| [`build_fixtures.py`](build_fixtures.py) | Builds the fixtures and enforces the fixture key |

## The parts

**One contract per usage path.** [`integration.json`](hub/SmolLM-integration/integration.json) lists the three checkpoints, the fixture that covers each, and per framework the entry point, the artifacts, the variants, and the unsupported targets with a reason. Adding SmolLM-360M to the contract took a manifest entry and a pin in each fleet lock (plus its ONNX conversion for Transformers.js); it needed no new contract code and no new fixture.

**Shared fixtures, by fixture key.** A fixture covers every checkpoint whose processor files are byte-identical and whose config matches wherever shrinking keeps it. SmolLM-135M and -360M share key `5fe5094035d3` and one fixture. SmolLM-1.7B ships a different `tokenizer.json` and `tokenizer_config.json` and uses full multi-head attention (32 query and 32 key/value heads, against 9/3 and 15/5), so it gets its own fixture (key `8a9d62b3b3b2`, 4/4 heads). [`build_fixtures.py`](build_fixtures.py) fails if a checkpoint assigned to a fixture has a different key. Each fixture is 3.2M parameters and 13 MB, overfit to the contract's prompts, with the tokenizer files copied unchanged.

**The contract is runnable markdown.** [`usage.md`](hub/SmolLM-integration/integrations/transformers/usage.md) uses [doc-builder runnable blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md):

- doc-builder renders it as a plain SmolLM-135M usage example (generation, then one training step). The runner plumbing is marked `# doc-builder: hide` and does not appear;
- doc-builder's pytest plugin runs it as written (`pytest -p doc_builder.pytest_plugin usage.md` passes);
- `run.py` collects the same blocks with doc-builder's own collector and runs them against each fixture and checkpoint. It passes the pinned revision as a local path, so the visible code stays `from_pretrained(model_id)`.

The Transformers.js contract, [`integration.js`](hub/SmolLM-integration/integrations/transformers.js/integration.js), is a plain script on the `onnx-community` conversions, with no training probe.

**Expected results live with each framework.** [`transformers/tests/contracts/expectations/`](transformers/tests/contracts/expectations/SmolLM/cpu) has one reviewed file per target (fixture or checkpoint), platform (`cpu`), and variant (`.eager`), each with the revision, environment, and runtime it was recorded with. Transformers.js has its own files under [`transformers.js/tests/contracts/expectations/`](transformers.js/tests/contracts/expectations/SmolLM/cpu). A Transformers PR that changes outputs updates its files in the same PR, and nothing on the Hub changes. `run.py --record` writes `<name>.candidate.json` and never overwrites an approved baseline; a reviewer approves by renaming.

**Fleet locks.** [`transformers/tests/contracts/fleet.lock`](transformers/tests/contracts/fleet.lock) pins the contract, both fixtures, the three checkpoints, and the dependency versions. [`transformers.js/tests/contracts/fleet.lock`](transformers.js/tests/contracts/fleet.lock) pins the same contract independently, with the two ONNX conversions it uses.

## Run it

```sh
PY=~/Dev/transformers/.venv/bin/python          # Python with transformers, torch, accelerate
pip install git+https://github.com/huggingface/doc-builder   # runnable blocks are on main
(cd example/hub/SmolLM-integration/integrations/transformers.js && npm install)
$PY example/run.py --framework transformers                  # 2 fixtures + 3 checkpoints
$PY example/run.py --framework transformers --variant eager
$PY example/run.py --framework transformers.js
$PY example/run.py --framework transformers --target SmolLM-slim --record
```

Results on CPU (Apple M5, Transformers `5.19.0.dev0`, torch 2.11.0, `@huggingface/transformers` 4.3.0, doc-builder `c69e96c`):

| Framework | Target | Runtime | Generated (default variant) | Default | Eager | Seconds |
| --- | --- | --- | --- | --- | --- | ---: |
| Transformers | SmolLM-slim | float32 | " Paris", " Berlin", " Rome", " London" | pass | pass | 2.8 |
| Transformers | SmolLM-1.7B-slim | float32 | same | pass | pass | 2.6 |
| Transformers | SmolLM-135M | bfloat16 | " Paris, the capital", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | pass | 5.8 |
| Transformers | SmolLM-360M | bfloat16 | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | pass | 11.0 |
| Transformers | SmolLM-1.7B | float32 | " Paris.\nThe", " Berlin.\nGermany", " Rome.\nThe", " London.\nThe" | pass | pass | 33.4 |
| Transformers.js | SmolLM-135M | fp32, ONNX | " Paris. It is", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | — | 1.7 |
| Transformers.js | SmolLM-360M | fp32, ONNX | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | — | 2.1 |
| Transformers.js | fixtures, SmolLM-1.7B | — | — | unsupported, with reasons | — | — |

A framework PR runs two fixture checks (about 5 s here) instead of three checkpoint checks (about 50 s and 8.2 GB of weights).

What building the example showed:

- **Variants need their own baselines.** SmolLM-135M loads in bfloat16 by default. Under eager attention its greedy continuation for France changes (" Paris. It is" instead of " Paris, the capital"). In float32 the fixtures and SmolLM-1.7B give the same results for both.
- **Frameworks need their own baselines.** Transformers.js runs the ONNX conversion in float32 and agrees with Transformers' bfloat16 run on three prompts out of four for 135M.
- **The contract can be wrong too.** A first version padded the training batch on the left. Each row's first token was then predicted from a fully masked padding position, where eager and SDPA disagree, so their losses differed by 0.2 on a float32 fixture. Right padding for training fixed it. The eager variant is what exposed the mistake.

## Not real yet

- The `hf-internal-testing` repositories are unpublished; the fleet locks point at the folders in [`hub/`](hub) (`"revision": "unpublished"`).
- `run.py` starts a new process per target. The shared runner should batch targets per process, because imports dominate check cost ([evidence](../docs/evidence.md)). It should also add the manifest and result schemas, isolation, and the input-sensitivity gate ([design](../docs/design.md)).
- Variants need a second, hidden `from_pretrained` call in the runnable block. A doc-builder hook for passing load arguments to a block would remove it.
- Baselines here are CPU only. A GPU job would need its own reviewed baselines: on an A10G, CPU-recorded logits held for text models but not for convolutional ones, and lower precision changed outputs ([evidence](../docs/evidence.md#on-gpu)).
