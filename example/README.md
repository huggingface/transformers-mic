# Full example: MICs and the runner prototype

A complete, runnable MIC for one usage path, batched greedy text generation, across three SmolLM checkpoints, in Transformers and Transformers.js. It shows every part of the [proposed standard](../README.md#the-proposed-standard) with real files and real outputs. The Hugging Face repositories are not published; their contents are in [`hub/`](hub).

Five more contracts (CLIP, BERT, T5 with UMT5, Whisper, Wav2Vec2) run through the same [runner prototype](#runner-prototype), with fixtures built by one generic recipe.

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
| [`run.py`](run.py) | The shared runner: read a framework's fleet lock, expand targets (each fixture once, then each checkpoint), run every contract in one process, compare with that framework's reviewed baselines and per-platform tolerances |
| [`build_fixtures.py`](build_fixtures.py) | Builds fixtures with one recipe for every model, enforces the fixture key, and runs the input-sensitivity and numeric-health gates |
| [`perturb.py`](perturb.py) | The input-sensitivity gate: perturbations defined per input name |
| [`patches.py`](patches.py) | Local workarounds for known Transformers bugs, named in the fleet lock's `local_patches` and recorded in each result |
| [`replay.py`](replay.py) | Replays a historical regression: runs `run.py` unchanged on the bad and fixed Transformers builds |
| [`schemas/`](schemas) | JSON schemas for the manifest, fleet lock, fixture build, baseline, tolerances, and result record, validated on every load |

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
$PY example/run.py --framework transformers.js --variant webgpu                    # WebGPU baselines
$PY example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu  # tokens vs CPU golden
$PY example/run.py --framework transformers --target SmolLM-slim --record
$PY example/run.py --framework transformers --fixtures-only      # what a framework PR runs
$PY example/build_fixtures.py --fixture whisper-tiny-slim        # rebuild one fixture
$PY example/replay.py d7ba937671 --contract T5                    # UMT5 #49135, bad vs fixed build
```

The new fixtures are not committed (only their `README.md` and `build.json`): build them once with `$PY example/build_fixtures.py`. The fleet lock pins each fixture's file hashes, and the runner refuses a fixture whose files differ. Rebuilding on the same machine gave identical bytes, including the trained Whisper fixture.

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
- **CPU golden outputs check WebGPU.** The Transformers.js variants `webgpu`, `webgpu-fp16`, and `webgpu-q4f16` run the same contract through onnxruntime-node's WebGPU backend (1.30, Apple M5). In fp32, WebGPU matched the CPU baseline: identical tokens, logits within 4e-5, deterministic over three runs. In fp16 and q4f16, every logit came back as exactly 0 and generation stopped after one token, with no error. CPU fp16 was fine (logits within 0.012 of fp32). The same happens in Chrome 154, where the GPU offers `shader-f16`, and only for these `torch.onnx`-exported fp16 graphs: the onnxruntime-genai fp16 export of SmolLM2 (fused attention and normalization) is correct on WebGPU. So the fault is in an onnxruntime WebGPU fp16 kernel for an op only the unfused export uses ([repro](../experiments/webgpu_browser/README.md)). `--golden cpu` catches it from tokens alone; WebGPU fp32 has its own reviewed baselines, and no fp16 baseline was recorded, since that would approve the bug. Each record also carries load, generate, and forward timings, which are kept but not compared ([measurements](../experiments/results/webgpu_golden.json)).
- **The contract can be wrong too.** A first version padded the training batch on the left. Each row's first token was then predicted from a fully masked padding position, where eager and SDPA disagree, so their losses differed by 0.2 on a float32 fixture. Right padding for training fixed it. The eager variant is what exposed the mistake.

## Runner prototype

Six contracts, one per usage path, run through `run.py` on CPU (Apple M5, Transformers `4d536e1f`, torch 2.11.0):

| Contract | Usage path | Fixture (weights, why) | Checkpoints |
| --- | --- | --- | --- |
| SmolLM | batched greedy generation | 2 fixtures, overfit (earlier build script) | 135M, 360M, 1.7B |
| CLIP | zero-shot image-caption matching | random: weakest input move 4.9e-3 | `openai/clip-vit-base-patch32` |
| BERT | fill-mask | random: weakest 2.8e-3 (padding mask) | `google-bert/bert-base-uncased` |
| T5 | text-to-text generation | T5 and UMT5 fixtures, random: weakest 1.3 and 3.1 | `google-t5/t5-small`; `google/umt5-small` unsupported (no `model_type`, so Auto classes cannot load it) |
| Whisper | English transcription with `generate` | **overfit**, 200 steps, 24 s: random weights moved logits by 8.6e-5 when the audio was reversed | `openai/whisper-tiny` |
| Wav2Vec2 | CTC transcription | random: weakest 0.60; amplitude blind spot accepted with a reason | `facebook/wav2vec2-base-960h`, with a local workaround, see below |

**One recipe for every model.** `build_fixtures.py` has no per-family code. It shrinks the real config with the generic rule from the [architecture sweep](../experiments/sweep.py), copies every non-weight file unchanged, and runs the contract's own runnable markdown against the random fixture while recording the arguments of each model call by name. The gate then perturbs those inputs by name ([`perturb.py`](perturb.py)): random weights are kept if every perturbation moves the logits by at least 10× the comparison tolerance (1e-3), and the fixture is overfit on the contract's own training call otherwise. Numeric health (float32, bfloat16, float16: finite outputs, loss, gradients) is checked on the saved fixture. The fixture's config differs from the real one only in its sizes (dropout, LayerDrop, and SpecAugment stay on), and its weights file has the real checkpoint's tensor names: a parameter the real checkpoint lacks is dropped from the fixture too, and the build checks that loading reports the same missing parameters. Of the six real checkpoints, only `facebook/wav2vec2-base-960h` lacks one (`masked_spec_embed`). On these families the gate made the same choice the synthetic-fault study supports: random weights everywhere except Whisper.

**Batching.** Runnable-markdown contracts execute in the runner's process, so imports are paid once:

| Run | Batched | One process per target (`--isolate`) |
| --- | ---: | ---: |
| Fixtures only, 8 targets, 6 contracts (median of 3) | **4.9 s** | 27.4 s |
| Everything, 17 targets including SmolLM-1.7B | 73 s | 90 s |

With the Wav2Vec2 workaround, both variants give 16 passes and 1 unsupported target (`google/umt5-small`).

Results were identical in both modes. With every model loaded in one process, SmolLM-1.7B took 51 s instead of 27 s, probably from memory pressure, so large real checkpoints may still warrant their own process.

**Replaying a regression from recorded inputs.** `replay.py` runs the unchanged runner on the build before and after a fix, against the reviewed fixtures and baselines. UMT5's non-causal decoder under SDPA ([#49135](https://github.com/huggingface/transformers/pull/49135)) is caught on the UMT5 fixture under the default runtime (teacher-forced log-probs off by 1.22, loss by 0.20) and passes on the fixed build. Under eager attention, and on T5, both builds are clean, as in the [earlier replay](../docs/evidence.md#historical-regression-replay). The fixture has random weights. The T5 batch-decode regression ([#47793](https://github.com/huggingface/transformers/pull/47793)) is not caught: it affected `tokenizer.decode()` on a batch, which this contract does not call.

**Baselines must reproduce.** `--record` reruns each target in a fresh process and refuses outputs that differ, and never records non-finite outputs or a failed contract assertion (`all_true`, `true`). Recording these contracts found two problems:

- **The contracts' own seeding was wrong.** `torch.manual_seed` does not seed NumPy, which Wav2Vec2's SpecAugment uses for its masks, so the real checkpoint's training loss varied from 73 to 93 between runs. The contracts now call `transformers.set_seed(0)` before loading.
- **A fragile assertion.** With LayerDrop on (0.1 in Wav2Vec2's config), a training step can skip any encoder layer; under eager attention the seeded step skipped layer 0, so "gradients reach `encoder.layers.0`" failed. The contract now checks the first convolution and `lm_head`, which are always on the path.
- **A Transformers bug.** Even then, `facebook/wav2vec2-base-960h` gave a NaN or garbage loss. `wav2vec2.masked_spec_embed` is missing from the checkpoint and is left uninitialized on load (values up to 3e35, sometimes NaN, different each load): `_init_weights` has no branch for it, here and in `wav2vec2_conformer`, `data2vec`, `wavlm`, `unispeech`, `unispeech_sat`, and `sew` (HuBERT, SEW-D, SpeechT5, and Wav2Vec2-BERT do initialize it). Any fine-tuning of these checkpoints with SpecAugment on, the default, feeds that garbage into the encoder. Reported as [#49385](https://github.com/huggingface/transformers/issues/49385). Until it is fixed, the Transformers fleet lock lists a local workaround under `local_patches`: [`patches.py`](patches.py) adds the missing `_init_weights` branch (`init.uniform_`, as HuBERT does), the runner applies it in every contract process, and each result records it in its environment. With it, the real checkpoint's baselines reproduce across processes and were approved; they must be re-recorded once the fix lands. The first version of the fixture could not see the bug: it saved the tensor, and the recipe disabled SpecAugment. Now the fixture omits the tensor like the real checkpoint and keeps SpecAugment on, and with the patch removed it fails on every run in both variants (training loss off by 8.1 and 6.8), while the real checkpoint fails differently each run. (Without a workaround, the fleet lock's `known_failures` reports such a target as `xfail` and fails once it passes again.)

**Per-platform tolerances.** `expectations/<contract>/<platform>/tolerances.json` overrides the manifest's tolerances for one platform, with a required reason. None is recorded yet: GPU and bfloat16 tolerances need the calibration runs listed in [next steps](../docs/next-steps.md).

## Not real yet

- The `hf-internal-testing` repositories are unpublished; the fleet locks point at the folders in [`hub/`](hub) (`"revision": "unpublished"`).
- Contracts run in the runner's own process, with no isolation beyond `--isolate` (one process per target). Selecting contracts from the files a framework PR changes is not implemented.
- Fixtures built by `build_fixtures.py` are reproducible on one machine; overfit fixtures may not be byte-identical elsewhere, so they must be published and pinned rather than rebuilt by each CI job.
- Variants need a second, hidden `from_pretrained` call in the runnable block. A doc-builder hook for passing load arguments to a block would remove it.
- Baselines here are CPU, plus WebGPU fp32 for Transformers.js. A CUDA GPU job would need its own reviewed baselines: on an A10G, CPU-recorded logits held for text models but not for convolutional ones, and lower precision changed outputs ([evidence](../docs/evidence.md#on-gpu)).

## What this example offers an agent

The manifest, runnable usage example, fleet lock, and reviewed baselines demonstrate the information an agent could use to choose a target, reproduce a tested path, and check fixed inputs. The [walkthroughs](../docs/evidence.md#how-agents-use-the-hub-today) show why that information is useful: agents inferred environments, reconciled conflicting loader guidance, and accepted plausible output without a reviewed reference.

The local example does not yet provide discovery from a Hub model page, a published last-passing record, or a complete environment-bootstrap service. Broad manifest requirements such as `transformers>=5` are not a compatibility guarantee; an agent needs the framework build and full environment associated with the passing record. The [design](../docs/design.md#agent-discovery-and-reproduction) describes that proposed connection, and the [controlled comparison](../docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic) is how to measure its benefit.

A passing slim check validates the fixture's path and overfit outputs. A real-checkpoint baseline validates agreement on that checkpoint's fixed reference inputs and runtime; it does not guarantee semantic correctness for an arbitrary prompt. This native SmolLM example also does not cover the custom-code loader workaround needed for the Microsoft Florence walkthrough.
