# Full example: MICs and the runner prototype

These source fixtures preserve the original split-repo evidence. New workspaces and published models use [one umbrella repository](../docs/hub-organization.md), with framework expectations beside usage. `publish_mic.py` publishes that layout atomically; the old publishers remain for historical workspaces.

The [transformers-mic CLI](../docs/cli.md) packages this backend and catalog for
model workspaces, HF-assisted drafts, fixture builds and reviewed expectations.

A runnable MIC for batched greedy generation on three SmolLM checkpoints, in Transformers and Transformers.js, covering every part of the [proposed standard](../README.md#the-proposed-standard). Published [fixtures](https://huggingface.co/hf-internal-testing?search_models=proto-) and [contracts](https://huggingface.co/hf-internal-testing?search_models=-integration) have their sources in [`hub/`](hub). Six more contracts (CLIP, BERT, T5/UMT5, Whisper, Wav2Vec2, MusicGen Melody) use the same [runner](#runner-prototype).

## Where each piece lives

| Piece ([owners](../docs/design.md#ownership)) | Here |
| --- | --- |
| `HuggingFaceTB/SmolLM-135M`, `-360M`, `-1.7B` (creator) | pinned in the fleet locks |
| [`SmolLM-integration`](https://huggingface.co/hf-internal-testing/SmolLM-integration): contract and manifest | [`hub/SmolLM-integration/`](hub/SmolLM-integration) |
| [`mic-SmolLM-slim`](https://huggingface.co/hf-internal-testing/mic-SmolLM-slim) (135M, 360M), [`mic-SmolLM-1.7B-slim`](https://huggingface.co/hf-internal-testing/mic-SmolLM-1.7B-slim) (1.7B) | [`hub/`](hub) |
| `onnx-community/SmolLM-135M-ONNX`, `-360M-ONNX`, loaded by Transformers.js | pinned in its lock |
| Fleet lock and expected results per framework | [`transformers/`](transformers/tests/contracts), [`transformers.js/`](transformers.js/tests/contracts) |

| Tool | What it does |
| --- | --- |
| [`run.py`](run.py) | Runner: lock → targets (fixtures, then checkpoints) → one process → compare with baselines |
| [`build_fixtures.py`](build_fixtures.py) | Fixture recipe, fixture key, gates; `--out --keep-random` builds a random-weight twin for experiments |
| [`perturb.py`](perturb.py) | Input-sensitivity gate |
| [`patches.py`](patches.py) | `local_patches` workarounds for known Transformers bugs |
| [`replay.py`](replay.py) | `run.py` on a regression's bad and fixed builds |
| [`select_contracts.py`](select_contracts.py) | Contracts a change can affect, from each fixture's model types (sub-configs included) and processor classes |
| [`review_candidates.py`](review_candidates.py) | Diff `--record` candidates against approved baselines; `--approve` accepts unchanged and new |
| [`publish_contracts.py`](publish_contracts.py), [`publish_fixtures.py`](publish_fixtures.py) | Upload to the Hub and pin the new revision and file hashes in the locks |
| [`report.py`](report.py) | Markdown job summary of a run, with repro commands |
| [`schemas/`](schemas) | JSON schemas, validated on load |

## The parts

**One contract per usage path.** [`integration.json`](hub/SmolLM-integration/integration.json) lists checkpoints, their fixtures, and per framework: entry point, artifacts, variants, unsupported targets with reasons. Adding SmolLM-360M took a manifest entry and a lock pin, no code or fixture.

**Shared fixtures, by [fixture key](../docs/design.md#grouping-checkpoints):**

- SmolLM-135M, -360M: [`mic-SmolLM-slim`](hub/mic-SmolLM-slim), key `a186080c7a70`, 3.2M parameters, bfloat16 (6.1 MB), overfit by reviewed choice (80 steps, `target_loss` 0.3; [why](../docs/evidence.md#bfloat16-on-cpu)).
- SmolLM-1.7B: different tokenizer files and full multi-head attention (32/32 heads vs 9/3, 15/5). [`mic-SmolLM-1.7B-slim`](hub/mic-SmolLM-1.7B-slim), key `8a9d62b3b3b2`, 4/4 heads, float32 (13 MB), overfit by an earlier build script.

**Runnable markdown.** [`usage.md`](hub/SmolLM-integration/integrations/transformers/usage.md) ([doc-builder blocks](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md)) renders as a plain SmolLM-135M example (generate, one training step), passes `pytest -p doc_builder.pytest_plugin usage.md`, and is what `run.py` runs, with each pinned target passed as a local path. Transformers.js: [`integration.js`](hub/SmolLM-integration/integrations/transformers.js/integration.js), a plain script on the `onnx-community` conversions, no training probe.

**Baselines and locks per framework:** one reviewed file per target, platform, and variant ([Transformers](transformers/tests/contracts/expectations/SmolLM/cpu-arm64), [Transformers.js](transformers.js/tests/contracts/expectations/SmolLM/cpu-arm64)); `--record` writes `<name>.candidate.json`, approval is a rename. Each [fleet lock](transformers/tests/contracts/fleet.lock) pins contract, fixtures, checkpoints, and dependencies; the [Transformers.js lock](transformers.js/tests/contracts/fleet.lock) pins the same contract independently, plus two ONNX conversions.

## Run it

Use an isolated Python 3.10 environment with the [lock's dependencies](transformers/tests/contracts/fleet.lock) and pinned Transformers build `8625785e`. Linux CI uses [`install_pinned.sh`](../.github/scripts/install_pinned.sh); it assumes Linux/Bash and is not a Mac setup script. A different installed build may change outputs and will be reported in `environment_differs`.

```sh
PY=.venv/bin/python                            # prepared environment described above
$PY -m pip install git+https://github.com/huggingface/doc-builder jsonschema  # runnable blocks are on main
(cd example/hub/SmolLM-integration/integrations/transformers.js && npm install)
$PY example/run.py --framework transformers --contract SmolLM # 2 fixtures + 3 checkpoints
$PY example/run.py --framework transformers --contract SmolLM --variant eager
$PY example/run.py --framework transformers.js
$PY example/run.py --framework transformers.js --variant webgpu                    # WebGPU baselines
$PY example/run.py --framework transformers.js --variant webgpu-fp16 --golden cpu  # tokens vs CPU golden
$PY example/run.py --framework transformers --target mic-SmolLM-slim --record
$PY example/run.py --framework transformers --fixtures-only      # what a framework PR runs
$PY example/build_fixtures.py --fixture mic-whisper-tiny-slim        # rebuild one fixture
$PY example/build_fixtures.py --fixture mic-whisper-tiny-slim --out /tmp/twin --keep-random  # random-weight twin
$PY example/replay.py d7ba937671 --contract T5                    # UMT5 #49135, bad vs fixed build
```

Only fixtures' `README.md` and `build.json` are committed (`--local-fixtures` needs `build_fixtures.py` once); the runner refuses files whose hashes differ from the lock.

Results on CPU (Apple M5, Transformers `5.19.0.dev0`, torch 2.11.0, `@huggingface/transformers` 4.3.0, doc-builder `c69e96c`):

| Framework | Target | Runtime | Generated (default variant) | Default | Eager | Seconds |
| --- | --- | --- | --- | --- | --- | ---: |
| Transformers | mic-SmolLM-slim | bfloat16 | " Paris Paris Paris Paris", " Berlin Berlin Berlin of", " Rome Rome London London", " London London London London" | pass | pass | 2.8 |
| Transformers | mic-SmolLM-1.7B-slim | float32 | " Paris", " Berlin", " Rome", " London" | pass | pass | 2.6 |
| Transformers | SmolLM-135M | bfloat16 | " Paris, the capital", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | pass | 5.8 |
| Transformers | SmolLM-360M | bfloat16 | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | pass | 11.0 |
| Transformers | SmolLM-1.7B | float32 | " Paris.\nThe", " Berlin.\nGermany", " Rome.\nThe", " London.\nThe" | pass | pass | 33.4 |
| Transformers.js | SmolLM-135M | fp32, ONNX | " Paris. It is", " Berlin.\nThe", " Rome. It is", " London.\nThe" | pass | — | 1.7 |
| Transformers.js | SmolLM-360M | fp32, ONNX | " Paris.\nThe", " Berlin.\nThe", " Rome.\nThe", " London.\nThe" | pass | — | 2.1 |
| Transformers.js | fixtures, SmolLM-1.7B | — | — | unsupported, with reasons | — | — |

A framework PR runs two fixture checks (~5 s) instead of three checkpoint checks (~50 s, 8.2 GB of weights).

- **Variants and frameworks need own baselines.** SmolLM-135M under eager says " Paris. It is", not " Paris, the capital"; Transformers.js matches Transformers' 135M on 3 of 4 prompts.
- **CPU golden outputs check WebGPU:** fp32 matches within 4e-5; fp16 and q4f16 return all-zero logits, caught from tokens by `--golden cpu` ([evidence](../docs/evidence.md), [repro](../experiments/webgpu_browser/README.md), [numbers](../experiments/results/webgpu_golden.json)). Only fp32 has reviewed baselines: an fp16 one would approve the bug.
- **Contracts can be wrong too.** A left-padded training batch predicted first tokens from fully masked positions, where eager and SDPA disagree (losses 0.2 apart); right padding fixed it.

## Runner prototype

Eight contracts through `run.py` on Apple M5, torch 2.11.0: the original six used Transformers `4d536e1f`; MusicGen Melody and checkpoint-only Whisper speech used `8625785e` (same tree).

| Contract | Usage path | Fixture (weights, why) | Checkpoints |
| --- | --- | --- | --- |
| SmolLM | batched greedy generation | 2 fixtures, overfit: mic-SmolLM-slim by reviewed choice (random weights repeat one answer), 1.7B by an earlier script | 135M, 360M, 1.7B |
| CLIP | zero-shot image-caption matching | random: weakest input move 4.9e-3 | `openai/clip-vit-base-patch32` |
| BERT | fill-mask | random: weakest 2.8e-3 (padding mask) | `google-bert/bert-base-uncased` |
| T5 | text-to-text generation | T5 and UMT5 fixtures, random: weakest 1.3 and 3.1 | `google-t5/t5-small`; `google/umt5-small` unsupported (no `model_type`; Auto classes cannot load it) |
| Whisper | English transcription with `generate` | **overfit**, 200 steps, 24 s: random weights moved logits 8.6e-5 on reversed audio | `openai/whisper-tiny` |
| Wav2Vec2 | CTC transcription | random: weakest 0.60; amplitude blind spot accepted with a reason | `facebook/wav2vec2-base-960h`, with a local workaround (below) |
| MusicgenMelody | text- and melody-conditioned generation with guidance | **overfit** by reviewed choice, 60 steps: random weights pass the gate (weakest 0.018) but generate the same codes for every prompt, melody or not; EnCodec widths set by `config_overrides` (59M → 3.5M parameters) | `facebook/musicgen-melody` |
| WhisperSpeech | greedy speech transcription versus an encoder-decoder assistant | checkpoint-only: both slim weight kinds miss #48108 on this frozen LibriSpeech sample | `openai/whisper-tiny` |

`build_fixtures.py` applies one [recipe](../docs/fixtures.md#recipe), with reviewed `config_overrides` for missed fields. Contract calls feed the [`perturb.py`](perturb.py) gate and overfit step. MusicGen Melody has fixture/checkpoint baselines on `cpu-arm64` and `cuda` in three variants, `cpu-x86_64` in two. Float32 score-argmax codes agree across these three platform classes; only the real CUDA checkpoint changes codes in bfloat16. Score argmaxes include delay-mask positions and are not exactly the codes sent to EnCodec ([controls](../docs/evidence.md#musicgen-melody-conditioning-controls)).

**Batching** (runnable-markdown contracts share the runner's process):

| Run | Batched | One process per target (`--isolate`) |
| --- | ---: | ---: |
| Fixtures only, 8 targets, 6 contracts (median of 3, before MusicGen Melody) | **4.9 s** | 27.4 s |
| Everything, 17 targets including SmolLM-1.7B | 73 s | 90 s |

Both: 16 pass, 1 unsupported (`google/umt5-small`). Batched, SmolLM-1.7B took 51 s vs 27 s (likely memory pressure), so large checkpoints may want their own process.

**Replay** (`replay.py`): UMT5 [#49135](https://github.com/huggingface/transformers/pull/49135) caught on the random fixture (log-probs off 1.22, loss 0.20), fixed build passes; T5 batch decode [#47793](https://github.com/huggingface/transformers/pull/47793) missed until the call was [added](../docs/improving-contracts.md). MusicGen Melody [#48679](https://github.com/huggingface/transformers/pull/48679) (conditioning dropped at generation step 0) is caught from its introducing commit by the overfit fixture and by its random twin ([overfit or random](../docs/evidence.md#overfit-or-random-weights-on-real-regressions)).

**Reproducible baselines** (`--record` reruns in a fresh process) found:

- **Wrong seeding:** `torch.manual_seed` misses NumPy (SpecAugment); real Wav2Vec2 loss varied 73–93. Now `transformers.set_seed(0)` before loading.
- **Fragile assertion:** LayerDrop skipped layer 0 under eager; gradients are now checked on the first convolution and `lm_head`.
- **Transformers bug [#49385](https://github.com/huggingface/transformers/issues/49385):** `masked_spec_embed`, absent from `facebook/wav2vec2-base-960h`, is left uninitialized (up to 3e35, sometimes NaN) in `wav2vec2`, `wav2vec2_conformer`, `data2vec`, `wavlm`, `unispeech`, `unispeech_sat`, `sew`, so SpecAugment fine-tuning reads garbage. [`patches.py`](patches.py) adds the missing init (`init.uniform_`, like HuBERT) via `local_patches`. The fixture omits the tensor and keeps SpecAugment, so unpatched it fails every run (loss off 8.1, 6.8).

### On GPU

Original six-contract A10G study, before MusicGen Melody (torch 2.14.0+cu130, Transformers `8625785e`), `contracts` mode of [`gpu-checks.yml`](../.github/workflows/gpu-checks.yml) ([run](https://github.com/huggingface/model-integration-contracts/actions/runs/37615109270), [rerun](https://github.com/huggingface/model-integration-contracts/actions/runs/37621572296), [numbers](../experiments/results/runner.json)):

- **`cuda` baselines hold at `1e-4`:** 16 targets × 3 variants (default, eager, bfloat16), reproduced in fresh processes and over two more runs.
- **GPU float32 vs CPU baselines:** tokens and labels identical on 15/16 (SmolLM-135M: " Paris, which is" vs " Paris, the capital"); logits within 4e-6 on text fixtures, 4e-4–2e-3 on convolutional models (TF32); dropout losses differ.
- **bfloat16 vs GPU float32:** no discrete output changed; logits up to 0.05 on fixtures, 2.3 on real Wav2Vec2.
- So no looser tolerance is needed: all drift above `1e-4` crosses platforms or dtypes. Batching and publish-don't-rebuild: [evidence](../docs/evidence.md#summary).

## Not real yet

- Contract edits: change [`hub/`](hub), test with `run.py --local-contracts`, then [`publish_contracts.py`](publish_contracts.py) pins the new revision in every lock. Transformers.js runs its contract from the local folder (for its `npm` packages), checked against the pin.
- Isolation is only `--isolate` (one process per target).
- Scheduled CI runs in this repo ([results](../docs/evidence.md#shadow-backfill)); in Transformers, only a non-required proof job on the `contracts-proof` branch, which vendors the runner and lock and does not include MusicGen Melody yet ([proof](../docs/transformers-ci-proof.md)).
- `build_fixtures.py` output reproduces on one machine only, so CI downloads the pinned fixtures.
- Variants need a second, hidden `from_pretrained`; a doc-builder load-argument hook would remove it.

## What this example offers an agent

Manifest, example, lock, and baselines let an agent pick a target, reproduce a tested path, and check fixed inputs, which the [walkthroughs](../docs/evidence.md#how-agents-use-the-hub-today) lacked. Missing: [discovery](../docs/design.md#agent-discovery-and-reproduction) from the model page, a published last-passing record with its exact build and environment (`transformers>=5` is not one), and the Florence custom-code workaround. Benefit: [controlled comparison](../docs/agent-walkthrough/README.md#controlled-comparison-with-and-without-a-mic).
