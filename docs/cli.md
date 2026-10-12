# transformers-mic CLI

`transformers-mic` packages the existing MIC runner, fixture builder, replay tools,
and reviewed catalog. The `tmic` command creates a working directory for a model,
proposes additional usage paths when needed, and measures expectations there.
It is an alpha implementation: generated plans still need code review and a
working model-specific build recipe. New artifacts use `hf-internal-testing/mic-{name}-slim` and
`mic-{name}-integration`. The nine legacy `proto-*-slim` repositories were moved
to `mic-*-slim`, preserving suffixes and revisions; historical evidence stays as recorded.

## Install and start

From this repository (Python 3.10+):

```sh
python -m pip install -e '.[runtime,dev]'
python -m pip install 'hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git'
hf auth login
tmic create google-bert/bert-base-uncased --out work/bert
cd work/bert
tmic build
tmic record --all-variants
tmic review
tmic review --approve
tmic check --all-variants
tmic publish                         # preview
tmic publish --execute               # upload to the logged-in Hub account
tmic check --all-variants             # check the published pins
tmic export --transformers-checkout /path/to/transformers
```

Use a checkout pinned to the workspace's `fleet.lock` dependency environment for
baseline reproduction. Installing the runtime extra alone does not freeze that
environment. The repository's GPU workflow uses `.github/scripts/install_pinned.sh`.
The PyPI release of doc-builder currently lacks the runnable-markdown collector;
install its Git version as above. The installed wheel bundles the backend and
catalog, so it also works outside this checkout; model weights download from the
immutable Hub pins.

`tmic create --offline` reuses a catalog contract without network requests.
Otherwise, `gh` must be installed and authenticated for GitHub history searches.
`--no-history` skips them; `--history-file failures.json` imports a normalized JSON
list from relore or another source. A failure object can contain `url`, `title`,
`body`, and revision information. Searches are bounded candidates, not a complete
inventory of past failures, and do not count as regression detection.

## When the LLM is used

Catalog reuse requires no inference call. For a new model, or with `create --llm`,
the planner uses **Hugging Face Inference Providers**, default model
**`Qwen/Qwen3.8-27B`**, with `provider="auto"`. Set `--provider` or `--llm-model`
on `create` and `update` to override them. Authentication uses the normal HF token
(`HF_TOKEN` or `hf auth login`) with Inference Providers permission and available
credits. Provider inference may be billed to that account.

The planner receives checkpoint config, the manifest schema, and failure
candidates. It proposes a manifest and runnable usage, stored in `.mic/plan.json`.
Invalid drafts receive validation feedback for one repair attempt; persistent
errors stop creation without executing the draft. It cannot generate expected
numeric values, loosen tolerances, or choose overfit
weights. Review generated `integration.json` and `usage.md` before running them:
syntax/schema checks do not sandbox Python. `create` alone does not execute the
proposed model code; `--build` and `--record` are explicit execution shortcuts.

## Random or overfit?

1. Reuse the existing pinned slim first. A new assertion does not automatically
   need new weights: measure and replay it on that fixture.
2. A fresh random fixture is sufficient if it passes the builder's sensitivity
   and finite-output gates for the proposed usage path.
3. The builder tries bounded overfitting when random fails. Use
   `--readable-output` or `--weights overfit` when meaningful decoded output is
   itself a requirement. `--weights random` forbids a silent overfit fallback.
4. If neither passes, the candidate is rejected. Some usage paths need a real
   checkpoint; declaring checkpoint-only targets is an explicit manifest choice.

`.mic/project.json` records the selected weights and reason; `build.json` contains
the gate evidence. A gate pass establishes usefulness for those probes, **not**
coverage of a past failure. Random and overfit can tie on historical coverage;
readability alone does not prove one detects more bugs. Legacy published fixtures
may predate gate metadata: they can be reused with reviewed baselines, but new
builds must pass the current gates.

## Add paths and prove regressions

```sh
tmic update 'Add a padded batch through the public pipeline API'
tmic record --all-variants
tmic review
tmic check --all-variants
tmic replay FIX_SHA --repo /path/to/transformers --match-output answer_logits
```

Update retains the old source under `.mic/history/` and keeps the fixture pin.
It rejects removal of existing output assertions. Retaining every existing call
is also a code-review requirement; output names alone cannot prove that. Use
`update --retrain` or `build --rebuild` only when the existing fixture is inadequate.
Builds are staged into immutable `.mic/artifacts/` directories and do not overwrite
previous approved expectations. LLM updates currently support single-checkpoint
contracts; grouped contracts can be edited directly and checked with the same CLI.

Replay archives bad/fixed Transformers revisions from a local Git repository,
executes the same contract and frozen baselines on each, and exits successfully
only for a failure on the bad revision followed by a pass on the fix. Without
`--match-output`, a catch may be on any output and needs inspection. No automatic
claim converts search results or sensitivity gates into replay evidence.

`record` runs each case twice before writing `*.candidate.json`. `review --approve`
accepts new and unchanged candidates, including added output fields. Changed or
removed existing values remain pending for manual review; the LLM never approves
them. Review new baselines for meaningful assertions before approval. `check`
uses approved files and pins; `--target` allows focused investigation, but cannot
satisfy the publication gate. Publication requires every declared variant checked
on the current sources and approved expectations. These receipts cover the
platforms actually run, not an untested CPU/GPU or dependency matrix.

## Hub and Transformers updates

`publish` previews the destination after validation. `publish --execute` delegates
to the existing checksum-checked Hub publishers and updates the lock to immutable
Hub revisions. It invalidates local check receipts, so check again before export.
Upload requires write access to the integration/fixture namespaces; catalog
artifacts remain under their existing owners.

`export --contract-name mic-bert` gives the destination a distinct name when
retaining a historical `BERT` contract. Export rejects names that differ only
by case, avoiding baseline-folder collisions on macOS and Windows.

`export` writes a reviewable `tests/contracts/fleet.lock` and expectations into the
provided checkout, preserves other contracts, and removes workspace paths from
pins. It does not commit, open a PR, or merge Transformers changes. CI can run:

```sh
tmic check --lock tests/contracts/fleet.lock --contract BERT --all-variants
```

`tmic status` shows project state. Workspaces contain source, history, candidate
expectations, checks and artifacts; keep them outside versioned source or ignore
`work/`. The current release implements Transformers orchestration; the bundled
prototype still includes Transformers.js contracts, but the CLI does not publish
or export a JavaScript fleet.

## Validation of this alpha

The [GPU package workflow](https://github.com/huggingface/transformers-mic/actions/runs/38227068563)
installed the wheel outside the checkout on an A10G, checked BERT's fixture and
checkpoint in default/eager/bfloat16, recorded fresh-process candidates, reviewed
them, exported the fleet and checked the exported pins. A separate fresh random
fixture passed the build gates and recorded a candidate. Source distributions
and wheels were checked for complete backend/catalog resources and exclusion of
weights, pending candidates and installed JavaScript dependencies.

A live default Qwen request produced a validated BERT draft. Fleet exercise now
covers 26 concrete architectures; the repeated SpeechT5 request is deduplicated.
`auto` uses BERT, T5 and ViT contracts with explicit Auto classes, rather than
building a separate model. The [third GPU pass](https://github.com/huggingface/transformers-mic/actions/runs/38260897058)
validated 17 architectures in default/eager/bfloat16, with fixture-only targets:
BERT, GPT2, T5, ModernBERT, ViT, CLIP, Table Transformer, GOT-OCR2, Whisper,
Wav2Vec2, Llama, Gemma3, Qwen2, Mistral3, LLaVA, SmolVLM and InternVL.
[Per-model results](../experiments/results/prototype-regeneration.json).

Subsequent GPU passes validated GPT-OSS, Qwen2-Audio, Qwen2.5-VL, DETR,
SpeechT5 and Qwen2.5-Omni, bringing the published fleet to **23 architectures**.
Each has immutable public `hf-internal-testing/mic-*-slim` and
`mic-*-integration` pins. GPT-OSS uses a dense slim derived from the quantized
checkpoint's paired MXFP4 blocks/scales; it does not test quantized kernels.

Three architectures remain rejected: CSM's processor passes an unsupported
argument to its audio feature extractor; Gemma3n has nonfinite gradients despite
finite forward/loss; PI0's native action calls execute but its fixture fails the
fixed state/image sensitivity gates. PI0 uses the documented `lerobot/pi0_base`
checkpoint. Numeric tolerances and the sensitivity threshold remain unchanged.
DETR and SpeechT5 explicitly exempt amplitude scaling that their input
normalization cancels; other probes remain required. The original failed
[normalization probes](../experiments/results/prototype-normalization-probes.json)
are retained. Initial fleet creation
skipped history collection; these checks establish fixture behavior, without
historical-regression or full-checkpoint coverage. Some GPU runs used documented,
model-scoped framework patches; the Transformers proof runs without them.

Regenerate or resume the declared fleet using the CLI:

```sh
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase prepare --no-history
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase run --resume
tmic batch --models-file experiments/proto-models.json --out work/fleet --phase run --resume --select whisper
```

`batch` records a stage result for every model and continues after failures.
Resume retains successful builds and repeats recording, review and checks;
`--publish` explicitly adds upload and published-pin checks. CPU expectations
must be recorded and checked before using GPU-generated contracts in CPU CI.
The `mic-published` workflow checks immutable Hub pins on GPU, records/reviews
fresh CPU expectations, checks both platforms and exports passing contracts.
Native CPU recording now runs on Transformers' actual `aws-m8i-l-cache`
runner, with torch 2.11, pinned scientific dependencies and four CPU threads.
The proof explicitly selects checkout `src` and asserts its framework import
path and thread count, because installed-package versions alone do not verify
the source under test. The
[native recording run](https://github.com/huggingface/transformers/actions/runs/38281747657)
checked and exported 22 models in all three variants; Whisper's prototype remains
blocked by the `suppress_tokens` generation error (#49456). New native BF16/FP16
platform references were recorded; no changed candidates were approved.
The 23-model proof fleet retains Whisper as a message-matched expected failure
and skips only Wav2Vec2 loss/gradients for #49385. Its historical six-contract
fleet stays separate. See [current proof scope](transformers-ci-proof.md).
`published_gpu_only=true` focuses `mic-published` on immutable GPU pins when
native CPU validation has already run separately.

Unit tests cover policy failures, draft repair, checkpoint-name inspection,
update preservation, validation invalidation and fleet export. The initial
[package validation report](../experiments/results/mic_cli_validation.json)
remains separate from the architecture fleet study.
