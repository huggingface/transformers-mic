# Transformers CI proof

Status: 2026-10-12. The expanded proof runs on `contracts-proof`; it is not
adopted for unrelated Transformers PRs. The package work was squashed and pushed
to transformers-mic `main` as `03500b0`.

## Results and tested source

| Check | Executed | Result |
| --- | ---: | --- |
| [Proof-branch CPU push](https://github.com/huggingface/transformers/actions/runs/38298088861), source `11c4611c` | 77 | **73 pass, four message-matched Whisper xfails** |
| [PR synthetic merge](https://github.com/huggingface/transformers/actions/runs/38298093605), source `1111c102` | 77 | **73 pass, four Whisper failures**; newer upstream source changes generation behavior |
| [Native CPU recording](https://github.com/huggingface/transformers/actions/runs/38281747657) | 66 checked variants | 22 models checked/exported; prototype Whisper blocked at generation |
| [Whisper candidates on fixed PR source](https://github.com/huggingface/transformers/actions/runs/38302181914), source `1111c102` | 3 variants, each repeated in a fresh process | **3 recorded, unpatched**; default/eager references still differ and native BF16 is new; no approval/export |
| Published GPU pins | 23 architectures, three variants each | Checked after publication; some model-scoped patches, as recorded per model |
| [Last three GPU models](https://github.com/huggingface/transformers-mic/actions/runs/38286933326) | 9 | **9 pass, unpatched**: DETR, SpeechT5, Qwen2.5-Omni |

The PR job checks GitHub's synthetic merge of the proof head into upstream
`78633ef9`; the push job checks the proof branch itself. Both explicitly select
checkout `src` and assert its import path. The same package versions do not mean
the same framework source. **No upstream main was merged into the proof branch.**

On the PR build, historical Whisper unexpectedly passes its error match. The
prototype reaches numeric checks: F32 label log-probabilities differ by 0.000112,
and its native BF16 reference is absent. These remain failures; no new Whisper
numbers were approved. [PR merge evidence](../experiments/results/native-cpu-pr-merge.json).
The passing push is evidence for the proof branch, not a green PR merge build.

The Transformers branch is
[`contracts-proof`](https://github.com/huggingface/transformers/tree/contracts-proof).
Its native recording workflow now pins the previously tested PR merge source
`1111c102092739070a59f8eb41f8008fc9709213` explicitly and records unpatched
candidates without approval or export. This selects the source for recording;
it does not merge upstream main into the branch or change the source used by
the push proof.

The focused recording succeeded for all three variants on m8i with four threads.
Default and eager retain the same `label_logprobs` delta of 0.000112135895;
training loss moves by 0.000002822373 within its existing tolerance. All generated
token and text outputs match their old F32 references exactly. Native BF16 has a
new candidate. The old F32 references report Python 3.10.12 and a CUDA-enabled
PyTorch build while executing on CPU; the new candidates use Python 3.11.17 and
CPU wheels. The causal contribution of source, build and runtime differences
to the F32 movement has not been isolated. No candidates were approved or
exported, and no error matches were removed. This is successful candidate
recording, not a green PR proof.
[Candidates and numeric review](../experiments/results/native-whisper-fixed-source.json).

## Scope and runtime

The installed `transformers-mic` package runs two locks: the six-contract,
eight-fixture historical control in its default variant, plus 23 prototype
architectures in default/eager/bfloat16 (**69 cases**). Full proofs use `--run-all`
and fail if any prototype case is missing. Zero-selection greens are not model
validation. [Final records](../experiments/results/native-cpu-final.json).

The expanded fleet is fixture-only, with immutable public Hub pins and no
historical replay. Auto uses explicit Auto classes for BERT/T5/ViT. GPT-OSS's
slim is dense and does not cover quantized kernels. Some GPU runs used documented
model-scoped patches; they do not establish unpatched upstream behavior.
[Per-model scope, pins and rejections](../experiments/results/prototype-regeneration.json).

CPU uses Intel m8i, four threads, torch/torchvision/torchaudio 2.11/0.26/2.11,
NumPy 1.26.4 and pinned scientific dependencies. Python 3.11.17 comes from the
image and is reported against the catalog's 3.10 pin. Native BF16/FP16 references
have separate platform files. Candidate review approved only new or unchanged
files; imported outputs also compare within existing tolerances.
[Recording review](../experiments/results/native-cpu-review.json),
[import comparison](../experiments/results/native-cpu-import-comparison.json).

The full contract step takes about **12 minutes**; the historical control takes
about 20 seconds. Prototype variants currently start a backend process per case.
The pilot's two-minute PR budget does not describe this expanded CLI run.

## Exceptions and noise

No runtime framework patches enter the Transformers proof.

- **Whisper / #49456:** the proof branch matches the precise generation error
  in one historical case and three prototype variants. Later checks remain
  blocked. The newer PR merge build needs its own reviewed Whisper references
  and removal of obsolete error matches.
- **Wav2Vec2 / #49385:** only `training.loss` and `training.gradients` are skipped.
  Controlled recording initializes the missing SpecAugment parameter; final
  checks run unpatched and retain those explicit skips.
- **Table Transformer BF16:** an eight-thread proof produced a 0.00195 score
  delta. Four threads match the recording profile. A focused fresh-process
  repeat matched every original native output exactly, and the aligned full
  push passed. No replacement reference was imported.
  [Noise evidence](../experiments/results/native-cpu-thread-noise.json),
  [exact repeat](../experiments/results/native-table-repeatability.json).

Numeric tolerances and the .001 sensitivity threshold remain unchanged.
DETR/SpeechT5 exempt only amplitude scaling canceled by input normalization;
the other probes remain required. [Original rejected probes](../experiments/results/prototype-normalization-probes.json).

## Remaining work

1. Whisper candidates are now recorded on the exact fixed PR merge source
   `1111c102`. Explain the reproducible F32 drift before approving changed
   references; review the new native BF16 candidate, then check and export on
   that same source. Remove obsolete error matches only for a fixed build;
   the branch-head generation bug remains present.
2. Diagnose the three rejected architectures: CSM's real audio processor path,
   Gemma3n's nonfinite gradients, and PI0's weak state/image sensitivity. Keep
   gates fixed; finite forward calls alone do not qualify a fixture.
3. Measure repeated native CPU noise and reduce subprocess overhead before
   proposing broader PR adoption. Full checkpoints and historical replay of
   the new prototypes remain separate work.

The [2026-10-09 pilot](transformers-ci-proof-pilot-2026-10-09.md) preserves the
original six-contract results, UMT5 regression demonstration, timings and
historical issue states. They are historical evidence, not current fleet scope.
