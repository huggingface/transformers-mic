# Transformers CI proof

Status: 2026-10-13. The expanded proof runs on `contracts-proof`; it is not
adopted for unrelated Transformers PRs. The package work was squashed and pushed
to transformers-mic `main` as `03500b0`.

**BF16 is the default precision for the current proof.** It checks the BF16
variant of each of the 23 prototypes and one historical SmolLM fixture with a
reviewed BF16 default: 24 cases in total. The earlier 77-case results below
include F32 runs and remain historical evidence.

## Results and tested source

| Check | Executed | Result |
| --- | ---: | --- |
| [BF16 proof-branch CPU push](https://github.com/huggingface/transformers/actions/runs/38364160524), head `6fac16cc` | 24 | **23 pass, one precise Whisper xfail** |
| [BF16 PR synthetic merge](https://github.com/huggingface/transformers/actions/runs/38364165079), proof head `6fac16cc` | 24 | **24 pass**, including Whisper |
| [Proof-branch CPU push](https://github.com/huggingface/transformers/actions/runs/38298088861), source `11c4611c` | 77 | **73 pass, four message-matched Whisper xfails** |
| [PR synthetic merge](https://github.com/huggingface/transformers/actions/runs/38298093605), source `1111c102` | 77 | **73 pass, four Whisper failures**; newer upstream source changes generation behavior |
| [Native CPU recording](https://github.com/huggingface/transformers/actions/runs/38281747657) | 66 checked variants | 22 models checked/exported; prototype Whisper blocked at generation |
| [Whisper candidates on fixed PR source](https://github.com/huggingface/transformers/actions/runs/38302181914), source `1111c102` | 3 variants, each repeated in a fresh process | **3 recorded, unpatched**; this recording preceded the focused BF16 review |
| Published GPU pins | 23 architectures, three variants each | Checked after publication; some model-scoped patches, as recorded per model |
| [Last three GPU models](https://github.com/huggingface/transformers-mic/actions/runs/38286933326) | 9 | **9 pass, unpatched**: DETR, SpeechT5, Qwen2.5-Omni |
| [Current-image observational PR merge](https://github.com/huggingface/transformers/actions/runs/38368933981), proof head `aec34847` | 24 | **22 pass, two numeric failures**, non-blocking: Wav2Vec2 logits 0.000977 and SmolVLM selected logits 0.00195 beyond tolerance |

The PR job checks GitHub's synthetic merge of the proof head into upstream;
the push job checks the proof branch itself. Both explicitly select checkout
`src` and assert its import path. The same package versions do not mean the
same framework source. **No upstream main was merged into the proof branch.**

On the earlier PR build, historical Whisper unexpectedly passed its error match.
The prototype reached numeric checks: F32 label log-probabilities differed by
0.000112, and its native BF16 reference was absent. This was resolved for the
current BF16 proof by reviewing the native BF16 candidate and selecting the
error match according to the checked-out framework source.
[Earlier PR merge evidence](../experiments/results/native-cpu-pr-merge.json).

The Transformers branch is
[`contracts-proof`](https://github.com/huggingface/transformers/tree/contracts-proof).
Its manual native recording workflow defaults to the previously tested PR merge
source `1111c102092739070a59f8eb41f8008fc9709213` and records unpatched BF16
candidates without approval or export. This selects the source for recording;
it does not merge upstream main into the branch or change the source used by
the push proof.

The focused recording succeeded for all three variants on m8i with four threads.
Default and eager retain the same `label_logprobs` delta of 0.000112135895;
training loss moves by 0.000002822373 within its existing tolerance. All generated
token and text outputs match their old F32 references exactly. Native BF16 had a
new candidate. The old F32 references report Python 3.10.12 and a CUDA-enabled
PyTorch build while executing on CPU; the new candidates use Python 3.11.17 and
CPU wheels. The causal contribution of source, build and runtime differences
to the F32 movement has not been isolated. At the time of this recording, no
candidates were approved or exported. The F32 candidates remain unapproved.
[Candidates and numeric review](../experiments/results/native-whisper-fixed-source.json).

The native Whisper BF16 candidate is now reviewed and included in the published
prototype workspace and proof branch. Its finite training checks and outputs reproduced in
fresh processes and in an independent m8i repeat. The source, PyTorch wheel,
and four/eight-thread controls produced identical outputs. The two changed F32
candidates remain unapproved. BF16 is the selected proof precision; the old F32
reference movement does not set the BF16 proof baseline.

The Mistral3 BF16 reference was also reviewed after independent m8i recordings
on both framework sources reproduced its output exactly. Predicted tokens,
finite gradients and the training update remained stable; existing tolerances
were retained. [Review evidence](../experiments/results/native-mistral3-bf16-review.json).

## Scope and runtime

The earlier full proof ran two locks: the six-contract,
eight-fixture historical control in its default variant, plus 23 prototype
architectures in default/eager/bfloat16 (**69 cases**). Full proofs use `--run-all`
and fail if any prototype case is missing. Zero-selection greens are not model
validation. [Final records](../experiments/results/native-cpu-final.json).

The current default selects only the prototype `bfloat16` variant and one
historical SmolLM BF16 fixture. Its coverage guard requires all 23 prototype
cases and the control, and verifies reviewed BF16 runtime expectations. The
other five historical contracts are outside this default proof while their
BF16 coverage is developed.

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

The earlier 77-case contract step took about **12 minutes**. The 24-case BF16
contract step took about two minutes on the proof runners. Prototype cases
currently start a backend process per case; repeated timing is still needed
before proposing wider PR adoption.

The observational PR run uses the current torch-light CI image: PyTorch
2.14.0+cpu and NumPy 2.3.5 in the first run. Its reviewed references were
recorded with PyTorch 2.11.0 and NumPy 1.26.4. Removing the 2.11 wheel
replacement reduced the PR install step from 5 minutes 32 seconds to 8
seconds. The two numeric differences above remain unreviewed; no tolerance or
baseline changed. The workflow completed successfully despite the contract
step's failure and uploaded its per-case results and version record.

## Exceptions and noise

No runtime framework patches enter the Transformers proof.

- **Whisper / #49456:** the proof branch matches the precise generation error
  in the BF16 prototype case. The newer PR merge build has the generation fix,
  removes that match at runtime and checks the reviewed BF16 reference.
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

1. Land the 24-case BF16 fleet as an observational PR check. The workflow
   selects contracts affected by changed files, writes per-case results and
   selection data to an artifact, and warns on failures without blocking a PR.
   The current self-hosted runner limits this wave to members and
   collaborators; external fork PRs need an isolated runner path. Verify the
   non-blocking behavior with a deliberately failing proof PR before merge.
2. Observe selected cases, passes, known failures, new failures, empty
   selections, setup errors and runtime across PRs. The observational job uses
   current torch-light CI image versions rather than reinstalling the historic
   recording pins. Review version-linked failures before changing baselines.
   Repeat native CPU runs on the fixed four-thread profile to identify noise
   before expanding coverage.
3. Add historical replay and full-checkpoint checks with owners for baseline
   changes. Diagnose CSM's audio path, Gemma3n's nonfinite gradients and PI0's
   weak sensitivity separately, with their validation gates unchanged.

The [2026-10-09 pilot](transformers-ci-proof-pilot-2026-10-09.md) preserves the
original six-contract results, UMT5 regression demonstration, timings and
historical issue states. They are historical evidence, not current fleet scope.
