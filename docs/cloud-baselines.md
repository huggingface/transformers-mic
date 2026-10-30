# Baseline collections through the dashboard and ARC

**Status (2026-10-26):** dashboard authentication and the MIC job API are deployed.
Both Apps are installed on `huggingface/transformers-mic`; workflow reads,
Serge's actor-permission check and authenticated progress callbacks are verified.
MIC can use `aws-m8i-l-cache`: the [CPU runner probe](https://github.com/huggingface/transformers-mic/actions/runs/40333451730)
ran in the approved CPU image on a Xeon 6975P-C reporting `amx_bf16` and
`avx512_bf16`, and `MIC_CPU_BASELINE_RUNNER` selects that group. A complete
dashboard CPU/CUDA collection on this runner remains unproven; earlier cloud runs
measured CPU on G5, which does not supply the native bfloat16 class.

The [direct GitHub transport](cli.md#temporary-test-without-dashboard-oauth)
uses the caller's existing GitHub CLI identity instead of dashboard authentication;
it does not bypass runner-group access requirements.

The CLI can also measure a single newly created or updated model through `tmic run`; see below. Cloud execution uses the dashboard's
GitHub identity, Serge's authorized workflow dispatcher, and the ARC runner
groups already used by `huggingface/transformers-mic`. It does not require a
separate cloud account or a GPU on the caller's machine.

## Choose where execution and upload happen

Create or update one model, with local creation/build and cloud CPU/CUDA baseline measurement:

```sh
tmic run org/model
# Temporary direct Actions path while dashboard App setup is deferred:
tmic run org/model --baseline-transport github
# Explicitly keep measurement local:
tmic run org/model --baseline-execution local
```

Generate the complete collection remotely and retrieve the models and baselines:

```sh
tmic auth login
tmic baselines run --out work/baselines
```

Have the workflow upload directly to a Hub dataset as well:

```sh
tmic baselines run --out work/baselines-hub \
  --upload-repo huggingface-mic/baseline-candidates
```

Leave the results in the cloud and return only the immutable Hub revision:

```sh
tmic baselines run --out work/baselines-cloud \
  --upload-repo huggingface-mic/baseline-candidates --no-download
```

The full collection means **both control and prototype fleets, CPU and CUDA,
every declared variant, and real checkpoints as well as fixtures**. Use
`--fleet control`, `--fleet prototypes`, or `--fixtures-only` to explicitly narrow
the requested scope. An omitted case, unstable measurement, unexpected hardware
skip or failed assertion makes the collection incomplete. Manifest-declared
unsupported targets remain visible with their reasons.
Device exclusions explicitly declared in manifest resources are accounted for
as unsupported on that profile; insufficient memory on an otherwise supported
profile remains a failure.

This refreshes baselines for the fleet's **existing immutable MIC fixtures**.
The returned model files are those pinned slim snapshots, not newly generated
weights. New contracts and rebuilt fixtures still follow the existing
`tmic create/build/run` source-review flow before entering the shared fleet.
For model-scoped runs, the user reviews the executable contract first, then
approves uploading its built input to a separate `tmic-baselines/<source-hash>`
branch in the model's Hub repository. The service accepts only an immutable
revision in an operator-approved namespace. It checks the snapshot's single-model
lock, source/build hashes and complete case matrix. The workflow reads that
snapshot; it never receives local paths, arbitrary shell commands or the caller's
Hub write token. Source execution still requires MIC repository write access.

`tmic baselines run --model-workspace work/my-model --fleet prototypes --out work/model-baselines`
is the underlying model command. Add `--transport github` to dispatch with the
caller's existing `gh` identity during testing. `tmic run` imports its candidates,
requests review for changed outputs, and binds publication to both hardware
profiles and the unchanged model input. The `check` stages validate this returned
cloud evidence rather than requiring a GPU on the caller's machine. `--local-only`
and `--baseline-execution local` retain the local workflow.

## Progress and resuming

The CLI prints progress to stderr as work happens remotely: queueing, environment
setup, model download, recording, validation, and publication. CPU and CUDA each
report processed-case counts and the current contract/variant. JSON results go
to stdout. Progress is updated approximately every ten seconds; it is not a
byte-for-byte stream of GitHub logs. The job record includes the Actions run URL.

Repeat the same command and `--out` to resume watching or retrieve results.
The request ID is persisted before submission, so a timeout or retry cannot
silently submit the same GPU job again. `main` is resolved only at initial
submission; resuming does not adopt a newer framework or fleet snapshot.

```sh
tmic baselines run --out work/baselines --no-wait
tmic baselines status --out work/baselines
tmic baselines run --out work/baselines
# A deliberate historical measurement:
tmic baselines run --out work/historical \
  --fleet-revision <dataset-commit-sha> --transformers-ref <framework-commit-sha>
```

Changing fleet selection, target scope, framework ref, upload destination or
dashboard origin requires another `--out`. The local `cloud.json` records the
job and immutable MIC, Transformers and fleet revisions. If dispatch was
ambiguous, the dashboard searches for the exact workflow correlation ID rather
than blindly dispatching again. An unresolved dispatch is reported as unknown.

Candidate expectations and reports are under
`artifacts/<profile>/<fleet>/<contract>/`. Artifacts carry results only: the
runner refuses any contract or fixture whose files differ from the fleet lock's
`files_sha256`, so `collection.json` lists the `repo`, `revision` and
`files_sha256` it measured, and the collection is accepted only when each of
them is pinned by the requested fleet. Each profile has
`collection.json` with the complete expected matrix, execution evidence,
runtime/package information and file hashes. Failed runs retain available
artifacts for investigation and cannot masquerade as complete collections.

## Hub uploads and reviewed baselines

Remote upload uses a dedicated trusted publisher job after **both** measurement
jobs succeed. Measurement jobs receive only `HF_HUB_READ_TOKEN`. The publisher's
write token is scoped to its upload step, which never executes downloaded model
code. Upload destinations must be in operator-approved namespaces.

The dataset contains `collections/<job-id>/cpu/` and `cuda/`, including measured
candidates, reference values, manifests, pinned models and validation evidence.
Only checksum-covered files and the collection manifests are uploaded; runtime
logs are not part of the public Hub collection. It is a measured artifact, not
automatic acceptance of changed framework expectations.

Alternatively, download first and upload using the caller's own `hf auth login`:

```sh
tmic baselines upload --out work/baselines --repo my-org/baseline-results
tmic baselines upload --out work/baselines --repo my-org/baseline-results --execute
```

Review candidates and open Hub model PRs for canonical expectations:

```sh
tmic baselines review --out work/baselines
tmic baselines review --out work/baselines --approve
# Changed values need an explicit decision and explanation:
tmic baselines review --out work/baselines --approve --approve-changed \
  --reason 'Reviewed the intended framework behavior change'
tmic baselines publish --out work/baselines
tmic baselines publish --out work/baselines --execute
```

`--approve` accepts only new/unchanged measurements. Changed values are displayed
as diffs; `--approve-changed` records their hashes and the supplied reason.
Exit 3 means decisions remain pending. Publishing previews all affected files,
then opens Hub PRs against the pinned parent revisions. It does not silently
merge them, alter tolerances, update fleet dataset `main`, or change the
framework's fleet pin. After review and Hub PR merge, adopt the new snapshots
through the normal fleet export/pin process. Split legacy repositories must be
migrated to the single `<name>-mic` layout before this publication path.

## Operator setup

1. Activate the dashboard-owned GitHub auth described in the dashboard's
   `docs/dashboard-auth.md`. Enable `traceExporter.micJobs.enabled`, retaining
   the persistent run store and Serge service-token connection.
2. Install the dashboard read App and Serge's write App on
   `huggingface/transformers-mic`. GitHub reads need Actions/Contents read;
   dispatch needs Actions write. The authenticated actor also needs repository
   write access. A login or CLI token alone does not grant GPU spending rights.
3. Ship Serge's updated repository/workflow allowlist. If environment overrides
   are used, include `huggingface/transformers-mic` in `DASHBOARD_REPOSITORIES`
   and `cloud-baselines.yml` in `DASHBOARD_DISPATCH_WORKFLOWS`, preserving existing
   entries. It dispatches the workflow on the default branch only.
4. Merge `cloud-baselines.yml` and its worker code into MIC `main`. Confirm that
   MIC can use `aws-g5-4xlarge-cache`. Both profiles default to this verified
   host group; the CPU profile uses CPU-only Torch and hides CUDA. **G5's CPU
   does not supply Transformers CI's native bfloat16 baseline class.** Models
   declaring bfloat16 need `cpu-x86_64-bf16`; float16 needs `cpu-x86_64-fp16`.
   Grant MIC access to an x86 runner with the matching instructions (such as
   Transformers' `aws-m8i-l-cache` for native bfloat16), then set
   `MIC_CPU_BASELINE_RUNNER` to that group. MIC's access to `aws-m8i-l-cache` is
   verified by the CPU runner probe. Model import/validation rejects mismatched CPU coverage;
   never rename a baseline from another hardware class to fill the gap.
   Operator variables `MIC_CPU_BASELINE_RUNNER` and `MIC_GPU_BASELINE_RUNNER`
   can select other approved ARC groups when full checkpoints need more memory
   or GPUs. Callers cannot supply arbitrary runner groups.
5. Baselines are measured in **Transformers' own CI images**: CPU in
   `huggingface/transformers-torch-light` (what `fast-integration` runs) and CUDA in
   `huggingface/transformers-all-latest-gpu` (the GPU nightly's image). The workflow
   resolves each image's current digest at dispatch and runs by digest, so every
   collection names the exact image it was measured in. The image's torch, numpy and
   system libraries are used as is; the framework commit and contract tools are
   installed into it the way Transformers CI does. A separate MIC image once pinned
   torch 2.11 while Transformers CI ran 2.14, and sensitive fixtures failed in CI
   against baselines from the other torch. Optional variables
   `MIC_CPU_BASELINE_IMAGE_REPO` / `MIC_GPU_BASELINE_IMAGE_REPO` select another
   repository. When Transformers rebuilds its image, re-record: changed values
   then surface in review instead of as CI failures.
6. Grant `HF_HUB_READ_TOKEN` to `huggingface/transformers-mic`. It is the
   `huggingface` **organization** secret Transformers' own CI uses for gated test
   checkpoints (exposed there as `HF_TOKEN`); an org admin adds this repository to
   the secret's selected repositories, so there is no separate token to create or
   rotate. `cloud-baselines.yml` and `nightly.yml` pass it as `HF_TOKEN`. Without
   it the runner logs "You are sending unauthenticated requests to the HF Hub" and
   every gated checkpoint (Gemma, Llama) fails with `401 GatedRepoError`, on CPU
   and CUDA alike; slim fixtures are public and unaffected. `tmic create` warns
   when a checkpoint is gated. The token's account must have accepted each gated
   model's license. For direct
   upload, configure `MIC_BASELINE_WRITE_TOKEN`, restricted to the intended Hub
   namespace. Set `MIC_BASELINE_UPLOAD_NAMESPACES` (CSV; default `huggingface-mic`)
   and matching dashboard `traceExporter.micJobs.uploadNamespaces` if needed.
   No upload is attempted when `--upload-repo` is absent.
7. For live case progress set Serge's `DASHBOARD_MIC_PROGRESS_URL` to
   `http://trace-exporter.transformers-ci.svc.cluster.local:8000/mic-jobs/progress`.
   The workflow calls `MIC_PROGRESS_URL`, default
   `https://serge.huggingface.tech/dashboard/mic-progress`. Serge verifies GitHub
   OIDC signature, issuer, audience, repository, workflow, branch and event;
   the dashboard binds progress to the actual Actions run and rejects stale
   sequence numbers. Allow the internal Serge-to-exporter network path. Native
   Actions job/step progress remains available if the callback relay is down.
8. Allow exporter egress to the GitHub API and signed Actions artifact storage
   (`*.blob.core.windows.net`, `*.actions.githubusercontent.com`,
   `*.githubusercontent.com`). In production the cluster boundary admits only
   GitHub's API ranges, so the exporter sends both artifact hops through its
   GitHub egress proxy (`traceExporter.auth.githubProxy`, transformers-ci #359);
   without it artifact downloads hang until the ingress returns 504. The App
   token is stripped before following a signed storage redirect. The CLI downloads only through the authenticated
   dashboard and needs no GitHub token.

The job API accepts only fixed fleets, immutable revisions, the fixed CPU/CUDA
profiles and approved upload namespaces. It accepts no user command, image,
arbitrary source repository or caller-supplied identity. Model inputs are separately restricted by `TCI_MIC_INPUT_NAMESPACES` on the dashboard and `MIC_BASELINE_INPUT_NAMESPACES` in the workflow (both default to `huggingface-mic`). The initial limits are
three submissions per user per hour and two active collection jobs globally.
Abandoned jobs age out of capacity accounting after eight hours.

Validation covers request authorization, retries, ownership, artifact traversal,
matrix accounting, independent checks, review hashes, Hub upload/PR previews,
direct cloud publication, and progress authentication. A live proof still needs
a bounded collection run with CPU measured on `aws-m8i-l-cache`.

## BatchNorm fixtures

After overfitting a slim fixture with BatchNorm, the builder recalibrates running
statistics on the captured training batch with dropout disabled. This changes
BatchNorm buffers only and records the layer/batch count in `build.json` training
evidence. The same input-sensitivity thresholds and dtype health checks must
then pass before the fixture can be used for cloud measurements.

GitHub baseline artifacts include hidden source files such as `.mic-repo.json`
and `.mic/baseline-lock.json`. For older uploads that omitted these files, the
GitHub client restores missing source dotfiles from the recorded immutable Hub
revision and checks their worker-recorded SHA-256 before validating the collection.
Measured candidates are never reconstructed this way. A successful remote run
can therefore be resumed after an artifact-download failure without new GPU work.
