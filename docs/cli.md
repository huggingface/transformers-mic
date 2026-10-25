# transformers-mic CLI

`transformers-mic` packages the existing MIC runner, fixture builder, replay tools,
and reviewed catalog. The `tmic` command creates a working directory for a model,
proposes additional usage paths when needed, and measures expectations there.
It is an alpha implementation: generated plans still need code review and a
working model-specific build recipe. New projects use one repository,
`huggingface-mic/<name>-mic`, for slim weights, the contract, usage and reviewed
framework expectations. See [the Hub layout](hub-organization.md).

## Install

From this repository (Python 3.11+):

```sh
python -m pip install -e '.[runtime,dev]'
python -m pip install 'hf-doc-builder @ git+https://github.com/huggingface/doc-builder.git'
hf auth login        # publishing needs write access to the target namespace
gh auth login        # history search; or pass --no-history to create
```

## Dashboard authentication

```sh
tmic auth login
tmic auth status
tmic auth logout
```

Login opens the Transformers CI dashboard. Sign in with GitHub, compare the code
with the one printed in your terminal, and explicitly authorize tmic. On SSH,
use `tmic auth login --no-browser` and open the printed URL elsewhere. Use
`--dashboard https://dashboard.example.com` on each auth command for another
deployment.

Dashboard access lasts eight hours and can be revoked from your profile. The
CLI stores its dashboard credential in
`~/.config/transformers-mic/dashboard-auth.json` (or under `$XDG_CONFIG_HOME`),
with file permissions `0600`. It stores no GitHub token and uses neither
Grafana nor your local `gh` login. Logout revokes the server session before
removing the local credential; connection failures leave it available to retry.

This authenticates the CLI for the remote baseline service below. Workflow
dispatch also requires repository write access. Baseline approval and publication
remain explicit operations; the auth commands themselves do not execute a job.
Local Hub publishing uses `hf auth login`; direct workflow publication uses the
operator's dedicated Hub write secret.

**Activation is pending:** browser OAuth needs the existing GitHub App's callback
URL/client secret, and Serge's App must be installed on `transformers-mic`.
Until then, use the GitHub-authenticated workflow test below.

## Cloud baseline collections

After dashboard activation, one command can measure the entire pinned collection
on the dashboard/Serge ARC runners and retrieve models and candidate baselines:

```sh
tmic baselines run --out work/baselines
# Or have the workflow upload directly, returning only the Hub revision:
tmic baselines run --out work/baselines-hub \
  --upload-repo huggingface-mic/baseline-candidates --no-download
```

The CLI shows remote stages and per-profile case counts. Repeat the same command
to resume the same job. See [cloud baselines](cloud-baselines.md) for local versus
cloud execution, artifact retrieval, direct or local Hub upload, baseline review,
Hub model PRs, and operator setup. These commands refresh the shared fleet's
existing immutable fixtures; new model creation still uses the workflow below.

```sh
# Explicitly narrow the scope for a first dashboard test:
tmic baselines run --out work/smoke --fleet control --fixtures-only
tmic baselines status --out work/smoke
# Inspect downloaded measurements, then preview a dataset upload:
tmic baselines review --out work/smoke
tmic baselines upload --out work/smoke --repo my-org/baseline-results
# Add --execute to upload. Canonical model changes use reviewed Hub PRs:
tmic baselines publish --out work/smoke
```

### Temporary test without dashboard OAuth

From the repository root, with MIC installed in the active Python environment:

```sh
gh auth login
python tools/try_cloud_baselines.py --out work/arc-smoke
```

This uses your `gh` identity and repository permissions to dispatch the real ARC
workflow directly, watches Actions job/step progress, downloads CPU/CUDA artifacts,
and validates their provenance and checksums. It defaults to the control fleet's
fixtures. Repeat the same command to resume; it never automatically repeats an
ambiguous dispatch. Use a separate `--out` for another collection. Resume these
test collections with the helper, rather than `tmic baselines run/status`.

To request both fleets and real checkpoints:

```sh
python tools/try_cloud_baselines.py --out work/arc-all \
  --fleet all --include-checkpoints
# Optional direct Hub upload (requires the workflow's scoped write secret):
python tools/try_cloud_baselines.py --out work/arc-hub \
  --upload-repo huggingface-mic/baseline-candidates
```

The downloaded collection works with `tmic baselines review/upload/publish`.
Repository image variables must be configured first; direct upload also needs
`MIC_BASELINE_WRITE_TOKEN`. This tests GitHub Actions/ARC and artifact retrieval;
it does not test dashboard/Serge authorization or the live case-progress relay.
There is no anonymous endpoint or production auth bypass. See
[operator setup](cloud-baselines.md#operator-setup) for the full service path.

## Automated mode

```sh
tmic run org/model
```

**Baseline measurement runs in the cloud by default.** Contract creation and
fixture building happen locally. Then `tmic run` uses `tmic baselines run` for
this model on cloud CPU and CUDA runners, covering every declared target and
variant. It stages an immutable input on a `tmic-baselines/<source-hash>` Hub
branch, retrieves and independently verifies the measured candidates, and uses
the existing review/publication flow. The model's main branch and fleet pins
change only after review. Cloud input requires the single `<name>-mic` layout
and an operator-approved Hub namespace (default `huggingface-mic`).

Until dashboard activation, use your GitHub CLI identity for the same model flow:

```sh
tmic run apple/mobilevit-xx-small --baseline-transport github
```

This directly dispatches Actions/ARC using `gh auth login`, without using the
dashboard/Serge path. Your HF login needs permission to stage the model snapshot
and, when approved, publish to the selected namespace. New model drafting also
uses your configured HF inference provider. To measure locally instead:

```sh
tmic run org/model --baseline-execution local
```

`--local-only` also selects local execution and stops before any Hub upload.
Cloud progress appears on stderr; `--baseline-timeout` defaults to six hours.
Repeat the same command to resume the persisted cloud job. Missing CPU/CUDA cases,
changed source/weights, or unreviewed/edited candidates block publication.
Hardware exclusions explicitly declared in the manifest remain visible.

This creates a workspace at `work/mic/org--model` and runs the same measured
workflow as the individual commands: reuse or draft the contract, select/build
weights, check Auto-model registrations, record reproducible candidates for every
declared variant, review them, validate all targets, publish the model snapshot,
validate its published pins, and update the shared fleet dataset. The default fleet
is `prototypes`; use `--fleet control` for the control fleet.

The command asks you to review the executable contract before building it. It
automatically approves only new or unchanged candidate baselines, using the
existing review rules. Changed baselines show a diff and need your confirmation
and an explanation, saved with file hashes in `.mic/automation-reviews.json`.
It previews each upload and asks before publishing the model or updating the fleet.
An intervening dataset update requires a fresh preview and decision.

```sh
# Validate locally, without uploading:
tmic run org/model --local-only
# Choose a workspace and write the final dataset SHA into a framework checkout:
tmic run org/model --out work/my-model --transformers-checkout /path/to/transformers
# For a new model, declare only fixture targets:
tmic run org/model --fixtures-only
```

Declared variants and targets are not silently dropped. Catalog contracts keep
their existing target scope, which can include real checkpoints. Use a runtime
that supports those variants. Model-specific build errors, missing dependencies,
and unsupported hardware stop progression and show the stage log; fix the problem
and retry. `--timeout SECONDS` bounds each stage (default 1800). New models use the
HF planner just as `create` does; `--no-history`, `--provider`, `--llm-model`,
`--weights`, `--namespace`, `--contract`, `--name`, and `--regenerate` are available.

Repeat the same command to resume. State lives in `.mic/automation.json`; logs
live in `.mic/automation/logs/`. Completed uploads are not repeated when resuming
an unchanged workspace. Source, lock, or baseline edits invalidate earlier checks
and restart validation. After `--local-only`, rerun without it to continue through
publication. No Git commit or framework PR is created automatically.

### Bounded contract repair

When a generated contract fails during build, factory checks or measurement,
`tmic run` can inspect the failure and draft a repair. It reads the pinned
architecture's parameter names/shapes on the meta device (without loading its
weights), plus the bounded failure log, current usage and manifest. The configured
HF planner proposes a change; the CLI displays its rationale and usage diff and
asks for approval **before** applying it. An approved repair backs up the old
source, forces a fixture rebuild, and reruns gates and measurements.

The manifest must remain byte-equivalent as JSON: output rules/tolerances, targets,
variants, resources and build recipes cannot change. Explicit assertions and
training gradient/update checks are preserved structurally. Review the code diff
for semantic changes; automated checks cannot prove arbitrary Python equivalent.
No failed measurement is converted to a pass, and no gate is bypassed.

The default limit is two proposals per workspace, including rejected or interrupted
attempts. `--max-contract-repairs 0` disables this; the accepted range is 0–5.
When the budget is exhausted or a proposal fails validation, the ordinary manual
retry prompt remains. Network/access errors retain the normal retry path.
Proposal JSON, diffs, log/source hashes, backups and decisions are retained under
`.mic/automation/repairs/` and `.mic/automation.json`. Resuming a pending repair
reuses the saved proposal rather than making another provider call. Editing the
source, log or proposal invalidates approval.

Agent mode emits a `contract-repair` question using the same one-use answer
protocol as other decisions. Answer `approve` to apply the displayed diff or
`defer` to leave source unchanged. As with other questions, obtain the operator's
decision before approving.

### Human and agent modes

```sh
tmic run org/model --mode human   # default: interactive terminal
tmic run org/model --mode agent   # JSON protocol, no stdin reads
```

Both modes use the same workspace and decisions. You can switch modes when
resuming. `tmic --help` introduces both modes; `tmic run --help` contains the
answer format and exit codes, so an agent can discover the interface directly.
`--non-interactive` is an alias for agent mode.

In agent mode, stdout contains one JSON result and progress/log output goes to
stderr. A pending question exits **3** and includes `protocol_version: 1`,
`status: "needs-input"`, a `resume` command, and `pending_question`:

```json
{
  "question_id": "<opaque ID>",
  "kind": "source-review",
  "question": "Review the executable contract before running it.",
  "details": "Paths and review details...",
  "choices": ["approve", "defer"],
  "requires_reason": false,
  "context": {"workspace_sha256": "<hash>", "files": []},
  "context_sha256": "<hash>"
}
```

Inspect the context and referenced files or diffs. Obtain the operator's decision
when needed, then write a JSON answer file:

```json
{"question_id": "<ID from pending_question>", "answer": "approve", "reason": "Reviewed the displayed snapshot"}
```

Run the emitted resume command with `--answer answer.json`. For example:

```sh
tmic run org/model --mode agent --out work/mic/org--model --answer answer.json
```

Use the **emitted command** to preserve options such as `--local-only`, fleet,
namespace and timeout. The answer file is deliberately absent from that command:
each accepted answer is consumed once, and the next question needs another
answer. Retry questions offer `retry` or `defer`; other questions offer `approve`
or `defer`. Changed-baseline approvals require a nonempty `reason`. `defer`
leaves the question pending and performs no approved action.

Questions bind the workspace snapshot, stage, selected destination, relevant
file hashes and, for fleet updates, the exact dataset base revision and pins.
Polling an unchanged question preserves its ID. Stale or reused approvals exit
**3** with `answer_error` and the current question; approval is never transferred
to another snapshot. Malformed answer files or answers without a pending
question exit **2**. Decisions and their contexts are recorded in
`.mic/automation.json`; changed-baseline explanations also retain their file-hash
receipts. A workspace lock prevents simultaneous runs consuming the same answer.

A successful local-only run (`status: "validated"`) or publication
(`status: "complete"`) exits **0**. Ctrl-C stops stage processes, saves progress
and exits **130**. Invalid invocations/state exit **2**; command-line syntax
errors use the normal argparse error output. Existing individual commands remain
available; `tmic auto` still checks factory mappings for a built fixture.

## Step by step: one model

**The order is: real checkpoint → contract → fixture.** You never write a fixture
first. You start from a real checkpoint id. `create` writes the contract (the
usage code and manifest), and `build` then derives the slim fixture from that
checkpoint. `build` needs the contract because it accepts weights only after
the contract's inputs move the outputs (the sensitivity gate).

1. **Create the workspace from the real checkpoint.**

   ```sh
   tmic create google-bert/bert-base-uncased --out work/bert --namespace huggingface-mic
   cd work/bert
   ```

   `--namespace` sets the owner of the single `<name>-mic` repository.
   It defaults to `huggingface-mic`; an explicit owner overrides that default.
   Nothing is uploaded until publication. The workspace holds usage and candidate
   references locally; reviewed expectations belong under
   `integrations/transformers/expectations/<platform>/`, alongside `usage.md`.

   If the checkpoint is in the bundled catalog, this reuses the published
   contract and needs no LLM call. Add `--regenerate` to build a new
   `<name>-mic` fixture instead of reusing the published one. If the checkpoint
   is new, the planner drafts the contract on your HF token (see
   [When the LLM is used](#when-the-llm-is-used)). By default the contract
   covers both the real checkpoint and the fixture. Use `--fixtures-only` to
   declare a slim-only contract.

2. **Read the draft.** Open `integration.json` and
   `integrations/transformers/usage.md` in the workspace's MIC model directory
   (`example/hub/<name>-mic/`) before running anything. The usage is executable
   Python, and nothing sandboxes it.

3. **Build the fixture.**

   ```sh
   tmic build
   ```

   This shrinks the real config, copies the processors byte for byte and tries
   random weights first. If the gates fail, it tries a bounded overfit. If both
   fail, the build is rejected. `.mic/project.json` records which weights won
   and why ([Random or overfit?](#random-or-overfit)).

   For models with fixed geometry or processor tensors, add reviewed entries to
   the fixture's `build` recipe before this step. For example:

   ```json
   {
     "copy_assets": ["attention_mask.safetensors"],
     "preserve_frozen_prefixes": ["model.geometry."]
   }
   ```

   `copy_assets` names extra files copied byte for byte from the pinned source.
   `preserve_frozen_prefixes` copies matching parameters and persistent buffers
   only when they have `requires_grad=False`; missing names and shape changes
   fail the build. Their source hashes are included in the fixture key and build
   record. The planner cannot add these options; review and edit the recipe yourself.

4. **Record the candidate expectations.**

   ```sh
   tmic record --all-variants
   ```

   Every case runs twice, each time in a fresh process, and the results are
   written to `integrations/transformers/expectations/<platform>/*.candidate.json`
   in the same MIC model directory as the usage.

5. **Review and approve them.**

   ```sh
   tmic review
   tmic review --approve
   ```

   `--approve` accepts only new and unchanged values. Changed values stay
   pending until someone approves them by hand.

6. **Check the approved expectations.**

   ```sh
   tmic check --all-variants
   tmic status
   ```

7. **Publish to the Hub.**

   ```sh
   tmic publish                 # preview the destinations
   tmic publish --execute       # upload; the lock is rewritten to immutable revisions
   tmic check --all-variants    # publishing invalidates receipts: check the published pins
   ```

   Uploads are public and pinned, so this step is outward-facing.

8. **Update the shared fleet and pin Transformers.**

   ```sh
   tmic export --fleet prototypes
   tmic export --fleet prototypes --execute --transformers-checkout /path/to/transformers
   ```

   The first command previews the update; `--execute` uploads it to the
   [`huggingface-mic/transformers-fleet`](https://huggingface.co/datasets/huggingface-mic/transformers-fleet)
   dataset. Only `tests/contracts.lock`, containing the dataset commit SHA, is
   written into Transformers. Expectations stay beside usage in the pinned model.
   Review and commit the pin change in your framework PR.

## Several models at once

List the models in a JSON file shaped like `experiments/proto-models.json`, with
`name`, `model_type` and `checkpoint` for each entry. Then:

```sh
tmic batch --models-file my-models.json --out work/fleet --phase prepare --no-history
tmic batch --models-file my-models.json --out work/fleet --phase run --resume
```

`prepare` runs step 1 for every model, always with `--regenerate`. Read the
drafts in `work/fleet/projects/*/` before you start `run`, which does steps 3–6. `run`
calls `review --approve` itself, so you do not get a manual review there:
inspect the approved expectations before you publish. `batch` keeps going past
failures and records every stage. `--select NAME` reruns one model. `--publish`
adds step 7, and it uploads only after all declared checks pass.

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
to the checksum-checked MIC publisher and updates the lock to an immutable
Hub revision. It invalidates local check receipts, so check again before export.
Upload requires write access to the MIC model namespace, which defaults to
`huggingface-mic`.

`export --contract-name mic-bert` gives the destination a distinct name when
retaining an existing `BERT` contract. Export rejects names that differ only
by case.

`export --fleet control` updates `fleet-lock.json` in the shared dataset;
`--fleet prototypes` updates `prototype-fleet.lock`. The default is `control`.
Exports preserve other entries and strip local paths. Environment changes must
be reviewed separately. Preview first, then use `--execute` to upload. A concurrent
Hub update rejects the commit; rerun to review the new state.
`--transformers-checkout PATH` writes only `tests/contracts.lock` after upload.
Neither command commits or opens a framework PR.

To adopt an existing dataset revision without exporting a workspace:

```sh
tmic pin --transformers-checkout /path/to/transformers --revision <dataset-commit>
# Resolve the latest dataset main to a SHA:
tmic pin --transformers-checkout /path/to/transformers
```

Run the pinned fleet from the Transformers checkout:

```sh
tmic check --fleet control --fixtures-only
tmic check --fleet prototypes --variant bfloat16 --fixtures-only
# Reproduce a previous dataset snapshot:
tmic check --fleet prototypes --fleet-revision <dataset-commit> --variant bfloat16 --fixtures-only
```

`tmic check` outside a MIC workspace uses `tests/contracts.lock` when present.
An explicit `--fleet-revision` overrides that pin; without a local pin the dataset
ref defaults to `main` and is resolved once. `--lock PATH` also accepts a custom
JSON fleet, or `tests/contracts.lock` for the pinned control fleet.

`tmic status` shows project state. Workspaces contain source, history, candidate
expectations, checks and artifacts; keep them outside versioned source or ignore
`work/`. The current release implements Transformers orchestration; the bundled
prototype still includes Transformers.js contracts, but the CLI does not publish
or export a JavaScript fleet.

## Files and publication

A MIC model has one location for usage and expectations, both in the workspace
and on the Hub:

```text
<name>-mic/
  config.json
  model.safetensors
  build.json
  integration.json
  integrations/
    transformers/
      usage.md
      expectations/
        <platform>/
          <target>.<variant>.json
          <target>.<variant>.candidate.json   # local candidates; never uploaded
```

`tmic record` writes candidates in this expectations directory. `tmic review`
approves files there, and `tmic check` reads them there. There is no second
expectations directory in the Transformers checkout.

`tmic publish --execute` combines the built weights, usage and reviewed expectations
in `huggingface-mic/<name>-mic` and pins the Hub commit. Local build artifacts are
staged under `.mic/artifacts/` until publication. `tmic export --execute` updates
the shared fleet dataset. Transformers keeps only the dataset SHA in
`tests/contracts.lock`; CI loads the fleet and model expectations from immutable
Hub snapshots.
