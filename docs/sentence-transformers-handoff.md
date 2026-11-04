# Sentence Transformers MIC handoff — 2026-11-04

## Current status

The first Linux nightly **completed successfully**:
https://github.com/huggingface/transformers-mic/actions/runs/41797350961

It ran in **record mode**, on the pinned preview MIC, generating Linux candidate
baselines. This is a successful candidate recording, not yet a comparison against
reviewed Linux baselines. Its artifacts have **not yet been downloaded or
published to the Space**. The user's latest functional request was to run the
nightly once so they could see the Space populated. Finish that next.

The public Space is created and reachable (HTTP 200), but still has empty history:
https://huggingface.co/spaces/huggingface-mic/sentence-transformers-ci

Direct app: https://huggingface-mic-sentence-transformers-ci.static.hf.space/

## Where the work is

Use the isolated worktree:

```
/Users/tarek/Dev/transformers-ci-playbooks/mic/worktrees/sentence-transformers-integration
```

Branch: `sentence-transformers-integration`, pushed to
`huggingface/transformers-mic`. Latest commit: `c1bd13d`.
No PR has been opened. Nothing has been merged to main.

Commits:

- `e9bde99`: integration, nightly, Space, tests and documentation.
- `35daca0`: temporary branch-only push trigger to launch the first nightly.
- `dfdd7cc`: removed that trigger; no more branch pushes trigger the nightly.
- `c1bd13d`: report status must match check versus record mode.

The shared `mic/checkout` was switched by another task to main during our work
(reflog shows a separate `runtime-accelerate` task). We copied our changes into
the isolated worktree. The shared checkout still has the earlier working-tree
copies of our changes. **Do not reset/clean it indiscriminately**. The pre-existing
untracked `docs/blog-post.md` is unrelated and must be preserved.

Python environment:

```
/Users/tarek/Dev/transformers-ci-playbooks/mic/checkout/.venv/bin/python
```

Its editable install points at `mic/checkout`, so use `PYTHONPATH=src` when running
package tests/code from the isolated worktree. Sentence Transformers main was
installed there: `6.3.0.dev0`, commit
`b26925868810d978e06d731f6b527c2c74579b1b`. The editable MIC metadata was refreshed
to 0.1.2 (it had stale 0.1.0 metadata).

## Immediate next action: populate the Space

Run from the isolated worktree:

```sh
gh run download 41797350961 --repo huggingface/transformers-mic \
  --name sentence-transformers-results --dir work/nightly-41797350961

# Inspect results and candidate files first.
find work/nightly-41797350961 -type f

PYTHONPATH=src ../../checkout/.venv/bin/python \
  tools/update_sentence_transformers_space.py \
  --results work/nightly-41797350961 \
  --out work/space-41797350961 \
  --run-id 41797350961 --attempt 1 \
  --repository huggingface/transformers-mic \
  --conclusion success --mode record --execute
```

The user authorized trying this nightly and populating the Space. Hub credentials
can create/upload the Space (already proven). The helper downloads current
history, appends/deduplicates the real Actions run, and uploads the static assets
and data. Use `mode=record` so the UI says **Candidates recorded**, not Passing.

Verify public `data.json` contains run id `41797350961` and the UI shows its
results. Then give the user the Space and Actions links. Do not fabricate a
passing baseline check or upload the local preview as an Actions result.

GitHub CLI works outside the filesystem/network sandbox (`gh api` approved
prefix). Network failures inside the sandbox can look like invalid auth; auth
is actually valid and has repository admin/push permissions. `curl -sS` can read
the public static Space.

## What was implemented

- Integration identifier is **`sentence_transformers`**. User corrected their
  original `sentencepiece` typo explicitly.
- `tmic integrations --integration transformers --integration sentence_transformers`
  generates/retains both integrations on an existing built MIC workspace.
  A single `--integration sentence_transformers` keeps only ST on that fixture.
- This is a **staged command**, not a new `tmic run` or `tmic create` integration
  flag. Fixture creation/build gates and automated cloud orchestration still
  use Transformers. Documented in `docs/cli.md`.
- Initial generation supports only `google/embeddinggemma-2`.
- Scenario calls `encode_query(query)`, `encode_document(passages)` and
  `similarity`, converts to a finite NumPy `(4,)` array, records all four scores
  with `atol=rtol=1e-4`. Fixed the user's undefined `document` variable.
- Each integration may override its complete `outputs` map; existing TF
  assertions remain unchanged.
- `record`, `review`, `check`: `--integration transformers|sentence_transformers|all`.
  Fleet checks require one integration per invocation.
- Baselines and validation receipts are integration-specific. Publishing requires
  all declared integrations/variants and copies all their reviewed baselines.
- Optional `<fleet>/integrations.csv` advertises integration names, bound to
  immutable contract revisions. Original three CSV tables retain their shape.
  Older exporters' stale metadata is ignored if the revision differs.
- Runner skips contracts without the selected integration. ST does not inherit
  Transformers known failures or local patches. Empty ST selection fails.
- `.github/workflows/sentence-transformers-nightly.yml`: daily 04:23 UTC Linux CPU;
  ST main by default, latest release selectable manually; TF main; check/record
  modes; immutable preview model input optional for bootstrap.
- `.github/scripts/sentence_transformers_nightly.py`: fleet selection across both
  fleets at one pinned revision, or verified preview MIC. Records provenance.
- `spaces/sentence-transformers/`: static maintainer dashboard with latest check,
  failure details, dependency/fleet/model pins, Actions links, up to 90 runs,
  separate candidate-recording status, stale notice after 36 hours.
- `tools/update_sentence_transformers_space.py`: builds/publishes dashboard data.
  Empty/incomplete runs are errors, not green. No notifications implemented or sent.
- Workflow `publish-results` is separate from execution, runs even on failures,
  uses `mic-publish` environment's existing `MIC_BASELINE_WRITE_TOKEN`, and is
  restricted to main. Hence the first branch run needs the local publish above.

## First fixture discovery and staged repair

The existing EmbeddingGemma fixture had `modules.json` referencing `1_Pooling`,
but lacked `1_Pooling/config.json`. ST failed to load it. The builder only copied
root-level assets, and fixture hashing also covered only root-level files.

Fixed copying/hashing for module subdirectories listed in `modules.json`, with
path validation and existing file-size/weight exclusions. Publisher already
creates parent directories for copied files.

Prepared a separate local fixture adding the missing config unchanged from
source checkpoint `google/embeddinggemma-2` at
`f9972ac22e696a105c482202646979e33e4b5a1e`. Verified model weights remained
byte-for-byte identical. Recorded repair provenance and updated hashes.

Original workspace (preserved):
`mic/checkout/work/mic/google--embeddinggemma-2`

Staged workspace:
`mic/checkout/work/sentence-transformers/embeddinggemma-2`

Corrected local fixture:
`mic/checkout/work/sentence-transformers/embeddinggemma-2/.mic/artifacts/embeddinggemma-2-st-assets`

Hub preview branch: `huggingface-mic/embeddinggemma-2-mic`,
`sentence-transformers-preview`.
Immutable preview SHA: **`a0787fefb042bed5f323e35d5edeaf4e979ce4f8`**.

https://huggingface.co/huggingface-mic/embeddinggemma-2-mic/commit/a0787fefb042bed5f323e35d5edeaf4e979ce4f8

This branch was based on published MIC SHA
`839a2b1f9c2b144e2203b2f4972ababfc888fd99`. Upload included the added ST usage,
manifest/layout, missing pooling config, and repaired build provenance. Existing
weights and TF usage/baselines stay inherited. **Model main and shared fleet pins
have not changed.** Preview build key was changed to reflect the asset repair.

Local macOS recording reproduced in two fresh processes, was approved as a new
baseline, then passed a check. Scores:

```
[-0.0694104284, 0.4365914464, -0.1546532661, -0.0959652290]
```

Runtime CPU ARM64, bfloat16. This does not substitute for Linux baselines.

## Validation and deployment evidence

- Full suite: **415 passed** in isolated worktree with `PYTHONPATH=src`.
- Additional report-only tests passed after the final one-line status fix.
- `actionlint .github/workflows/sentence-transformers-nightly.yml` passed.
- `git diff --check` passed.
- JavaScript syntax checked with Node; dashboard rendered in headless Chrome.
  Screenshot: `/private/tmp/mic-st-space.png` (local preview, never uploaded).
- Live Space initial commit:
  `26bd8a8861869632adcef86fed9e7265af85417a`.
- Public app and empty `data.json` verified via curl.
- Linux recording run **41797350961 succeeded**, exact job artifacts not yet inspected.

## Remaining rollout after the immediate Space update

1. Inspect/review Linux candidate scores from that run.
2. Publish reviewed Linux baselines and corrected fixture through the normal MIC
   publication process; preserve existing TF checks and validate all declared
   integrations/variants before model main/fleet publication.
3. Add the integration metadata to the shared fleet using the new exporter.
4. Review/merge the branch (no PR yet), enabling the daily schedule and main-only
   automatic Space publisher. Ensure `MIC_BASELINE_WRITE_TOKEN` can write the Space.
5. Run a real baseline comparison and verify it reaches the Space.
6. Later, choose maintainer/contact and notification policy. User said eventually
   ping on failures; no current authorization or need to message anyone.

Compatibility note: CSV metadata is backward-readable, but corrected fixtures
with nested module assets require the updated runner's hashing support. Do not
promise old runner code can validate these new fixture snapshots unchanged.

No approvals should be inferred for changing tolerances or accepting changed
existing baseline values merely to make checks pass.
