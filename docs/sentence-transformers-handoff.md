# Sentence Transformers MIC rollout — 2026-11-04

## Merged and active

- Code: PR [#193](https://github.com/huggingface/transformers-mic/pull/193), merged at `7035afb63fea1eac348b65a6d4f25af9ebcff18a`.
- Model main: [`372872b46b401801f61265a4ae4aae437540af38`](https://huggingface.co/huggingface-mic/embeddinggemma-2-mic/commit/372872b46b401801f61265a4ae4aae437540af38).
- [ST usage on model main](https://huggingface.co/huggingface-mic/embeddinggemma-2-mic/blob/main/integrations/sentence_transformers/usage.md).
- Fleet: [`83b70d778c7d0a14661f058d43a3ce1c7bb39758`](https://huggingface.co/datasets/huggingface-mic/transformers-fleet/commit/83b70d778c7d0a14661f058d43a3ce1c7bb39758).
- Prototypes fleet `EmbeddingGemma` now advertises `transformers; sentence_transformers` and pins the model-main revision above. Only `prototypes/contracts.csv` and `prototypes/integrations.csv` changed.
- [Results Space](https://huggingface.co/spaces/huggingface-mic/sentence-transformers-ci).
- Nightly: daily **04:23 UTC**, selecting ST MICs from both fleets. Its separate publisher runs after success or failure, using the main-only `mic-publish` environment and `MIC_BASELINE_WRITE_TOKEN`.
- Notifications are not enabled; user said that is for later.

## Latest fleet run: numerical mismatch, automatically published

[Run 41810330232](https://github.com/huggingface/transformers-mic/actions/runs/41810330232)
ran with no preview overrides and successfully discovered EmbeddingGemma from the
shared fleet. It compared the expected four scores and reported:

```
scores: max delta 0.00511 over tolerance
```

The fixture job failed; `publish-results` succeeded and published the failure to
the Space. **Do not describe this run as passing.** No tolerance or baseline was
changed to hide the mismatch.

The run used newer upstream commits than the baseline recording:

| Dependency | Baseline recording | Latest fleet run |
| --- | --- | --- |
| Sentence Transformers | `b26925868810d978e06d731f6b527c2c74579b1b` | `3eeec5b5a912809e1034847f9d6d3d4a2d0a5f62` |
| Transformers | `f2754311602414ed561cdf8ed9d852390f514404` | `0c7f5ab8c579917b90bfa7502131e6420ec03978` |

Both recorded `torch==2.15.0+cpu`, `torchvision==0.29.2+cpu`, and `numpy==2.4.6`.
The cause of the numerical difference has not been isolated. Investigate the
upstream changes and runtime/hardware differences before proposing a baseline
update. Existing baseline review rules still apply.

## Successful earlier evidence

- [Recording 41797350961](https://github.com/huggingface/transformers-mic/actions/runs/41797350961): generated a new Linux candidate after two fresh-process executions.
- [Comparison 41807695456](https://github.com/huggingface/transformers-mic/actions/runs/41807695456): passed against that Linux baseline and was automatically published by Actions.
- Local suite: 416 passed. GitHub full runtime and dev-only Python 3.11/3.13 jobs passed before merge. Workflow lint and browser rendering checked.
- Before model-main publication, verified all existing Transformers usage and baseline files byte-for-byte unchanged, and every existing fixture file hash unchanged. Only the missing `1_Pooling/config.json` was added to the fixture. Weights are unchanged.
- The model-main contract files and repaired fixture hashes were revalidated after publication.

## Usage and implementation

The identifier is `sentence_transformers` (the user's original `sentencepiece`
was a typo). Generation currently supports `google/embeddinggemma-2`.

```sh
# On an existing built workspace:
tmic integrations --integration transformers --integration sentence_transformers
tmic record --integration sentence_transformers --all-variants
tmic review --integration sentence_transformers
tmic check --integration all --all-variants
```

This is a staged command; automated `tmic run`/fixture building still use the
Transformers path. The ST usage calls `encode_query`, `encode_document` on four
passages, and `similarity`; it converts the result to a finite NumPy `(4,)` array.
Each integration owns its assertions/baselines. ST does not inherit Transformers
known failures or patches. Publication checks every declared integration/variant.

Optional fleet integration metadata is bound to immutable model revisions.
Existing CSV columns remain unchanged. Corrected fixtures with nested module
assets require the updated runner's hashing support; existing Transformers
checkout dataset pins were not modified by this rollout.

## Local work and provenance

Worktree:
`/Users/tarek/Dev/transformers-ci-playbooks/mic/worktrees/sentence-transformers-integration`

Branch: `sentence-transformers-integration` (feature merged via squash).
Use `PYTHONPATH=src ../../checkout/.venv/bin/python` there: the shared editable
venv otherwise points at `mic/checkout`.

Useful local artifacts under this worktree:

- `work/nightly-41797350961/`: recording results and Linux candidate.
- `work/nightly-41810330232/`: failing fleet run results and exact dependencies.
- `work/st-rollout/plan.json`, `result.json`, `*.before.json`, `*.after.json`: rollout snapshots and immutable commit references.
- `work/st-rollout/prepare_rollout.py`, `publish_rollout.py`: audited promotion scripts. Do not rerun blindly; they enforce old parent revisions.

Original workspace preserved:
`mic/checkout/work/mic/google--embeddinggemma-2`.

Staged workspace:
`mic/checkout/work/sentence-transformers/embeddinggemma-2`.

The shared `mic/checkout` was switched by another task while this work was in
progress. It still contains earlier working-tree copies of these changes. Do not
reset/clean it indiscriminately; unrelated `docs/blog-post.md` must be preserved.

Preview branch retained for provenance: `sentence-transformers-preview` on the
model repo. Preview revision with Linux baseline:
`c91d7879a0d979efbcc6013b4c02df6c4f762b6a`. The main/fleet rollout no longer depends
on selecting that branch explicitly.
