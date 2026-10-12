"""Download the pinned Hub snapshots the experiments read, for a fresh machine.

    python prefetch.py

The experiments themselves run offline (HF_HUB_OFFLINE=1). OLMo is only used
for its config and tokenizer, so its 1B weights are skipped.
"""

import os

os.environ["HF_HUB_OFFLINE"] = "0"

from huggingface_hub import snapshot_download  # noqa: E402

import experiment as exp  # noqa: E402

CONFIG_ONLY = {"olmo"}

for family, (repo, revision, _) in exp.SOURCES.items():
    patterns = ["*.json", "*.txt", "*.model"] if family in CONFIG_ONLY else None
    path = snapshot_download(repo, revision=revision, allow_patterns=patterns)
    exp.log("prefetched", family=family, repo=repo, path=path)
repo, revision, _ = exp.CLIP_REFERENCE
exp.log("prefetched", family="clip_reference", path=snapshot_download(repo, revision=revision))
