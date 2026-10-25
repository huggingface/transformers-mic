"""Trusted publisher: validate downloaded measurements without executing their code."""

import json
import os
from pathlib import Path
import re
import sys

from transformers_mic.baselines import upload
from transformers_mic.cloud_progress import Progress
from transformers_mic.workspace import read, write


def main(root):
    root = Path(root).resolve()
    repo = os.environ["UPLOAD_REPO"]
    if not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo
    ) or repo.split("/")[0] not in os.environ["UPLOAD_NAMESPACES"].split(","):
        raise ValueError("Hub upload namespace not allowed")
    request = {
        "id": os.environ["JOB_ID"],
        "fleet_revision": os.environ["FLEET_REVISION"],
        "transformers_sha": os.environ["TRANSFORMERS_SHA"],
        "mic_sha": os.environ["MIC_SHA"],
        "fleets": os.environ["FLEETS"].split(","),
        "fixtures_only": os.environ["FIXTURES_ONLY"] == "true",
    }
    write(
        root / "cloud.json", {"job": request, "request": {"profiles": ["cpu", "cuda"]}}
    )
    progress = Progress(request["id"], "publish")
    progress.emit("uploading", 0, 1, repo)
    upload(root, repo, True)
    receipt = read(root / "cloud.json")["upload"]
    write(root / "publication.json", receipt)
    progress.emit("complete", 1, 1, repo)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main(sys.argv[1])
