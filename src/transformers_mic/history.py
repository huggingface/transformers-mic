"""Read failure candidates from GitHub or a normalized relore export."""

import json
import subprocess


def collect(
    model, model_type=None, repo="huggingface/transformers", limit=50, imported=None
):
    if imported:
        with open(imported) as handle:
            rows = json.load(handle)
        if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
            raise ValueError(
                "History export must be a JSON list of issue/failure objects"
            )
        return {"source": "import", "model": model, "items": rows, "complete": False}
    terms = list(
        dict.fromkeys([model.split("/")[-1], model_type or model.split("/")[-1]])
    )
    found = {}
    for term in terms:
        for kind in ("issues", "prs"):
            command = [
                "gh",
                "search",
                kind,
                term,
                "--repo",
                repo,
                "--limit",
                str(limit),
                "--json",
                "number,title,body,url,state",
            ]
            result = subprocess.run(command, capture_output=True, text=True)
            if result.returncode:
                raise ValueError(
                    f"GitHub history query failed: {result.stderr.strip()}"
                )
            for row in json.loads(result.stdout):
                found[row["url"]] = {**row, "kind": kind, "matched": term}
    return {
        "source": "github",
        "repo": repo,
        "model": model,
        "queries": terms,
        "limit_per_query": limit,
        "complete": False,
        "items": list(found.values()),
    }
