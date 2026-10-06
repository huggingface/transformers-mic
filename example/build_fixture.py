"""Build the slim fixture in SmolLM-135M-slim/ from the pinned real checkpoint.

    python example/build_fixture.py

Shrinks only the model config and weights (2 layers, hidden size 64), trains
the fixture until it answers the contract's prompts, and copies the processor
files from the real checkpoint byte for byte. Requires the pinned snapshot in
the local HF cache (experiments/prefetch.py downloads it).
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "experiments"))
import experiment as exp  # noqa: E402

OUT = HERE / "SmolLM-135M-slim"
PROCESSOR_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
)
SEED = 17


def main():
    source = exp.snapshot(exp.SOURCES["text"])
    task = exp.TextTask(source)
    torch.manual_seed(SEED)
    model = task.make_model()
    training = exp.train(task, model, max_steps=500)
    if not training["converged"]:
        raise SystemExit(f"fixture did not converge: {training}")
    for file in OUT.glob("*.json"):
        file.unlink()
    model.save_pretrained(OUT)
    for name in PROCESSOR_FILES:
        shutil.copyfile(source / name, OUT / name)
    files = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(OUT.iterdir())
        if file.is_file() and file.name not in ("README.md", "build.json")
    }
    build = {
        "source": {"repo": exp.SOURCES["text"][0], "revision": exp.SOURCES["text"][1]},
        "seed": SEED,
        "training": training,
        "copied_unchanged": list(PROCESSOR_FILES),
        "sha256": files,
    }
    (OUT / "build.json").write_text(json.dumps(build, indent=1) + "\n")
    exp.log("built", **training)


if __name__ == "__main__":
    main()
