"""Opt-in usage paths on an existing MIC fixture."""

import json
from .workspace import Workspace, write

NAME = "sentence_transformers"
SUPPORTED = {"google/embeddinggemma-2"}
USAGE = """# Sentence Transformers: query and document similarity

```py runnable:similarity
import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import set_seed

set_seed(17)
model_id = "huggingface-mic/embeddinggemma-2-mic"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
revision = os.environ.get("MIC_MODEL_REVISION") or None  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{}"))  # doc-builder: hide
model = SentenceTransformer(
    model_id,
    revision=revision,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_kwargs=variant,
)
model.eval()
query = "Which planet is known as the Red Planet?"
passages = [
    "Venus is often called Earth's twin because of its similar size and proximity.",
    "Mars, known for its reddish appearance, is often referred to as the Red Planet.",
    "Jupiter, the largest planet in our solar system, has a prominent red spot.",
    "Saturn, famous for its rings, is sometimes mistaken for the Red Planet.",
]
query_emb = model.encode_query(query)
doc_emb = model.encode_document(passages)
scores = model.similarity(query_emb, doc_emb).detach().float().cpu().numpy().squeeze(0)
assert isinstance(scores, np.ndarray) and scores.shape == (4,)
assert np.isfinite(scores).all()
print(scores)
if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    print(json.dumps({"runtime": {"device": model.device.type, "dtype": str(model.dtype).removeprefix("torch.")}, "scores": scores.tolist()}))
```
"""


def generate(args):
    """Generate selected scenarios without running code or replacing fixture weights."""
    workspace = Workspace(args.workspace)
    selected = list(dict.fromkeys(args.integration))
    manifest = workspace.manifest
    if NAME in selected and workspace.state["model"] not in SUPPORTED:
        raise ValueError(
            "Sentence Transformers generation currently supports google/embeddinggemma-2"
        )
    if "transformers" in selected and "transformers" not in manifest["integrations"]:
        raise ValueError("Generate the Transformers usage with tmic create first")
    workspace.snapshot()
    if NAME in selected and NAME not in manifest["integrations"]:
        entry = f"integrations/{NAME}/usage.md"
        manifest["integrations"][NAME] = {
            "entry_point": entry,
            "format": "runnable-markdown",
            "requires": ["sentence-transformers", "torch", "numpy"],
            "targets": ["fixtures"],
            "variants": {"default": {}},
            "training": False,
            "outputs": {
                "runtime": {"compare": "exact"},
                "scores": {"compare": "allclose", "atol": 0.0001, "rtol": 0.0001},
            },
        }
        path = workspace.contract_root / entry
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(USAGE)
    manifest["integrations"] = {
        name: manifest["integrations"][name] for name in selected
    }
    write(workspace.contract_root / "integration.json", manifest)
    workspace.update_layout()
    lock = workspace.lock
    lock["contracts"][workspace.state["contract"]]["integrations"] = selected
    write(workspace.lock_path, lock)
    workspace.save_state(validated=False, validations={})
    print(json.dumps({"integrations": selected, "workspace": str(workspace.root)}))
    return 0
