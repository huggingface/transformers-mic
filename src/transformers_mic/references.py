"""Give the planner public model-card examples and local API signatures."""

import ast
import importlib.util
from pathlib import Path
import re

from huggingface_hub import hf_hub_download


def collect(model, revision, config):
    result = {}
    try:
        card = Path(hf_hub_download(model, "README.md", revision=revision)).read_text()
        blocks = re.findall(r"```(?:python|py)\s*\n(.*?)```", card, re.S)
        result["model_card_examples"] = "\n\n".join(blocks)[:18000]
    except (OSError, ValueError):
        pass
    spec = importlib.util.find_spec("transformers")
    if spec and spec.submodule_search_locations:
        family = config.get("model_type", "").replace("-", "_")
        path = (
            Path(next(iter(spec.submodule_search_locations)))
            / "models"
            / family
            / f"modeling_{family}.py"
        )
        if path.exists():
            tree = ast.parse(path.read_text())
            architectures = set(config.get("architectures", []))
            api = {}
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name in architectures:
                    methods = {}
                    for method in node.body:
                        if isinstance(method, ast.FunctionDef) and method.name in (
                            "forward",
                            "generate",
                        ):
                            methods[method.name] = {
                                "signature": ast.unparse(method.args),
                                "doc": (ast.get_docstring(method) or "")[:6000],
                            }
                    api[node.name] = methods
            result["architecture_api"] = api
    return result
