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
    # A config can name a class Transformers does not have (Cosmos3-Nano's
    # Cosmos3ForConditionalGeneration): show the classes it maps the model_type to.
    classes = exported(config.get("architectures") or []) or auto_classes(config)
    if classes:
        result["transformers_classes"] = classes
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
            architectures = set(classes)
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
    names = {
        architecture: found
        for architecture in classes
        if (found := parameter_names(architecture, config))
    }
    if names:
        # Checkpoint keys are often remapped on load; without the real names the
        # planner guesses from the checkpoint layout and names missing parameters.
        result["parameter_names"] = names
    return result


def exported(names):
    """The names Transformers exports as model classes."""
    import transformers

    return [n for n in names if isinstance(getattr(transformers, n, None), type)]


def auto_classes(config, limit=3):
    """Head classes the Auto factories map this model_type to, for configs without exported architectures."""
    from transformers.models.auto import modeling_auto

    first = ("MODEL_MAPPING_NAMES", "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES")
    others = sorted(
        n
        for n in dir(modeling_auto)
        if n.startswith("MODEL_") and n.endswith("_MAPPING_NAMES")
    )
    found = []
    for name in (*first, *others):
        mapping = getattr(modeling_auto, name, None)
        cls = (
            mapping.get(config.get("model_type")) if isinstance(mapping, dict) else None
        )
        if isinstance(cls, str) and cls not in found:
            found.append(cls)
    return found[:limit]


def parameter_names(architecture, config, limit=80):
    """Loaded parameter names: the first block of each stack and the non-repeated modules."""
    import torch
    import transformers

    cls = getattr(transformers, architecture, None)
    if not isinstance(cls, type) or not issubclass(cls, transformers.PreTrainedModel):
        return None
    try:
        with torch.device("meta"):
            real = [
                n for n, _ in cls(cls.config_class.from_dict(config)).named_parameters()
            ]
    except Exception:  # advisory context; validation still checks the draft
        return None
    return sorted(n for n in real if ".0." in n or not re.search(r"\.\d+\.", n))[:limit]
