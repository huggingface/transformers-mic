"""User workflows; expected values are produced only by the execution backend."""

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import shutil
import tempfile

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
import jsonschema

from . import backend, fleet, history, planner, policy, references, resources
from .hub import (
    DEFAULT_NAMESPACE,
    canonical_references,
    umbrella_references,
    upgrade_contract,
)
from .workspace import Workspace, catalog_patches, digest, read, write


def manifest_schema():
    return read(resources.backend() / "schemas/integration.schema.json")


def validate_plan(plan, model, fixture, old=None):
    jsonschema.validate(plan, planner.PLAN_SCHEMA)
    manifest = plan["manifest"]
    jsonschema.validate(manifest, manifest_schema())
    if set(manifest["checkpoints"]) != {model} or set(manifest["fixtures"]) != {
        fixture
    }:
        raise ValueError(
            "Generated plan must contain exactly the requested checkpoint and fixture"
        )
    integration = manifest["integrations"]["transformers"]
    if (
        integration["entry_point"] != "integrations/transformers/usage.md"
        or integration["format"] != "runnable-markdown"
    ):
        raise ValueError(
            "Generated entry point must be integrations/transformers/usage.md, runnable-markdown"
        )
    recipe = manifest["fixtures"][fixture].get("build")
    if not isinstance(recipe, dict):
        raise ValueError("Generated fixture needs a build recipe")
    previous_recipe = old["fixtures"][fixture].get("build", {}) if old else {}
    chosen = [
        key
        for key in (
            "overfit",
            "accept_insensitive",
            "accept_unhealthy",
            "copy_assets",
            "preserve_frozen_prefixes",
        )
        if recipe.get(key) != previous_recipe.get(key)
    ]
    if chosen:
        # Name the keys so a repair attempt knows what to remove.
        raise ValueError(
            f"The LLM cannot select overfit, exempt insensitive inputs, or preserve source "
            f"tensors/assets; remove {', '.join(chosen)} from the build recipe"
        )
    if (
        manifest["checkpoints"][model]["fixture"] != fixture
        or manifest["fixtures"][fixture]["built_from"] != model
    ):
        raise ValueError(
            "Generated fixture must be built from and assigned to the requested checkpoint"
        )
    for rule in manifest["outputs"].values():
        if rule["compare"] == "allclose" and (
            rule["atol"] != 1e-4 or rule["rtol"] != 1e-4
        ):
            raise ValueError("Generated numeric tolerances must be atol=rtol=1e-4")
    if old and not set(old["outputs"]).issubset(manifest["outputs"]):
        raise ValueError("Update removes existing output assertions")
    if old and any(
        manifest["outputs"][key] != rule for key, rule in old["outputs"].items()
    ):
        raise ValueError(
            "Update changes existing comparison rules; review those changes manually"
        )
    if not {"default", "eager", "bfloat16"}.issubset(integration.get("variants", {})):
        raise ValueError(
            "Generated contract must declare default, eager and bfloat16 variants"
        )
    if old:
        previous = old["integrations"]["transformers"]
        if previous.get("training") and not integration.get("training"):
            raise ValueError("Update disables existing training assertions")
        if not set(previous["targets"]).issubset(integration["targets"]):
            raise ValueError("Update removes existing target types")
        if any(
            integration["variants"].get(key) != value
            for key, value in previous["variants"].items()
        ):
            raise ValueError("Update changes existing variants")
    if integration.get("training"):
        required = {
            "training.loss": "allclose",
            "training.gradients": "all_true",
            "training.updated": "true",
        }
        if any(
            manifest["outputs"].get(key, {}).get("compare") != rule
            for key, rule in required.items()
        ):
            raise ValueError(
                "Training contracts require training.loss (allclose), training.gradients (all_true nonempty named booleans), and training.updated (true)"
            )
    if integration.get("variants", {}).get("bfloat16") != {"dtype": "bfloat16"}:
        raise ValueError("Use bfloat16 variant kwargs {dtype: bfloat16}")
    # Syntax is checked without importing or running generated code.
    blocks = re.findall(
        r"```(?:py|python) runnable:[^\n]*\n(.*?)```", plan["usage"], re.S
    )
    if not blocks:
        raise ValueError("Generated usage contains no runnable Python blocks")
    ast.parse("\n".join(blocks))
    leaked = visible_plumbing(blocks)
    if leaked:
        raise ValueError(
            "usage.md is published documentation: runner plumbing (variant, MIC_*, "
            "os.environ) must stay on '# doc-builder: hide' lines or inside a hidden "
            f"block, but these lines are visible: {leaked[:5]}"
        )
    return manifest


def visible_plumbing(blocks):
    """Lines a reader would see that depend on the MIC runner."""
    leaked = []
    for block in blocks:
        hidden_indent = None
        for line in block.splitlines():
            indent = len(line) - len(line.lstrip())
            if hidden_indent is not None and (
                not line.strip() or indent > hidden_indent
            ):
                continue
            hidden_indent = None
            if "# doc-builder: hide" in line:
                if line.split("#", 1)[0].rstrip().endswith(":"):
                    hidden_indent = indent
                continue
            if re.search(r"\bvariant\b|MIC_|os\.environ", line):
                leaked.append(line.strip())
    return leaked


def check_imports(plan):
    """Reject usage that imports names Transformers does not export."""
    import ast

    imported = []
    for block in re.findall(r"^```py[^\n]*\n(.*?)^```", plan["usage"], re.M | re.S):
        try:
            tree = ast.parse(block)
        except SyntaxError:  # a continuation block; the build reports real errors
            continue
        imported += [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "transformers"
            for alias in node.names
        ]
    if not imported:
        return
    import difflib

    import transformers

    missing = [n for n in imported if not hasattr(transformers, n)]
    if missing:
        known = [n for n in dir(transformers) if not n.startswith("_")]
        hints = {n: difflib.get_close_matches(n, known, n=3) for n in missing}
        raise ValueError(f"usage imports names transformers does not export: {hints}")


def check_targets(plan):
    """A drafted contract measures its own fixture: PR CI runs fixtures only."""
    targets = plan["manifest"]["integrations"]["transformers"].get("targets", [])
    if len(set(targets)) != len(targets):
        raise ValueError(f"targets repeat an entry: {targets}")
    if plan["manifest"].get("fixtures") and "fixtures" not in targets:
        raise ValueError(
            f"targets {targets} omit 'fixtures' although the manifest declares a "
            "fixture; use ['fixtures', 'checkpoints'] (or ['fixtures'])"
        )


CPU_PARAMETER_LIMIT = 1_000_000_000


def keep_large_checkpoints_off_cpu(manifest, parameters):
    """A checkpoint above ~1B parameters runs on GPU only, and only in bfloat16.

    The CPU runner is a memory-limited container: a float32 training step on a
    2B checkpoint (gemma-2-2b-it, SmolLM-1.7B) kills it with exit 137, and the
    artifact upload dies with it, so the failure has no log. On the 22 GB A10G a
    float32 step on gemma-2b (weights + gradients) runs out of memory too, so the
    float32 variants stay on the slim fixture.
    """
    integration = manifest["integrations"]["transformers"]
    resources = integration.setdefault("resources", {})
    full_precision = [
        name
        for name, kwargs in integration.get("variants", {}).items()
        if (kwargs or {}).get("dtype") != "bfloat16"
    ]
    if len(full_precision) == len(integration.get("variants", {})):
        full_precision = []  # no bfloat16 variant to keep
    changed = []
    for repo in manifest.get("checkpoints", {}):
        if (parameters.get(repo) or 0) <= CPU_PARAMETER_LIMIT:
            continue
        spec = resources.setdefault(repo, {})
        before = json.dumps(spec, sort_keys=True)
        if "cpu" in spec.get("devices", ["cpu", "cuda"]):
            spec["devices"] = ["cuda"]
        for name in full_precision:
            spec.setdefault("variants", {}).setdefault(name, {})["devices"] = []
        if json.dumps(spec, sort_keys=True) != before:
            changed.append(repo)
    if any((parameters.get(repo) or 0) > CPU_PARAMETER_LIMIT for repo in manifest.get("checkpoints", {})):
        # Such a checkpoint is loaded visibly in bfloat16 (check_float16_load), so a
        # variant without kwargs would never reload: the slim fixture keeps its float32
        # coverage only if the full-precision variants say so.
        for name in full_precision:
            kwargs = integration["variants"][name] or {}
            if "dtype" not in kwargs:
                integration["variants"][name] = {"dtype": "float32", **kwargs}
    return changed


def parameter_count(model, revision):
    info = HfApi().model_info(model, revision=revision, expand=["safetensors"])
    return getattr(info.safetensors, "total", None)


def check_partial_training(plan, context):
    """Above ~3B parameters the real checkpoint's training step must freeze most of it."""
    count = context.get("parameters") or 0
    if count > 3e9 and not context.get("fixtures_only") and "requires_grad_(" not in plan["usage"]:
        raise ValueError(
            f"the checkpoint has {count / 1e9:.1f}B parameters: a full bfloat16 training step "
            "(weights plus gradients) does not fit the 22 GB GPU; freeze the model with "
            "requires_grad_(False) and train a small meaningful subset"
        )


def check_undefined_names(plan):
    """Reject usage that reads a name no block sharing its state ever defines."""
    import builtins
    import symtable

    groups = {}
    for name, block in re.findall(
        r"^```py runnable:([\w-]+)[^\n]*\n(.*?)^```", plan["usage"], re.M | re.S
    ):
        groups[name] = groups.get(name, "") + block
    missing = set()
    for code in groups.values():
        try:
            top = symtable.symtable(code, "usage", "exec")
        except SyntaxError:  # the build reports real syntax errors
            continue
        known = set(dir(builtins)) | {
            s.get_name()
            for s in top.get_symbols()
            if s.is_assigned() or s.is_imported() or s.is_namespace()
        }
        tables = [top]
        while tables:
            table = tables.pop()
            tables += table.get_children()
            for symbol in table.get_symbols():
                read = symbol.is_referenced() and (
                    symbol.is_global()
                    if table is not top
                    else not (symbol.is_assigned() or symbol.is_imported())
                )
                if read and symbol.get_name() not in known:
                    missing.add(symbol.get_name())
    if missing:
        raise ValueError(f"usage reads names it never defines: {sorted(missing)}")


def check_model_classes(plan, config):
    """Reject a fixture model_class or Auto class that cannot load this model type."""
    import transformers
    from transformers.models.auto import modeling_auto

    model_type = config.get("model_type")
    supported = sorted(
        {
            (name.removesuffix("_MAPPING_NAMES"), mapping[model_type])
            for name in dir(modeling_auto)
            if name.startswith("MODEL") and name.endswith("_MAPPING_NAMES")
            for mapping in [getattr(modeling_auto, name)]
            if isinstance(mapping, dict) and isinstance(mapping.get(model_type), str)
        }
    )
    if not supported:  # a type Transformers does not map: nothing to compare with
        return
    hint = f"classes that load {model_type!r}: {sorted({c for _, c in supported})}"
    for name, spec in plan["manifest"].get("fixtures", {}).items():
        cls = (spec.get("build") or {}).get("model_class")
        if cls and not hasattr(transformers, cls):
            raise ValueError(f"fixture model_class {cls!r} is not a Transformers class; {hint}")
    for auto in sorted(set(re.findall(r"\b(AutoModel\w*)\.from_pretrained", plan["usage"]))):
        mapping = getattr(getattr(transformers, auto, None), "_model_mapping", None)
        names = getattr(mapping, "_model_mapping", None)
        if isinstance(names, dict) and model_type not in names:
            raise ValueError(f"{auto} does not map {model_type!r}; {hint}")


def check_code_fences(plan):
    """Every Python block runs: a plain ```python fence is skipped by the runner."""
    plain = re.findall(r"^```(?:python|py)[ \t]*$", plan["usage"], re.M)
    if plain:
        raise ValueError(
            f"{len(plain)} code block(s) are not runnable; fence every block as "
            "```py runnable:<name> (continuations ```py runnable:<name>:2)"
        )


def check_no_remote_code(plan):
    """The contract tests Transformers' own implementation, never a checkpoint's remote code."""
    if re.search(r"trust_remote_code\s*=\s*True", plan["usage"]):
        raise ValueError(
            "usage passes trust_remote_code=True; load the class Transformers maps for "
            "the model_type and use its public API (a checkpoint's auto_map is ignored)"
        )


def check_image_shapes(plan):
    """Reject the broadcast-image idiom that builds (H, W, 1) instead of (3, H, W).

    view(H, 1, 1) + view(1, W, 1) is (H, W, 1); .expand(3, H, W) then raises. Three
    drafts in a row (DINOv3, ResNet, Swinv2) did this despite the prompt bullet.
    """
    usage = plan["usage"]
    rows = re.search(r"\.view\(\s*\w+\s*,\s*1\s*,\s*1\s*\)", usage)
    cols = re.search(r"\.view\(\s*1\s*,\s*\w+\s*,\s*1\s*\)", usage)
    if rows and cols and re.search(r"\.expand\(\s*3\s*,", usage):
        raise ValueError(
            "view(H, 1, 1) + view(1, W, 1) is (H, W, 1), and .expand(3, H, W) raises: "
            "build images as view(1, H, 1) + view(1, 1, W), or as (H, W, 3) uint8 arrays"
        )


def check_loss_has_gradients(plan):
    """Reject a loss whose dependencies never call the model with gradients.

    A target computed under no_grad is fine; a loss built only from tensors computed
    under no_grad / inference_mode has no gradient path into the model.

    The prompt says so; EmbeddingGemma2 and DINOv2-with-registers drafts still
    trained on their inference-block features and failed with no grad_fn or with
    every gradient False.
    """
    import ast

    groups = {}
    for name, block in re.findall(
        r"^```py runnable:([\w-]+)[^\n]*\n(.*?)^```", plan["usage"], re.M | re.S
    ):
        groups[name] = groups.get(name, "") + block

    def without_grad(node):
        return any(
            isinstance(item.context_expr, ast.Call)
            and getattr(item.context_expr.func, "attr", "") in ("no_grad", "inference_mode")
            for item in node.items
        )

    for code in groups.values():
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        frozen, deps, calls = set(), {}, set()

        def calls_model(node):
            return any(
                isinstance(n, ast.Call)
                and isinstance(n.func, (ast.Name, ast.Attribute))
                and (getattr(n.func, "id", None) or getattr(getattr(n.func, "value", None), "id", None)) == "model"
                for n in ast.walk(node)
            )

        def visit(statements, inside):
            for statement in statements:
                if isinstance(statement, ast.With):
                    visit(statement.body, inside or without_grad(statement))
                    continue
                if isinstance(statement, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                    targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                    names = {n.id for t in targets for n in ast.walk(t) if isinstance(n, ast.Name)}
                    read = {n.id for n in ast.walk(statement.value) if isinstance(n, ast.Name)}
                    for name in names:
                        if inside:
                            frozen.add(name)
                        else:
                            frozen.discard(name)
                            if calls_model(statement.value):
                                calls.add(name)
                            else:
                                calls.discard(name)
                            deps[name] = read - {name} | (deps.get(name, set()) if isinstance(statement, ast.AugAssign) else set())
                for field in ("body", "orelse"):
                    if isinstance(getattr(statement, field, None), list) and not isinstance(statement, ast.With):
                        visit(getattr(statement, field), inside)

        visit(tree.body, False)
        if "loss" not in deps:
            continue
        seen, stack, stale = set(), ["loss"], []
        while stack:
            name = stack.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in frozen:
                stale.append(name)
            stack += deps.get(name, ())
        if stale and not seen & calls:
            raise ValueError(
                f"the training loss only uses {sorted(stale)}, computed under torch.no_grad(): "
                "recompute the model outputs with gradients in the training block"
            )


def loaded_parameters(context):
    """The checkpoint size the contract loads: none for a fixtures-only contract (Kimi K2.5's 1T)."""
    return None if context.get("fixtures_only") else context.get("parameters")


def check_float16_load(plan, config, parameters=None):
    """Load the model in a dtype that trains and fits the GPU.

    Above 1B the contract is measured in bfloat16 only, and a float32 copy does not
    fit the GPU before the hidden bfloat16 reload (Aya Expanse 8B stored in float16,
    Audio Flamingo 3 stored in float32: about 33 GB), so it loads in bfloat16.
    Below, a float16 checkpoint overflows without loss scaling: it loads in float32.
    """
    stored = config.get("dtype") or config.get("torch_dtype")
    if (parameters or 0) > CPU_PARAMETER_LIMIT:
        wanted = "bfloat16"
    elif stored == "float16":
        wanted = "float32"
    else:
        return
    loads = re.findall(r"^[^#\n]*\b(\w+)\.from_pretrained\(model_id[^\n]*$", plan["usage"], re.M)
    lines = re.findall(r"^[^#\n]*\b\w+\.from_pretrained\(model_id[^\n]*$", plan["usage"], re.M)
    for name, line in zip(loads, lines):
        if re.search(r"Processor|Tokenizer|Extractor", name) or f"torch.{wanted}" in line:
            continue
        raise ValueError(
            f"load {name} with dtype=torch.{wanted}: "
            + ("above 1B parameters the contract is measured in bfloat16 only and float32 does not fit the GPU"
               if wanted == "bfloat16" else "the checkpoint is stored in float16 (the bfloat16 variant still overrides it)")
        )


def check_label_arguments(plan, config):
    """Reject model(..., labels=) on a class whose forward takes no labels.

    Most forwards swallow unknown keywords, so the mistake surfaces at build time
    as a None loss (Owlv2ForObjectDetection has no loss at all; Mask2Former takes
    mask_labels and class_labels).
    """
    import inspect

    import transformers

    if not re.search(r"\bmodel\([^)]*\blabels=", plan["usage"]):
        return
    model_type = config.get("model_type")
    names = set(re.findall(r"\b(\w+)\.from_pretrained", plan["usage"]))
    classes = []
    for name in sorted(names):
        cls = getattr(transformers, name, None)
        mapping = getattr(getattr(cls, "_model_mapping", None), "_model_mapping", None)
        if isinstance(mapping, dict):  # an Auto class: the class it loads for this type
            cls = getattr(transformers, mapping.get(model_type) or "", None)
        if isinstance(cls, type) and issubclass(cls, transformers.PreTrainedModel):
            classes.append(cls)
    params = {cls.__name__: inspect.signature(cls.forward).parameters for cls in classes}
    if params and not any("labels" in p for p in params.values()):
        found = {name: [p for p in ps if "label" in p] for name, ps in params.items()}
        raise ValueError(
            f"usage passes labels= but the forward takes no labels argument (label "
            f"arguments: {found}); pass those, or compute the loss from the outputs"
        )


def check_parameter_names(plan, config):
    """Reject usage that names parameters the architecture does not have."""
    names = set(
        re.findall(r"[\"']([A-Za-z_][\w.]*\.(?:weight|bias))[\"']", plan["usage"])
    )
    if not names:
        return
    import torch
    import transformers

    cls = getattr(transformers, plan["manifest"].get("architecture", ""), None)
    if not isinstance(cls, type) or not issubclass(cls, transformers.PreTrainedModel):
        return
    try:
        with torch.device("meta"):
            real = {
                n for n, _ in cls(cls.config_class.from_dict(config)).named_parameters()
            }
    except Exception:  # inspection is advisory; the build still runs the contract
        return
    missing = sorted(names - real)
    if missing:
        import difflib

        # Checkpoint keys (language_model.model.layers...) are remapped on load;
        # name the closest loaded parameter for each miss.
        def nearest(name):
            suffix = "." + ".".join(name.split(".")[-2:])
            same = [r for r in real if ("." + r).endswith(suffix)] or real
            return (difflib.get_close_matches(name, same, n=1, cutoff=0.0) or ["?"])[0]

        closest = {name: nearest(name) for name in missing}
        # The first block and the non-repeated modules show the real naming scheme.
        hints = sorted(n for n in real if ".0." in n or not re.search(r"\.\d+\.", n))[
            :60
        ]
        raise ValueError(
            f"Usage names parameters {cls.__name__} does not have after loading "
            f"(checkpoint keys are remapped on load). Closest loaded names: {closest}; "
            f"pick the right module from the real names: {hints}"
        )


def check_config_facts(plan, config):
    """Overrides must name real config fields; requires gains the model's imports."""
    full = config
    try:  # config.json omits defaults; the config class knows every field
        from transformers import CONFIG_MAPPING

        full = CONFIG_MAPPING[config["model_type"]].from_dict(config).to_dict()
    except Exception:
        pass
    for name, spec in plan["manifest"].get("fixtures", {}).items():
        fields = (spec.get("build") or {}).get("config_overrides", {}).get("fields", {})
        for path in fields:
            node = full
            for part in path.split(".")[:-1]:
                node = node.get(part) if isinstance(node, dict) else None
            if not isinstance(node, dict) or path.split(".")[-1] not in node:
                subs = sorted(k for k, v in full.items() if isinstance(v, dict))
                raise ValueError(
                    f"config_overrides field {path!r} is not in the checkpoint config; "
                    f"nested configs need their prefix, one of {subs}"
                )
        # The fixture runs the checkpoint's own tokenizer and shrink keeps its
        # vocabulary; a smaller vocab_size makes real token ids out of range.
        for path in [p for p in fields if p.split(".")[-1] == "vocab_size"]:
            node = full
            for part in path.split("."):
                node = node.get(part) if isinstance(node, dict) else None
            if isinstance(node, int) and fields[path] < node:
                del fields[path]
    integration = plan["manifest"]["integrations"]["transformers"]
    requires = " ".join(integration.get("requires", []))
    text = json.dumps(config)
    needed = [
        package
        for package, present in (
            ("timm", '"timm_wrapper"' in text),
            ("torchvision", "vision_config" in config),
        )
        if present and package not in requires
    ]
    if "device_map" in plan.get("usage", "") and "accelerate" not in requires:
        needed.append("accelerate")
    if sentencepiece_tokenizer(config.get("model_type")):
        needed += [p for p in ("sentencepiece", "protobuf") if p not in requires]
    # The config and usage determine these packages, so add them rather than
    # spending a planner retry on a fact the draft cannot get wrong twice.
    if needed:
        integration["requires"] = [*integration.get("requires", []), *needed]


def sentencepiece_tokenizer(model_type):
    """Whether the model type's tokenizer is SentencePiece-backed (e.g. SiglipTokenizer)."""
    try:
        import transformers
        from transformers.models.auto.tokenization_auto import TOKENIZER_MAPPING_NAMES
    except ImportError:
        return False
    names = TOKENIZER_MAPPING_NAMES.get(model_type) or ()
    for name in names if isinstance(names, (tuple, list)) else (names,):
        cls = getattr(transformers, name, None) if name else None
        if cls is None:
            continue
        # Without sentencepiece installed the class is a dummy naming its backends.
        if "sentencepiece" in (getattr(cls, "_backends", None) or ()):
            return True
        if any(base.__name__ == "SentencePieceBackend" for base in getattr(cls, "__mro__", ())):
            return True
    return False


def propose_valid(context, args, model, fixture, old=None):
    # A provider can emit schema-valid JSON that violates runner conventions.
    # Return validation feedback for one repair; never execute an invalid draft.
    for attempt in range(3):
        plan = None
        try:
            plan, llm = planner.propose(
                context,
                model=args.llm_model,
                provider=args.provider,
                bill_to=args.bill_to,
            )
            if isinstance(plan.get("name"), str):  # "Qwen3.5-0.8B" is a slug away from valid
                plan["name"] = re.sub(r"[^A-Za-z0-9_-]", "-", plan["name"])
            validate_plan(plan, model, fixture, old)
            check_imports(plan)
            check_undefined_names(plan)
            check_targets(plan)
            check_code_fences(plan)
            check_no_remote_code(plan)
            check_image_shapes(plan)
            check_loss_has_gradients(plan)
            if context.get("config"):
                check_float16_load(plan, context["config"], loaded_parameters(context))
                check_config_facts(plan, context["config"])
                check_model_classes(plan, context["config"])
                check_label_arguments(plan, context["config"])
                check_parameter_names(plan, context["config"])
            check_partial_training(plan, context)
            return plan, llm
        except (ValueError, jsonschema.ValidationError) as exc:
            if attempt == 2:
                raise
            context = {
                **context,
                "previous_draft": plan,
                "validation_error": str(exc)[:3000],
                "repair_request": "Fix the validation error and return the complete corrected draft.",
            }


def model_context(model, revision=None):
    api = HfApi()
    info = api.model_info(model, revision=revision)
    if info.gated:
        print(
            f"tmic: {model} is gated: cloud and nightly measurement of the checkpoint "
            "need the HF_HUB_READ_TOKEN secret to belong to an account that accepted "
            "its license; fixtures are unaffected",
            file=sys.stderr,
        )
    config = read(hf_hub_download(model, "config.json", revision=info.sha))
    return info.sha, config


def create(args):
    workspace = Workspace(args.out)
    workspace.ensure_new()
    catalog = read(resources.catalog_lock())
    names = [
        n for n, c in catalog["contracts"].items() if args.model in c["checkpoints"]
    ]
    if args.contract:
        names = [n for n in names if n == args.contract]
    if names and args.regenerate and not args.llm:
        name = names[0]
        original = read(
            resources.catalog() / "hub" / f"{name}-integration/integration.json"
        )
        old_fixture = original["checkpoints"][args.model]["fixture"]
        namespace = args.namespace or DEFAULT_NAMESPACE
        slug = args.name or re.sub(r"[^A-Za-z0-9_-]", "-", args.model.split("/")[-1])
        fixture = f"{namespace}/{slug}-mic"
        manifest = json.loads(json.dumps(original).replace(old_fixture, fixture))
        manifest["checkpoints"] = {args.model: {"fixture": fixture}}
        manifest["fixtures"] = {fixture: manifest["fixtures"][fixture]}
        manifest["fixtures"][fixture].update(built_from=args.model, covers=[args.model])
        if args.fixtures_only:
            manifest["integrations"]["transformers"]["targets"] = ["fixtures"]
        pin = catalog["contracts"][name]["checkpoints"][args.model]
        source = resources.catalog() / "hub" / f"{name}-integration"
        usage = (
            source / manifest["integrations"]["transformers"]["entry_point"]
        ).read_text()
        workspace.from_plan(
            args.model,
            pin,
            namespace,
            slug,
            manifest,
            usage,
            args.weights,
            args.readable_output,
        )
        # Preserve frozen assets used by the catalog usage (e.g. speech).
        for item in source.iterdir():
            if item.is_dir() and item.name != "integrations":
                shutil.copytree(
                    item, workspace.contract_root / item.name, dirs_exist_ok=True
                )
        workspace.save_state(origin="catalog-regenerated")
        lock = workspace.lock
        lock["local_patches"] = catalog_patches(catalog, name)
        write(workspace.lock_path, lock)
        write(
            workspace.root / ".mic/plan.json",
            {
                "origin": "catalog-regenerated",
                "source_contract": name,
                "scope": "fixtures" if args.fixtures_only else "all",
                "manifest": manifest,
            },
        )
    elif names and not args.llm:
        name = names[0]
        workspace.from_catalog(args.model, name, args.weights, args.readable_output)
        write(
            workspace.root / ".mic/plan.json",
            {
                "origin": "catalog",
                "contract": name,
                "checkpoint_pins": workspace.lock["contracts"][name]["checkpoints"],
                "decision": "Reuse published usage, fixtures and expectations; no LLM needed.",
            },
        )
        if not args.offline and not args.no_history:
            metadata = read(
                workspace.root
                / "example/hub"
                / workspace.fixture.split("/")[-1]
                / "build.json"
            )
            write(
                workspace.root / ".mic/history.json",
                history.collect(
                    args.model,
                    metadata.get("key_fields", {}).get("model_type"),
                    args.history_repo,
                    args.limit,
                    args.history_file,
                ),
            )
    else:
        if args.offline:
            raise ValueError(
                "No catalog contract matches this model; new models need Hub metadata and an LLM usage plan"
            )
        revision, config = model_context(args.model, args.revision)
        namespace = args.namespace or DEFAULT_NAMESPACE
        slug = re.sub(r"[^A-Za-z0-9_.-]", "-", args.model.split("/")[-1])
        fixture = f"{namespace}/{args.name or slug}-mic"
        evidence = (
            {"items": [], "complete": False, "source": "not requested"}
            if args.no_history
            else history.collect(
                args.model,
                config.get("model_type"),
                args.history_repo,
                args.limit,
                args.history_file,
            )
        )
        context = {
            "model": args.model,
            "revision": revision,
            "config": config,
            "fixture_repo": fixture,
            "manifest_schema": manifest_schema(),
            "failure_candidates": evidence["items"][:30],
            "parameters": parameter_count(args.model, revision),
            "fixtures_only": bool(args.fixtures_only),
            "request": "Design a small useful integration path; include inference and a training step where supported.",
        }
        context.update(references.collect(args.model, revision, config))
        sample = resources.catalog() / "hub/BERT-integration"
        context["example_plan"] = {
            "name": "BERT",
            "manifest": read(sample / "integration.json"),
            "usage": (sample / "integrations/transformers/usage.md").read_text(),
            "rationale": "Use the model's public API and measure expectations by execution.",
            "historical_candidates": [],
        }
        plan, llm = propose_valid(context, args, args.model, fixture)
        manifest = plan["manifest"]
        if args.name:
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", args.name):
                raise ValueError("Contract name must be a directory slug")
            plan["name"] = args.name
        if args.fixtures_only:
            manifest["integrations"]["transformers"]["targets"] = ["fixtures"]
        for repo in keep_large_checkpoints_off_cpu(
            manifest, {args.model: context["parameters"]}
        ):
            print(f"tmic: {repo} is above 1B parameters: measuring it on GPU, in bfloat16 only", file=sys.stderr)
        workspace.from_plan(
            args.model,
            revision,
            namespace,
            plan["name"],
            manifest,
            plan["usage"],
            args.weights,
            args.readable_output,
        )
        write(workspace.root / ".mic/plan.json", {**plan, "llm": llm})
        write(workspace.root / ".mic/history.json", evidence)
    workspace.update_layout()
    print(
        json.dumps(
            {
                "workspace": str(workspace.root),
                "contract": workspace.state["contract"],
                "origin": workspace.state["origin"],
                "next": [
                    "tmic build",
                    "tmic record",
                    "tmic review",
                    "tmic check",
                    "tmic publish",
                ],
            }
        )
    )
    if args.build or args.record:
        args.workspace = str(workspace.root)
        args.rebuild = False
        status = build(args)
        if status:
            return status
    if args.record:
        args.workspace = str(workspace.root)
        args.variant, args.isolate, args.target = "default", False, None
        return execute(args, record=True)
    return 0


def research(args):
    workspace = Workspace(args.workspace)
    result = history.collect(
        workspace.state["model"],
        args.model_type,
        args.history_repo,
        args.limit,
        args.history_file,
    )
    write(workspace.root / ".mic/history.json", result)
    print(
        json.dumps(
            {
                "candidates": len(result["items"]),
                "complete": False,
                "path": str(workspace.root / ".mic/history.json"),
            }
        )
    )
    return 0


def build(args):
    workspace = Workspace(args.workspace)
    state, lock, manifest = workspace.state, workspace.lock, workspace.manifest
    fixture = workspace.fixture
    pin = lock["contracts"][state["contract"]]["fixtures"][fixture]
    requested = args.weights if args.weights else state["weights"]
    readable = args.readable_output or state["readable_output"]
    if "fixtures" not in manifest["integrations"]["transformers"]["targets"]:
        choice = {
            "weights": None,
            "accepted": True,
            "reason": "Checkpoint-only path; the manifest explicitly excludes slim execution.",
        }
        workspace.save_state(weight_decision=choice)
        print(json.dumps(choice))
        return 0
    local_metadata = (workspace.lock_path.parent / pin["path"]).resolve() / "build.json"
    if not args.rebuild and (
        pin["revision"] != "unpublished"
        or (pin.get("files_sha256") and local_metadata.exists())
    ):
        metadata = read(local_metadata)
        choice = policy.decision(metadata, requested, readable, reused=True)
        if choice["accepted"]:
            workspace.save_state(weight_decision=choice)
            print(json.dumps(choice))
            return 0
    original_manifest = workspace.manifest
    recipe = manifest["fixtures"][fixture].get("build")
    if not recipe:
        raise ValueError(
            "This legacy fixture has no build recipe; add one before rebuilding"
        )
    if requested == "overfit" or readable:
        recipe["overfit"] = (
            "Explicit CLI overfit/readable-output requirement; historical detection still needs replay validation."
        )
    elif requested == "random":
        recipe.pop("overfit", None)
    write(workspace.contract_root / "integration.json", manifest)
    with tempfile.TemporaryDirectory(
        prefix="build-", dir=workspace.root / ".mic"
    ) as temp:
        out = Path(temp) / "fixture"
        if local_metadata.exists() and not getattr(args, "reshrink", False):
            # The builder reuses the shrunk config this record holds when the
            # recipe is unchanged, so a rebuild does not depend on today's rule.
            out.mkdir()
            shutil.copy(local_metadata, out / "build.json")
        # Explicit random builds must not silently fall back to overfit.
        stage_args = [state["contract"], fixture, str(out)]
        if requested == "random":
            stage_args.append("random")
        try:
            result = backend.invoke("build-stage", workspace.root, stage_args)
            if result.returncode:
                return result.returncode
            metadata = read(out / "build.json")
            choice = policy.decision(metadata, requested, readable)
            if not choice["accepted"]:
                print(json.dumps(choice))
                return 1
            artifact_hash = digest(metadata["sha256"])
            target = (
                workspace.root
                / ".mic/artifacts"
                / fixture.split("/")[-1]
                / artifact_hash
            )
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(out, target)
            pin.update(
                revision="unpublished",
                files_sha256=artifact_hash,
                path=os.path.relpath(target, workspace.lock_path.parent),
            )
            write(workspace.lock_path, lock)
            workspace.save_state(
                weight_decision=choice,
                validated=False,
                validation=None,
                weights=requested,
                readable_output=readable,
            )
            print(
                json.dumps(
                    {
                        **choice,
                        "artifact": str(target),
                        "next": "tmic record; review the new candidates before publishing",
                    }
                )
            )
        finally:
            write(workspace.contract_root / "integration.json", original_manifest)
    return 0


def execute_one(args, record=False):
    workspace = Workspace(args.workspace)
    options = [
        "--framework",
        "transformers",
        "--contract",
        workspace.state["contract"],
        "--local-contracts",
        "--variant",
        args.variant,
        "--require-all",
    ]
    if args.isolate:
        options.append("--isolate")
    if args.target:
        for target in args.target:
            options += ["--target", target]
    if record:
        options.append("--record")
    if getattr(args, "no_baseline", False):
        options.append("--no-baseline")
    output = (
        workspace.root
        / ".mic"
        / f"{'record' if record else 'check'}.{args.variant}.jsonl"
    )
    output.unlink(missing_ok=True)
    options += ["--output", str(output)]
    result = backend.invoke("run", workspace.root, options)
    if not record:
        # Publication requires every declared target, not a focused partial check.
        rows = (
            [json.loads(line) for line in output.read_text().splitlines()]
            if output.exists()
            else []
        )
        evidence = {
            "variant": args.variant,
            "records": rows,
            "lock_sha256": hashlib.sha256(workspace.lock_path.read_bytes()).hexdigest(),
            "manifest_sha256": hashlib.sha256(
                (workspace.contract_root / "integration.json").read_bytes()
            ).hexdigest(),
            "usage_sha256": hashlib.sha256(
                (
                    workspace.contract_root
                    / workspace.manifest["integrations"]["transformers"]["entry_point"]
                ).read_bytes()
            ).hexdigest(),
        }
        key = args.variant
        validations = workspace.state.get("validations", {})
        if not args.target and result.returncode == 0 and bool(rows):
            validations[key] = {
                **evidence,
                "fingerprint": workspace.validation_fingerprint(),
            }
        else:
            validations.pop(key, None)
        workspace.save_state(
            validated=not args.target and result.returncode == 0 and bool(rows),
            validation=evidence,
            validations=validations,
        )
    return result.returncode


def execute(args, record=False):
    workspace = Workspace(args.workspace)
    variants = (
        list(
            workspace.manifest["integrations"]["transformers"].get(
                "variants", {"default": {}}
            )
        )
        if args.all_variants
        else [args.variant]
    )
    status = 0
    for variant in variants:
        args.variant = variant
        status = max(status, execute_one(args, record))
    return status


def review(args):
    workspace = Workspace(args.workspace)
    options = [
        "--framework",
        "transformers",
        "--expectations",
        str(workspace.expectations),
        "--manifest",
        str(workspace.contract_root / "integration.json"),
    ]
    if args.platform:
        options += ["--platform", args.platform]
    if args.approve:
        options.append("--approve")
        workspace.save_state(validated=False)
    return backend.invoke("review_candidates", workspace.root, options).returncode


def update(args):
    workspace = Workspace(args.workspace)
    old = workspace.manifest
    if len(old["checkpoints"]) != 1 or len(old["fixtures"]) != 1:
        raise ValueError(
            "LLM updates currently require a single-checkpoint contract; edit grouped contracts manually and use record/review/check"
        )
    entry = old["integrations"]["transformers"]["entry_point"]
    evidence_path = workspace.root / ".mic/history.json"
    context = {
        "model": workspace.state["model"],
        "fixture_repo": workspace.fixture,
        "manifest_schema": manifest_schema(),
        "existing_manifest": old,
        "existing_usage": (workspace.contract_root / entry).read_text(),
        "request": args.request,
        "failure_candidates": read(evidence_path)["items"][:30]
        if evidence_path.exists()
        else [],
    }
    plan, llm = propose_valid(
        context, args, workspace.state["model"], workspace.fixture, old
    )
    backup = workspace.snapshot()
    write(workspace.contract_root / "integration.json", plan["manifest"])
    (workspace.contract_root / entry).write_text(plan["usage"])
    write(
        workspace.root / ".mic/update-plan.json",
        {**plan, "llm": llm, "previous_source": str(backup)},
    )
    workspace.save_state(validated=False, validation=None)
    print(
        json.dumps(
            {
                "updated": workspace.state["contract"],
                "previous_source": str(backup),
                "weights": "retained",
                "next": "record and replay using the existing fixture first",
            }
        )
    )
    if args.retrain:
        args.rebuild = True
        args.weights, args.readable_output = None, False
        return build(args)
    return 0


def replay(args):
    workspace = Workspace(args.workspace)
    options = [
        args.fix,
        "--contract",
        workspace.state["contract"],
        "--local-contracts",
        "--variant",
        args.variant,
    ]
    if args.bad:
        options += ["--bad", args.bad]
    env = {**os.environ, "TRANSFORMERS_REPO": str(Path(args.repo).resolve())}
    result = backend.invoke("replay", workspace.root, options, capture=True, env=env)
    print(result.stdout, end="")
    if result.returncode:
        print(result.stderr, end="")
        return result.returncode
    rows = [
        json.loads(line) for line in result.stdout.splitlines() if line.startswith("{")
    ]
    write(workspace.root / ".mic/replay.json", rows)
    caught = [r for r in rows if r.get("verdict") == "caught"]
    if args.match_output:
        caught = [
            r
            for r in caught
            if any(f.startswith(args.match_output + ":") for f in r["bad_failures"])
        ]
    return 0 if caught else 1


def publication_ready(workspace):
    state = workspace.state
    proof = state.get("validation") or {}
    if proof.get("cloud_collection"):
        from .model_baselines import verify

        verify(workspace, Path(proof["cloud_collection"]), save=False)
    paths = {
        "lock_sha256": workspace.lock_path,
        "manifest_sha256": workspace.contract_root / "integration.json",
        "usage_sha256": workspace.contract_root
        / workspace.manifest["integrations"]["transformers"]["entry_point"],
    }
    if not state.get("validated") or any(
        proof.get(key) != hashlib.sha256(path.read_bytes()).hexdigest()
        for key, path in paths.items()
    ):
        raise ValueError(
            "Run tmic check on all declared targets after reviewing candidates; source or lock changes invalidate validation"
        )
    if list(workspace.expectations.rglob("*.candidate.json")):
        raise ValueError(
            "Unreviewed candidate baselines remain; run tmic review before publishing"
        )
    fingerprint = workspace.validation_fingerprint()
    variants = workspace.manifest["integrations"]["transformers"].get(
        "variants", {"default": {}}
    )
    if any(
        state.get("validations", {}).get(v, {}).get("fingerprint") != fingerprint
        for v in variants
    ):
        raise ValueError(
            "Every declared variant needs a successful check on the current sources and approved expectations; use tmic check --all-variants"
        )


def publish(args):
    workspace = Workspace(args.workspace)
    publication_ready(workspace)
    print(
        json.dumps(
            {
                "contract": workspace.state["contract"],
                "fixture": workspace.fixture,
                "execute": args.execute,
                "next": "tmic publish --execute, then tmic export --transformers-checkout PATH",
            }
        )
    )
    if not args.execute:
        return 0
    result = backend.invoke(
        "publish_mic", workspace.root, ["--private"] if args.private else []
    )
    workspace.save_state(validated=False)
    print("Run tmic check --all-variants on the published pins before export.")
    return result.returncode


def export(args):
    workspace = Workspace(args.workspace)
    publication_ready(workspace)
    state = workspace.state
    source = workspace.lock
    if source.get("local_patches"):
        raise ValueError(
            "Validate without local patches before exporting to Transformers; use narrow known_failures for unresolved framework bugs"
        )
    name = state["contract"]
    contract = source["contracts"][name]
    pins = [contract["integration"], *contract.get("fixtures", {}).values()]
    if any(p["revision"] == "unpublished" or not p.get("files_sha256") for p in pins):
        raise ValueError(
            "Publish immutable contract and fixture revisions before exporting to Transformers"
        )
    if workspace.contract_digest() != contract["integration"]["files_sha256"]:
        raise ValueError(
            "Local contract differs from its published pin; publish the reviewed source before export"
        )
    fleet_name = getattr(args, "fleet", "control")
    path, revision = fleet.download(fleet_name)
    if expected := getattr(args, "expected_revision", None):
        if revision != expected:
            raise ValueError(
                "Dataset changed since preview; review the current fleet before exporting"
            )
    old = read(path)
    destination_name = getattr(args, "contract_name", None) or name
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", destination_name):
        raise ValueError("Destination contract name must be a safe folder name")
    existing_names = set(old["contracts"])
    if any(
        n != destination_name and n.casefold() == destination_name.casefold()
        for n in existing_names
    ):
        raise ValueError(
            "Destination name differs only by case from an existing contract; use --contract-name"
        )
    if old["environment"] != source["environment"]:
        differences = "; ".join(
            f"{key}: fleet={old['environment'].get(key, '<unset>')}, model={source['environment'].get(key, '<unset>')}"
            for key in sorted(set(old["environment"]) | set(source["environment"]))
            if old["environment"].get(key) != source["environment"].get(key)
        )
        raise ValueError(
            f"Destination environment differs ({fleet_name} at {revision}): {differences}. "
            "Validate a separate fleet environment migration, or measure the model "
            "with the destination environment before exporting. Retrying unchanged will not fix this."
        )
    # Hub-only paths keep the exported lock independent of this workspace.
    clean = upgrade_contract(canonical_references(contract))
    for pin in [clean["integration"], *clean.get("fixtures", {}).values()]:
        pin.pop("path", None)
    old["contracts"][destination_name] = clean
    renamed_targets = {}
    for repo, pin in contract.get("fixtures", {}).items():
        canonical = umbrella_references(canonical_references(repo))
        if current := clean.get("fixtures", {}).get(canonical):
            renamed_targets[pin.get("target", repo.split("/")[-1])] = current.get(
                "target", canonical.split("/")[-1]
            )

    # Only this workspace's contract belongs to the update; keep other cases intact.
    known = old.setdefault("known_failures", {})
    for key in list(known):
        if key.startswith(destination_name + "/"):
            del known[key]
    for key, value in source.get("known_failures", {}).items():
        if key.startswith(name + "/"):
            target = key[len(name) + 1 :]
            known[destination_name + "/" + renamed_targets.get(target, target)] = value
    files = fleet.operations(fleet_name, old)
    executed = getattr(args, "execute", False)
    checkout = getattr(args, "transformers_checkout", None)
    if executed:
        commit = HfApi().create_commit(
            repo_id=fleet.REPOSITORY,
            repo_type="dataset",
            parent_commit=revision,
            operations=[
                CommitOperationAdd(path_in_repo=path, path_or_fileobj=text.encode())
                for path, text in files.items()
            ],
            commit_message=f"Update {destination_name} in the {fleet_name} fleet",
        )
        revision = commit.oid
        if checkout:
            fleet.pin(checkout, revision)
    print(
        json.dumps(
            {
                "repository": fleet.REPOSITORY,
                "fleet": fleet_name,
                "contract": destination_name,
                "revision": revision,
                "execute": executed,
                "contracts": old["contracts"],
                "next": "Review the dataset update and the tests/contracts.lock diff."
                if executed
                else "Review this preview, then repeat with --execute to upload.",
            }
        )
    )
    return 0


def pin(args):
    revision = fleet.resolve(args.revision)
    path = fleet.pin(args.transformers_checkout, revision)
    print(
        json.dumps(
            {"repository": fleet.REPOSITORY, "revision": revision, "pin": str(path)}
        )
    )
    return 0
