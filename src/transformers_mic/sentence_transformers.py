"""Sentence Transformers inference contracts and complete shrunk model pipelines."""

import hashlib
import json
import platform
import re
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

from huggingface_hub import HfApi, snapshot_download

CLASSES = ("SentenceTransformer", "CrossEncoder", "SparseEncoder", "MultiVectorEncoder")
QUERIES = ["Which planet is known as the Red Planet?", "Where do penguins live?"]
DOCUMENTS = [
    "Venus is often called Earth's twin.",
    "Mars is known for its reddish appearance and is called the Red Planet.",
    "Penguins live in the Southern Hemisphere, including Antarctica.",
    "Jupiter is the largest planet in our solar system and has a prominent red spot.",
]


def source_files(repo, revision):
    return Path(
        snapshot_download(
            repo,
            revision=revision,
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.jinja",
                "*.tiktoken",
            ],
            ignore_patterns=["onnx/*", "openvino/*", "*.index.json"],
        )
    )


def model_kind(source, requested="auto"):
    if requested != "auto":
        if requested not in CLASSES:
            raise ValueError(f"Unsupported ST class: {requested}")
        return requested
    config_path = source / "config_sentence_transformers.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    saved = config.get("model_type")
    if saved in CLASSES:
        return saved
    modules_path = source / "modules.json"
    modules = json.loads(modules_path.read_text()) if modules_path.exists() else []
    types = " ".join(m["type"] for m in modules)
    if saved == "ColBERT" or "pylate." in types or "MultiVector" in types:
        return "MultiVectorEncoder"
    if "Splade" in types or "Sparse" in types:
        return "SparseEncoder"
    backbone = json.loads((source / "config.json").read_text())
    architectures = backbone.get("architectures") or []
    if any(a.endswith("ForRetrieval") for a in architectures):
        return "MultiVectorEncoder"
    if any(a.endswith("ForSequenceClassification") for a in architectures):
        return "CrossEncoder"
    return "SentenceTransformer"


def environment():
    result = {"python": platform.python_version()}
    for name in (
        "torch",
        "transformers",
        "tokenizers",
        "sentence-transformers",
        "huggingface-hub",
    ):
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError as error:
            raise ValueError(
                "ST contracts require transformers-mic[runtime,sentence-transformers]"
            ) from error
    return result


def usage(repo, kind, images=False, image_documents=False, left_padding=False):
    setup = [
        "import torch",
        f"from sentence_transformers import {kind}",
        "import json  # doc-builder: hide",
        "import os  # doc-builder: hide",
        "",
        f"model_id = {repo!r}",
        "model_id = os.environ.get('MIC_MODEL', model_id)  # doc-builder: hide",
        "revision = os.environ.get('MIC_MODEL_REVISION') or None  # doc-builder: hide",
        "variant = json.loads(os.environ.get('MIC_MODEL_KWARGS', '{}'))  # doc-builder: hide",
        "torch.manual_seed(0)",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'",
        f"model = {kind}(",
        "    model_id,",
        "    device=device,",
        "    revision=revision,  # doc-builder: hide",
        "    model_kwargs=variant,  # doc-builder: hide",
        ")",
        f"queries = {QUERIES!r}",
        f"documents = {DOCUMENTS!r}",
    ]
    if left_padding:
        setup.append("model.tokenizer.padding_side = 'left'")
    if images or image_documents:
        setup += [
            "from PIL import Image",
            "images = [Image.new('RGB', (32, 32), color) for color in ('red', 'green', 'blue', 'yellow')]",
        ]
    if image_documents:
        setup += ["documents = images"]
    if kind == "CrossEncoder":
        setup += [
            "scores = model.predict([(query, document) for query in queries for document in documents], show_progress_bar=False, convert_to_tensor=True)",
            "assert scores.shape == (8,)",
            "assert torch.isfinite(scores).all()",
        ]
        outputs = ["'scores': scores.float().cpu().tolist()"]
    else:
        setup += [
            "query_embeddings = model.encode_query(queries, show_progress_bar=False, convert_to_tensor=True)",
            "document_embeddings = model.encode_document(documents, show_progress_bar=False, convert_to_tensor=True)",
            "scores = model.similarity(query_embeddings, document_embeddings)",
            "assert scores.shape == (2, 4)",
            "assert torch.isfinite(scores).all()",
        ]
        outputs = ["'scores': scores.float().cpu().tolist()"]
        if kind == "MultiVectorEncoder":
            setup += [
                "assert all(torch.isfinite(item).all() for item in [*query_embeddings, *document_embeddings])",
                "assert not torch.allclose(query_embeddings[0][:1], query_embeddings[1][:1])",
            ]
            outputs += [
                "'query_lengths': [len(item) for item in query_embeddings]",
                "'document_lengths': [len(item) for item in document_embeddings]",
                "'query_probe': query_embeddings[0][:4, :8].float().cpu().tolist()",
                "'document_probe': document_embeddings[0][:4, :8].float().cpu().tolist()",
            ]
        elif kind == "SparseEncoder":
            setup += [
                "query_dense = query_embeddings.to_dense()",
                "document_dense = document_embeddings.to_dense()",
                "assert torch.isfinite(query_dense).all() and torch.isfinite(document_dense).all()",
                "assert (query_dense[0] - query_dense[1]).abs().max() > 0.001",
            ]
            outputs += [
                "'query_shape': list(query_dense.shape)",
                "'document_shape': list(document_dense.shape)",
                "'query_probe': query_dense[:, [1, 10, 100, 1000, 2000, 3000, 4000, 5000]].float().cpu().tolist()",
                "'document_probe': document_dense[:, [1, 10, 100, 1000, 2000, 3000, 4000, 5000]].float().cpu().tolist()",
            ]
        else:
            setup += [
                "assert torch.isfinite(query_embeddings).all() and torch.isfinite(document_embeddings).all()",
                "assert (query_embeddings[0] - query_embeddings[1]).abs().max() > 0.001",
                "individual = model.encode_query(queries, batch_size=1, show_progress_bar=False, convert_to_tensor=True)",
                "batch_max_delta = float((query_embeddings - individual).abs().max())",
                "if query_embeddings.dtype == torch.float32:",
                "    torch.testing.assert_close(query_embeddings, individual, atol=1e-4, rtol=1e-4)",
            ]
            outputs += [
                "'batch_max_delta': batch_max_delta",
                "'query_embeddings': query_embeddings.float().cpu().tolist()",
                "'document_embeddings': document_embeddings.float().cpu().tolist()",
            ]
        if images:
            setup += [
                "image_embeddings = model.encode(images, show_progress_bar=False, convert_to_tensor=True)",
                "image_scores = model.similarity(query_embeddings, image_embeddings)",
                "assert image_scores.shape == (2, 4)",
                "assert torch.isfinite(image_scores).all()",
                "assert (image_embeddings[0] - image_embeddings[1]).abs().max() > 0.001",
            ]
            outputs += [
                "'image_embeddings': image_embeddings.float().cpu().tolist()",
                "'image_scores': image_scores.float().cpu().tolist()",
            ]
    if kind == "MultiVectorEncoder":
        setup = [line.replace(", convert_to_tensor=True", "") for line in setup]
    setup += [
        "",
        "if os.environ.get('MIC_RUNNER'):  # doc-builder: hide",
        "    backbone = next(module for module in model.modules() if isinstance(module, __import__('transformers').PreTrainedModel))",
        "    result = {",
        "        'runtime': {'device': model.device.type, 'dtype': str(next(model.parameters()).dtype).removeprefix('torch.'), 'attn_implementation': backbone.config._attn_implementation},",
        *["        " + item + "," for item in outputs],
        "    }",
        "    print(json.dumps(result))",
    ]
    title = f"# {kind}: encoding and scoring\n\n"
    description = "Run the public Sentence Transformers API with a small, fixed retrieval example.\n\n"
    return (
        title
        + description
        + "```py runnable:retrieval\n"
        + "\n".join(setup)
        + "\n```\n"
    )


def create(args):
    from .commands import finish_create, validate_plan
    from .workspace import Workspace, write

    versions = environment()
    info = HfApi().model_info(args.model, revision=args.revision)
    source = source_files(args.model, info.sha)
    kind = model_kind(source, args.sentence_transformers)
    config = json.loads((source / "config.json").read_text())
    slug = args.name or "st-" + re.sub(
        r"[^A-Za-z0-9_-]", "-", args.model.split("/")[-1]
    )
    fixture = f"{args.namespace}/{slug}-mic"
    images = config["model_type"] in ("clip", "embedding_gemma2")
    image_documents = kind == "MultiVectorEncoder" and config["model_type"] == "colpali"
    left_padding = kind == "CrossEncoder" and config["model_type"] in {
        "qwen3",
        "llama",
        "mistral",
    }
    code = usage(args.model, kind, images, image_documents, left_padding)
    fields = re.findall(r"^        '([a-z_]+)':", code, re.MULTILINE)
    outputs = {
        field: {"compare": "exact"}
        if field == "runtime" or field.endswith(("_shape", "_lengths"))
        else {"compare": "allclose", "atol": 1e-4, "rtol": 1e-4}
        for field in fields
    }
    manifest = {
        "mic": "0.3",
        "name": f"{kind}: {args.model}",
        "usage": f"{kind} public inference and retrieval scoring",
        "architecture": (config.get("architectures") or [config["model_type"]])[0],
        "checkpoints": {args.model: {"fixture": fixture}},
        "fixtures": {
            fixture: {
                "built_from": args.model,
                "covers": [args.model],
                "build": {"seed": 17, "sentence_transformers_class": kind},
            }
        },
        "integrations": {
            "transformers": {
                "entry_point": "integrations/transformers/usage.md",
                "format": "runnable-markdown",
                "requires": [
                    "sentence-transformers>=6.2",
                    "transformers>=5",
                    "torch",
                    "pillow",
                ],
                "targets": ["fixtures"]
                if args.fixtures_only
                else ["fixtures", "checkpoints"],
                "variants": {
                    "default": {},
                    "eager": {"attn_implementation": "eager"},
                    "bfloat16": {"dtype": "bfloat16"},
                },
                "training": False,
            }
        },
        "outputs": outputs,
    }
    plan = {
        "name": slug,
        "manifest": manifest,
        "usage": code,
        "rationale": "Use the saved ST pipeline and its public inference API.",
        "historical_candidates": [],
    }
    validate_plan(plan, args.model, fixture)
    workspace = Workspace(args.out)
    workspace.from_plan(
        args.model,
        info.sha,
        args.namespace,
        slug,
        manifest,
        code,
        args.weights,
        args.readable_output,
    )
    lock = workspace.lock
    lock["environment"] = versions
    write(workspace.lock_path, lock)
    workspace.save_state(origin="sentence-transformers")
    workspace.update_layout()
    write(workspace.root / ".mic/plan.json", plan)
    print(
        json.dumps(
            {
                "workspace": str(workspace.root),
                "class": kind,
                "fixture": fixture,
                "next": ["tmic build", "tmic record --all-variants"],
            }
        )
    )
    return finish_create(args, workspace)


def initialize(source, kind, seed, builder):
    import sentence_transformers as st
    import torch
    from sentence_transformers.base.modules import Dense, Module, Transformer
    from sentence_transformers.base.modules.transformer import (
        _resolve_retrieval_model_class,
    )
    from transformers import (
        AutoModel,
        AutoModelForMaskedLM,
        AutoModelForSequenceClassification,
    )

    changes = []
    backbones = []

    def load_backbone(self, path, task, config, backend, is_peft_model, **kwargs):
        if is_peft_model or backend != "torch":
            raise ValueError("ST MIC fixtures require a native torch backbone")
        factory = {
            "feature-extraction": AutoModel,
            "fill-mask": AutoModelForMaskedLM,
            "sequence-classification": AutoModelForSequenceClassification,
        }.get(task)
        small = builder.shrink(config)
        if task == "retrieval":
            model = _resolve_retrieval_model_class(small)._from_config(small)
        elif factory is not None:
            model = factory.from_config(small)
        else:
            raise ValueError(f"Unsupported ST fixture task: {task}")
        backbones.append(model)
        return model

    def random_weights(cls, model_name_or_path, model=None, **kwargs):
        if model is None:
            raise ValueError(
                "ST fixture cannot infer the shape of unnamed module weights"
            )
        return model

    torch.manual_seed(seed)
    with (
        patch.object(Transformer, "_load_model", load_backbone),
        patch.object(Module, "load_torch_weights", classmethod(random_weights)),
    ):
        model = getattr(st, kind)(
            str(source), device="cpu", local_files_only=True, trust_remote_code=False
        )
    if len(backbones) != 1:
        raise ValueError(f"Expected one ST backbone, got {len(backbones)}")
    dimension = model[0].get_embedding_dimension()
    for module in list(model)[1:]:
        if isinstance(module, st.sentence_transformer.modules.Pooling):
            old = module.embedding_dimension
            module.embedding_dimension = dimension
            modes = module.pooling_mode
            module.pooling_output_dimension = dimension * (
                len(modes) if isinstance(modes, (tuple, list)) else 1
            )
            changes.append(
                {"module": type(module).__name__, "from": old, "to": dimension}
            )
        elif isinstance(module, Dense):
            old = (module.in_features, module.out_features)
            output = min(module.out_features, 64)
            module.in_features, module.out_features = dimension, output
            module.linear = torch.nn.Linear(dimension, output, bias=module.bias)
            if getattr(module, "use_residual", False) and dimension != output:
                module.residual = torch.nn.Linear(dimension, output, bias=False)
            changes.append({"module": "Dense", "from": old, "to": (dimension, output)})
        if hasattr(module, "get_embedding_dimension"):
            current = module.get_embedding_dimension()
            if current is not None:
                dimension = current
    return model.eval(), backbones[0], changes


def build(
    builder,
    name,
    fixture,
    contract_name,
    manifest,
    root,
    revisions,
    out,
    keep_random=False,
    **kwargs,
):

    repo = fixture["built_from"]
    recipe = fixture["build"]
    kind = recipe["sentence_transformers_class"]
    source = source_files(repo, revisions[repo])
    builder.report_phase("Initialize complete ST pipeline", kind)
    model, backbone, changes = initialize(source, kind, recipe["seed"], builder)
    out.mkdir(parents=True, exist_ok=True)

    def save():
        model.save_pretrained(str(out), create_model_card=False)
        copied = []
        for file in source.iterdir():
            if (
                file.is_file()
                and file.name
                not in {
                    "config.json",
                    "sentence_bert_config.json",
                    "config_sentence_transformers.json",
                    "modules.json",
                    "README.md",
                }
                and file.suffix in {".json", ".txt", ".model", ".jinja", ".tiktoken"}
            ):
                destination = out / file.name
                destination.write_bytes(file.read_bytes())
                copied.append(destination.relative_to(out).as_posix())
        return copied

    copied = save()
    entry = root / manifest["integrations"]["transformers"]["entry_point"]
    batch, captured = builder.contract_batch(
        entry, out, model_class=type(backbone), method="forward"
    )
    threshold = builder.GATE_FACTOR * max(
        rule["atol"]
        for rule in manifest["outputs"].values()
        if rule["compare"] == "allclose"
    )
    gate = {
        "random": builder.sensitivity(
            builder.called_module(backbone, captured),
            batch,
            threshold,
            builder.config_insensitive(backbone.config, batch),
        )
    }
    weights, training = "random", None
    if not gate["random"]["pass"] or recipe.get("overfit"):
        if keep_random:
            raise ValueError(f"ST random fixture failed sensitivity: {gate}")
        training = builder.overfit(
            builder.called_module(backbone, captured),
            batch,
            recipe.get("max_steps", 500),
            recipe.get("target_loss", 0.05),
        )
        model.eval()
        gate["overfit"] = builder.sensitivity(
            builder.called_module(backbone, captured),
            batch,
            threshold,
            builder.config_insensitive(backbone.config, batch),
        )
        if not training["converged"] or not gate["overfit"]["pass"]:
            raise ValueError(f"ST overfit fixture failed sensitivity: {gate}")
        weights = "overfit"
        copied = save()
    builder.contract_batch(entry, out, model_class=type(backbone), method="forward")
    health = builder.numeric_health(type(backbone), out, batch, captured)
    if not all(row["finite"] for row in health.values()):
        raise ValueError(f"ST fixture failed numeric health: {health}")
    source_hashes = {
        p.relative_to(source).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in source.rglob("*")
        if p.is_file()
        and ".cache" not in p.parts
        and p.suffix in {".json", ".txt", ".model", ".jinja", ".tiktoken"}
        and not {"onnx", "openvino"}.intersection(p.relative_to(source).parts)
    }
    fields = {
        "config": {
            key: value
            for key, value in backbone.config.to_dict().items()
            if key not in builder.VOLATILE
        },
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "source_files": source_hashes,
        "st_class": kind,
        "module_changes": changes,
    }
    record = {
        "fixture": name,
        "key": hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest(),
        "key_fields": fields["config"],
        "sentence_transformers": fields,
        "covers": {repo: revisions[repo]},
        "built_from": {"repo": repo, "revision": revisions[repo]},
        "seed": recipe["seed"],
        "dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
        "builder": "transformers_mic.sentence_transformers",
        "builder_revision": builder.builder_revision(),
        "recipe": recipe,
        "shrink": "rule",
        "model_class": type(backbone).__name__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights": weights,
        "gate": gate,
        "numeric_health": health,
        "copied_unchanged": copied,
        "captured": captured,
        "environment": environment(),
        "sha256": builder.run.file_hashes(out, recursive=True),
    }
    if training:
        record["training"] = training
    builder.run.validate(record, "fixture-build", f"{name} build.json")
    (out / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "README.md").write_text(
        f"---\nlibrary_name: sentence-transformers\ntags: [mic, test-artifact]\nbase_model: {repo}\n---\n\n"
        f"# {kind} MIC fixture\n\nShrunk pipeline for {repo}. Its weights are {weights} and do not represent the source model's retrieval quality.\n"
    )
    return record
