"""Transformers usage.md from a few model-specific slots.

Every contract shares a scaffold: the hidden MIC plumbing, the device and seed, the
variant reload, the training step and the hidden result record. Only the imports,
the loading, the inference code, extra sections, the training inputs and loss, the
checked parameters and the outputs differ. The planner writes those slots and
`render` writes the rest, so a draft cannot get the scaffold wrong: the fences, the
plumbing and the record were where most drafts failed (google/switch-base-8,
2026-11-05). `extract` is the inverse, for existing pages.

Slots (a JSON object; optional ones may be missing or empty):
    title          the page's H1
    name           the contract name in the standard intro ("BERT")
    intro          optional: replaces the standard intro paragraph
    preamble       optional paragraphs between the intro and the first section
    block          the runnable block name ("fill_mask")
    heading, text  the first section's heading and optional paragraph
    imports        import lines; torch and set_seed are added when missing
    setup          optional code before the model loads (processor, data files)
    model_class    the class loading the checkpoint
    load_kwargs    optional fixed from_pretrained arguments: a dict literal such as
                   '{"dtype": torch.float32}', or a name `setup` defines; variants
                   are merged over it
    inference      the first block's code after the model loads
    sections       [{"heading", "text", "code"}] extra runnable sections, in order
    training       null, or {"text", "inputs", "loss", "optimizer"}:
                     inputs     code before model.train() (labels, freezing)
                     loss       code that assigns `loss`, usually one line
                     optimizer  "all", "trainable" (parameters that require grad;
                                for partially frozen models), or the code of a
                                custom `optimizer = ...` line
                     record     optional {"name": "expression"} extra training keys
    record_setup   optional code at the top of the hidden record (after checked)
    checked        parameter names whose gradients the record checks (training only),
                   or the source of an expression computing them (suffix lookups)
    outputs        {"name": "expression"} in record order, after runtime
    footer         optional closing paragraph
"""

import ast
import json
import re
import textwrap

INTRO = (
    "This page is the Transformers contract of the {name} MIC and also a usage example. "
    "[doc-builder](https://github.com/huggingface/doc-builder/blob/main/docs/runnable-code-blocks.md) "
    "renders the `runnable` blocks below as documentation, and its pytest plugin runs them as "
    "written against `{checkpoint}`. The MIC runner runs the same blocks against each slim fixture "
    "and each checkpoint in [`integration.json`](../../integration.json). Lines marked "
    "`# doc-builder: hide` only matter there: they swap in the pinned checkpoint or variant and "
    "print the result record."
)
HIDE = "  # doc-builder: hide"
SEED = "set_seed(0)  # seeds torch, NumPy, and Python: loading, dropout, and data augmentation draw random numbers"
RUNTIME = """        "runtime": {
            "device": model.device.type,
            "dtype": str(model.dtype).removeprefix("torch."),
            "attn_implementation": model.config._attn_implementation,
        },"""
GRADIENTS = '"gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},'
OPTIMIZERS = {
    "all": "optimizer = torch.optim.SGD(model.parameters(), lr=0.01)",
    "trainable": "optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)",
}
BEFORE = {
    "all": "before = {name: p.detach().clone() for name, p in model.named_parameters()}",
    # Partially frozen models are large: snapshot only what trains, on the CPU.
    "trainable": "before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}",
}
UPDATED = {
    "all": '"updated": any(not torch.equal(p, before[n]) for n, p in named.items()),',
    "trainable": '"updated": any(not torch.equal(p.detach().cpu(), before[n]) for n, p in model.named_parameters() if n in before),',
}
RUNNER = 'if os.environ.get("MIC_RUNNER"):'
PLUMBING = (
    re.compile(r"^import json\s*# doc-builder: hide$"),
    re.compile(r"^import os\s*# doc-builder: hide$"),
    re.compile(r'^model_id = os\.environ\.get\("MIC_MODEL", model_id\)\s*# doc-builder: hide$'),
    re.compile(r'^variant = json\.loads\(os\.environ\.get\("MIC_MODEL_KWARGS", "\{\}"\)\)\s*# doc-builder: hide$'),
)


def _imports(lines):
    text = lines.strip("\n")
    if not re.search(r"^import torch$", text, re.M):
        text = "import torch\n" + text
    if not re.search(r"\bset_seed\b", text):
        text += "\nfrom transformers import set_seed"
    return text


def _mode(training):
    optimizer = training.get("optimizer") or "all"
    return optimizer if optimizer in OPTIMIZERS else "trainable" if "requires_grad" in optimizer else "all"


def _record(slots):
    training = slots.get("training")
    lines = [f"{RUNNER}{HIDE}"]
    setup = (slots.get("record_setup") or "").strip("\n")
    checked = slots.get("checked") or []
    if isinstance(checked, str):
        checked_line = f"    checked = {checked}"
    elif checked:
        names = ", ".join(json.dumps(n) for n in checked)
        checked_line = f"    checked = ({names}{',' if len(checked) == 1 else ''})"
    else:  # the training code already defines checked (LFM2)
        checked_line = None
    if training:
        lines.append("    named = dict(model.named_parameters())")
        lines.append("    names = list(named)  # for suffix lookups: next(n for n in names if n.endswith(...))")
    # checked may use names the setup computes (an expert's suffix lookup), or the reverse.
    setup_first = bool(setup) and not re.search(r"\bchecked\b", setup)
    if setup and setup_first:
        lines.append(textwrap.indent(setup, "    "))
    if training and checked_line:
        lines.append(checked_line if "\n" not in checked_line else textwrap.indent(checked_line.strip(), "    "))
    if setup and not setup_first:
        lines.append(textwrap.indent(setup, "    "))
    lines += ["    result = {", RUNTIME]
    lines += [f"        {json.dumps(name)}: {expr}," for name, expr in slots["outputs"].items()]
    if training:
        lines += ['        "training": {', '            "loss": loss.item(),', f"            {GRADIENTS}"]
        lines += [f"            {json.dumps(k)}: {v}," for k, v in (training.get("record") or {}).items()]
        lines += [f"            {UPDATED[_mode(training)]}", "        },"]
    lines += ["    }", "    print(json.dumps(result))"]
    return "\n".join(lines)


def _training(training):
    mode = _mode(training)
    optimizer = training.get("optimizer") or "all"
    lines = [
        "model.train()",
        OPTIMIZERS.get(optimizer, optimizer.strip()),
        training["loss"].strip("\n"),
        "loss.backward()",
        BEFORE[mode] + HIDE,
        "optimizer.step()",
        'print(f"loss: {loss.item():.4f}")',
    ]
    inputs = (training.get("inputs") or "").strip("\n")
    return f"{inputs}\n\n" + "\n".join(lines) if inputs else "\n".join(lines)


def _section(heading, text, block, code):
    parts = [f"## {heading}"]
    if text:
        parts.append(text.strip())
    parts.append(f"```py runnable:{block}\n{code.strip(chr(10))}\n```")
    return "\n\n".join(parts)


def _load(model_class, load_kwargs, variant):
    if not load_kwargs:
        args = "model_id, **variant" if variant else "model_id"
    elif not load_kwargs.strip().startswith("{"):  # a name the setup defines
        args = f"model_id, **{{**{load_kwargs.strip()}, **variant}}" if variant else f"model_id, **{load_kwargs.strip()}"
    else:
        inner = load_kwargs.strip()[1:-1].strip().rstrip(",")
        args = f"model_id, **{{{inner}, **variant}}" if variant else f"model_id, **{load_kwargs.strip()}"
    return f"model = {model_class}.from_pretrained({args}).to(device)"


def render(slots, checkpoint):
    """usage.md for the slots, with `checkpoint` as the documented model_id."""
    block, model_class = slots["block"], slots["model_class"]
    first = [
        _imports(slots["imports"]),
        f"import json{HIDE}",
        f"import os{HIDE}",
        "",
        f'model_id = "{checkpoint}"',
        f'model_id = os.environ.get("MIC_MODEL", model_id){HIDE}',
        f'variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{{}}")){HIDE}',
        'device = "cuda" if torch.cuda.is_available() else "cpu"',
        SEED,
        "",
    ]
    if slots.get("setup"):
        first.append(slots["setup"].strip("\n"))
    first += [
        _load(model_class, slots.get("load_kwargs"), False),
        # Free the default model first: a real checkpoint does not fit the GPU twice.
        f"if variant:{HIDE}",
        f"    del model{HIDE}",
        "    " + _load(model_class, slots.get("load_kwargs"), True) + HIDE,
        "",
        slots["inference"].strip("\n"),
    ]
    blocks = [[slots.get("heading") or "Run the model", slots.get("text", ""), "\n".join(first)]]
    for section in slots.get("sections") or []:
        blocks.append([section["heading"], section.get("text", ""), section["code"].strip("\n")])
    training = slots.get("training")
    if training:
        blocks.append(["One training step", training.get("text", ""), _training(training)])
    blocks[-1][2] += "\n\n" + _record(slots)
    sections = [
        _section(heading, text, block if i == 0 else f"{block}:{i + 1}", code)
        for i, (heading, text, code) in enumerate(blocks)
    ]
    intro = slots.get("intro") or INTRO.format(name=slots["name"], checkpoint=checkpoint)
    page = [f"# {slots['title']}", intro]
    if slots.get("preamble"):
        page.append(slots["preamble"].strip())
    page += sections
    if slots.get("footer"):
        page.append(slots["footer"].strip())
    return "\n\n".join(page) + "\n"


# ---------------------------------------------------------------- extraction


def _lines(source, statements):
    """The source lines of consecutive statements, with the comments between them."""
    if not statements:
        return ""
    lines = source.split("\n")
    start = statements[0].lineno - 1
    while start > 0 and lines[start - 1].strip().startswith("#"):
        start -= 1
    return textwrap.dedent("\n".join(lines[start : statements[-1].end_lineno])).strip("\n")


def _is_load(statement):
    """(class, fixed kwargs source or None, passes **variant) for `model = X.from_pretrained(model_id, ...)...`."""
    if not (isinstance(statement, ast.Assign) and ast.unparse(statement.targets[0]) == "model"):
        return None
    call = statement.value
    while isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr in ("to", "eval", "cuda"):
        call = call.func.value
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == "from_pretrained"):
        return None
    if not (call.args and ast.unparse(call.args[0]) == "model_id") or not isinstance(call.func.value, ast.Name):
        return None
    fixed, variant = [], False
    for keyword in call.keywords:
        if keyword.arg is not None:
            fixed.append(f"{json.dumps(keyword.arg)}: {ast.unparse(keyword.value)}")
        elif isinstance(keyword.value, ast.Name) and keyword.value.id == "variant":
            variant = True
        elif isinstance(keyword.value, ast.Name) and not fixed:
            fixed = keyword.value.id  # **load_kwargs: a name the setup defines
        elif isinstance(keyword.value, ast.Dict):
            for key, value in zip(keyword.value.keys, keyword.value.values):
                if key is None and ast.unparse(value) == "variant":
                    variant = True
                elif key is None and isinstance(value, ast.Name) and not fixed:
                    fixed = value.id  # **{**load_kwargs, **variant}
                elif key is None:
                    raise ValueError(f"cannot express load unpacking {ast.unparse(value)}")
                else:
                    name = json.dumps(key.value) if isinstance(key, ast.Constant) else ast.unparse(key)
                    fixed.append(f"{name}: {ast.unparse(value)}")
        else:
            raise ValueError(f"cannot express load arguments **{ast.unparse(keyword.value)}")
    if isinstance(fixed, str):
        return call.func.value.id, fixed, variant
    return call.func.value.id, ("{" + ", ".join(fixed) + "}") if fixed else None, variant


def _is_reload(statement):
    return (
        isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "variant"
        and any(_is_load(s) for s in statement.body)
        and all(_is_load(s) or ast.unparse(s) == "del model" for s in statement.body)
    )


def _first_block(code):
    code = "\n".join(line for line in code.split("\n") if not any(p.match(line.strip()) for p in PLUMBING))
    body = ast.parse(code).body
    model_id = next(i for i, s in enumerate(body) if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "model_id")
    loads = [i for i, s in enumerate(body) if _is_load(s)]
    if not loads:
        raise ValueError("no model = X.from_pretrained(model_id, ...) statement")
    model_class, load_kwargs, _ = _is_load(body[loads[0]])

    def scaffold(s):
        source = ast.unparse(s)
        return (
            (isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "device")
            or source.startswith("set_seed(")
        )

    setup = [s for s in body[model_id + 1 : loads[0]] if not scaffold(s) and not _is_reload(s)]
    inference = [s for s in body[loads[0] + 1 :] if not _is_reload(s)]
    return {
        "imports": _lines(code, body[:model_id]),
        "setup": "\n\n".join(_lines(code, [s]) for s in setup) if setup else "",
        "model_class": model_class,
        "load_kwargs": load_kwargs,
        "inference": _join(code, inference),
    }


def _join(code, statements):
    """Source of statements, keeping runs of adjacent statements together."""
    groups, run = [], []
    for s in statements:
        if run and s.lineno > run[-1].end_lineno + 1 + _gap(code, run[-1], s):
            groups.append(run)
            run = []
        run.append(s)
    if run:
        groups.append(run)
    return "\n\n".join(_lines(code, g) for g in groups)


def _gap(code, previous, statement):
    """Lines between two statements that are comments or blank: they do not split a run."""
    lines = code.split("\n")[previous.end_lineno : statement.lineno - 1]
    return sum(1 for line in lines if line.strip().startswith("#"))


def _record_slots(record, source):
    """checked, record_setup, outputs and extra training keys from the hidden record."""

    def code_of(_):
        return source

    checked, setup, outputs, extra = [], [], {}, {}
    for s in record.body:
        target = ast.unparse(s.targets[0]) if isinstance(s, ast.Assign) else None
        if target == "named" or (target == "names" and ast.unparse(s.value) == "list(named)"):
            continue
        if target == "checked":
            try:
                checked = list(ast.literal_eval(s.value))
            except ValueError:  # computed, e.g. by suffix lookups
                checked = _lines(code_of(record), [s]).partition("=")[2].strip()
        elif target == "result" and isinstance(s.value, ast.Dict):
            for key, value in zip(s.value.keys, s.value.values):
                name = getattr(key, "value", None)
                if name == "runtime":
                    continue
                if name == "training" and isinstance(value, ast.Dict):
                    for k, v in zip(value.keys, value.values):
                        if getattr(k, "value", None) not in ("loss", "gradients", "updated"):
                            extra[k.value] = ast.get_source_segment(source, v)
                elif isinstance(name, str):
                    outputs[name] = ast.get_source_segment(source, value)
                else:
                    raise ValueError("result keys must be strings")
        elif ast.unparse(s) == "print(json.dumps(result))":
            continue
        else:
            setup.append(s)
    return checked, setup, outputs, extra


def _training_block(code):
    body = ast.parse(code).body
    train = next(i for i, s in enumerate(body) if ast.unparse(s) == "model.train()")
    record = next(i for i, s in enumerate(body) if isinstance(s, ast.If) and "MIC_RUNNER" in ast.unparse(s.test))
    optimizer, loss = "all", []
    for s in body[train + 1 : record]:
        source = ast.unparse(s)
        if isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "optimizer":
            normalized = re.sub(r"lr=[\d.e-]+", "lr=0.01", source)
            optimizer = next((k for k, v in OPTIMIZERS.items() if v == normalized), source)
            if "lr=0.01" not in source:
                optimizer = source
        elif source in ("loss.backward()", "optimizer.step()") or source.startswith("print(f") and "loss" in source:
            continue
        elif isinstance(s, ast.Assign) and ast.unparse(s.targets[0]) == "before":
            continue
        else:
            loss.append(s)
    if not any("loss" in [ast.unparse(t) for t in getattr(s, "targets", [])] or isinstance(s, ast.With) for s in loss):
        raise ValueError("training step assigns no loss")
    return {
        "inputs": _join(code, body[:train]),
        "loss": _join(code, loss),
        "optimizer": optimizer,
    }, body[record], code


def extract(usage):
    """Slots from an existing usage.md: the inverse of `render`.

    Raises ValueError for a page the template cannot express.
    """
    title = re.search(r"^# (.+)$", usage, re.M).group(1)
    fences = list(re.finditer(r"^## (.+?)\n\n(.*?)```py runnable:([\w-]+)[^\n]*\n(.*?)^```\n?", usage, re.M | re.S))
    if not fences:
        raise ValueError("no runnable sections")
    blocks = [(m.group(1), m.group(2).strip(), m.group(3), m.group(4)) for m in fences]
    name_match = re.search(r"Transformers contract of the (.+?) MIC", usage)
    head = usage[usage.index("\n", usage.index("# ")) : fences[0].start()].strip()
    paragraphs = [p.strip() for p in head.split("\n\n") if p.strip()]
    standard = bool(name_match and paragraphs and paragraphs[0].startswith("This page is the Transformers contract"))
    heading, text, block, code = blocks[0]
    slots = {
        "title": title,
        "name": name_match.group(1) if name_match else title.split(":")[0],
        "intro": None if standard else (paragraphs[0] if paragraphs else None),
        "preamble": "\n\n".join(paragraphs[1:]),
        "block": block,
        "heading": heading,
        "text": text,
        **_first_block(code),
        "sections": [],
        "training": None,
        "record_setup": "",
        "checked": [],
        "outputs": {},
        "footer": usage[fences[-1].end() :].strip(),
    }
    later = blocks[1:]
    trains = [i for i, b in enumerate(later) if re.search(r"^model\.train\(\)$", b[3], re.M)]
    if trains and trains[0] != len(later) - 1:
        raise ValueError("sections after the training step")
    for heading_i, text_i, _, code_i in later[: trains[0] if trains else None]:
        slots["sections"].append({"heading": heading_i, "text": text_i, "code": code_i.strip("\n")})
    if trains:
        training, record, source = _training_block(later[-1][3])
        training["text"] = later[-1][1]
        slots["training"] = training
    else:
        # The record closes the last block: split it off.
        owner = slots["sections"][-1] if slots["sections"] else None
        source = owner["code"] if owner else slots["inference"]
        body = ast.parse(source).body
        index = next((i for i, s in enumerate(body) if isinstance(s, ast.If) and "MIC_RUNNER" in ast.unparse(s.test)), None)
        if index is None:
            raise ValueError("no MIC_RUNNER result record")
        rest = _join(source, body[:index])
        if owner:
            owner["code"] = rest
        else:
            slots["inference"] = rest
        record = body[index]
    checked, setup, outputs, extra = _record_slots(record, source)
    slots["checked"], slots["outputs"] = checked, outputs
    slots["record_setup"] = _join(source, setup) if setup else ""
    if extra:
        slots["training"]["record"] = extra
    return slots


# ---------------------------------------------------------------- planner replies

SLOTS_SCHEMA = {
    "type": "object",
    "required": ["title", "block", "imports", "model_class", "inference", "outputs"],
    "properties": {
        "title": {"type": "string"},
        "name": {"type": "string"},
        "block": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
        "heading": {"type": "string"},
        "text": {"type": "string"},
        "imports": {"type": "string"},
        "setup": {"type": "string"},
        "model_class": {"type": "string", "pattern": "^[A-Za-z_][A-Za-z0-9_]*$"},
        "load_kwargs": {"type": ["string", "null"]},
        "inference": {"type": "string", "minLength": 1},
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["heading", "code"],
                "properties": {"heading": {"type": "string"}, "text": {"type": "string"}, "code": {"type": "string"}},
            },
        },
        "training": {
            "type": ["object", "null"],
            "required": ["loss"],
            "properties": {
                "text": {"type": "string"},
                "inputs": {"type": "string"},
                "loss": {"type": "string", "pattern": "loss"},
                "optimizer": {"type": "string"},
                "record": {"type": "object", "additionalProperties": {"type": "string"}},
            },
        },
        "record_setup": {"type": "string"},
        "checked": {"type": ["array", "string"], "items": {"type": "string"}},
        "outputs": {"type": "object", "minProperties": 1, "additionalProperties": {"type": "string"}},
    },
}
CLI_BUILD = ("seed", "overfit", "accept_insensitive", "accept_unhealthy", "copy_assets", "preserve_frozen_prefixes")
# The manifest fields that need judgment; assemble() writes the rest.
PARTIAL_MANIFEST = ("name", "usage", "architecture", "requires", "resources", "build", "outputs")


def partial_manifest(manifest, fixture):
    """The judgment fields of a full manifest: what a slots reply carries."""
    integration = manifest["integrations"]["transformers"]
    derived = {"runtime", "training.loss", "training.gradients", "training.updated"}
    partial = {
        "name": manifest.get("name"),
        "usage": manifest.get("usage"),
        "architecture": manifest.get("architecture"),
        "requires": integration.get("requires", []),
        # Seed is fixed and the rest are the CLI's choices, never the planner's.
        "build": {k: v for k, v in manifest["fixtures"][fixture].get("build", {}).items() if k not in CLI_BUILD},
        "outputs": {k: {"compare": v["compare"]} for k, v in manifest["outputs"].items() if k not in derived},
    }
    if integration.get("resources"):
        partial["resources"] = integration["resources"]
    return partial


def assemble(partial, slots, model, fixture, fixtures_only=False):
    """The full MIC manifest from a slots reply's partial manifest."""
    rules = {"runtime": {"compare": "exact"}}
    for name, rule in (partial.get("outputs") or {}).items():
        rule = {"compare": rule} if isinstance(rule, str) else dict(rule)
        if rule.get("compare") == "allclose":
            rule.update(atol=1e-4, rtol=1e-4)
        rules[name] = rule
    training = bool(slots.get("training"))
    if training:
        rules["training.loss"] = {"compare": "allclose", "atol": 1e-4, "rtol": 1e-4}
        rules["training.gradients"] = {"compare": "all_true"}
        rules["training.updated"] = {"compare": "true"}
    integration = {
        "entry_point": "integrations/transformers/usage.md",
        "format": "runnable-markdown",
        "requires": partial.get("requires") or ["transformers>=5", "torch"],
        "targets": ["fixtures"] if fixtures_only else ["fixtures", "checkpoints"],
        "variants": {"default": {}, "eager": {"attn_implementation": "eager"}, "bfloat16": {"dtype": "bfloat16"}},
        "training": training,
    }
    if partial.get("resources"):
        integration["resources"] = _resources(partial["resources"], model)
    manifest = {
        "mic": "0.3",
        "name": partial.get("name") or slots["title"],
        "usage": partial.get("usage") or slots["title"],
        "checkpoints": {model: {"fixture": fixture}},
        "fixtures": {fixture: {"built_from": model, "covers": [model], "build": {"seed": 17, **(partial.get("build") or {})}}},
        "integrations": {"transformers": integration},
        "outputs": rules,
    }
    if partial.get("architecture"):
        manifest["architecture"] = partial["architecture"]
    return manifest


def _resources(resources, model):
    """The schema's resource fields only: they are advisory, and an invented key
    (download_200) must not cost a draft."""
    from .commands import manifest_schema

    entry = manifest_schema()["properties"]["integrations"]["additionalProperties"]["properties"]["resources"]
    known = set(entry["additionalProperties"]["properties"])
    kept = {}
    for target, fields in resources.items() if isinstance(resources, dict) else ():
        if target == "fixture" or target == model:
            fields = {k: v for k, v in (fields or {}).items() if k in known}
            if fields:
                kept[target] = fields
    return kept


def expand(reply, model, fixture, fixtures_only=False):
    """A planner reply with slots, as the full plan every check reads (manifest + usage)."""
    import jsonschema

    slots = reply["slots"]
    jsonschema.validate(slots, SLOTS_SCHEMA)
    # The record's runtime and training entries are the template's.
    reserved = ("runtime", "training", "loss", "gradients", "updated")
    outputs = {k: v for k, v in slots["outputs"].items() if k not in reserved and not k.startswith("training.")}
    slots = {"name": reply.get("name"), **slots, "outputs": outputs}
    if reply.get("manifest", {}).get("outputs"):
        reply = {**reply, "manifest": {**reply["manifest"], "outputs": {
            k: v for k, v in reply["manifest"]["outputs"].items() if k not in reserved and not k.startswith("training.")}}}
    return {
        **{k: v for k, v in reply.items() if k != "slots"},
        "manifest": assemble(reply.get("manifest") or {}, slots, model, fixture, fixtures_only),
        "usage": render(slots, model),
    }


def from_usage_reply(reply, fixture):
    """A full-usage reply in slots mode, as a slots reply; None if the page does not fit.

    Drafts asked for slots still sometimes write the whole page; the template then
    rewrites its scaffold instead of rejecting it. Its manifest keeps the judgment
    fields only, whatever else it has.
    """
    try:
        slots = extract(reply["usage"])
    except (ValueError, SyntaxError, StopIteration, KeyError, AttributeError):
        return None
    manifest = reply.get("manifest") or {}
    integration = (manifest.get("integrations") or {}).get("transformers") or {}
    build = next(iter((manifest.get("fixtures") or {}).values()), {}).get("build") or {}
    partial = {
        **{k: manifest[k] for k in ("name", "usage", "architecture") if k in manifest},
        "requires": integration.get("requires") or manifest.get("requires"),
        "resources": integration.get("resources") or manifest.get("resources"),
        "build": {k: v for k, v in (manifest.get("build") or build).items() if k not in CLI_BUILD},
        "outputs": manifest.get("outputs") or {},
    }
    return {**{k: v for k, v in reply.items() if k != "usage"}, "slots": slots,
            "manifest": {k: v for k, v in partial.items() if v is not None}}


def as_slots(plan, fixture):
    """An approved plan (manifest + usage) in the slots form, for the planner's example; None if it does not fit."""
    try:
        slots = extract(plan["usage"])
    except (ValueError, SyntaxError, StopIteration, KeyError, AttributeError):
        return None
    return {**{k: v for k, v in plan.items() if k not in ("usage", "manifest")}, "slots": slots,
            "manifest": partial_manifest(plan["manifest"], fixture)}

