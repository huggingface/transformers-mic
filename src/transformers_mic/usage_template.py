"""Transformers usage.md from a few model-specific slots.

Every approved contract has the same scaffold: the hidden MIC plumbing, the device
and seed, the variant reload, a training step and the hidden result record. Only
the imports, the loading, the inference code, extra sections, the training inputs
and loss, the checked parameters and the outputs differ. The planner writes those
slots and `render` writes the rest, so a draft cannot get the scaffold wrong: the
fences, the plumbing and the result record were where most drafts failed
(google/switch-base-8, 2026-11-05).

Slots (a JSON object):
    title          the page's H1
    name           the contract name in the intro ("BERT")
    intro          optional: replaces the standard intro paragraph
    preamble       optional paragraphs between the intro and the first section
    block          the runnable block name ("fill_mask")
    heading, text  the first section's heading and optional paragraph
    imports        import lines; torch and set_seed are added when missing
    setup          optional code before the model loads (processor, data files)
    model_class    the class loading the checkpoint
    inference      the first block's code after the model loads
    sections       [{"heading", "text", "code"}] extra runnable sections, in order
    training       {"text", "inputs", "loss"} or null; loss is an expression
    checked        parameter names whose gradients the record checks (training only)
    outputs        {"name": "expression"} in record order, after runtime
    footer         optional closing paragraph
"""

import json
import re

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
TRAINING_RECORD = """        "training": {
            "loss": loss.item(),
            "gradients": {n: bool(named[n].grad is not None and torch.count_nonzero(named[n].grad) > 0 and torch.isfinite(named[n].grad).all()) for n in checked},
            "updated": any(not torch.equal(p, before[n]) for n, p in named.items()),
        },"""
TRAINING_STEP = """model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss = {loss}
loss.backward()
before = {{name: p.detach().clone() for name, p in model.named_parameters()}}{hide}
optimizer.step()
print(f"loss: {{loss.item():.4f}}")"""


def _imports(lines):
    text = lines.strip("\n")
    if not re.search(r"^import torch$", text, re.M):
        text = "import torch\n" + text
    if not re.search(r"\bset_seed\b", text):
        text += "\nfrom transformers import set_seed"
    return text


def _record(slots):
    training = slots.get("training")
    lines = [f"if os.environ.get(\"MIC_RUNNER\"):{HIDE}"]
    if training:
        lines.append("    named = dict(model.named_parameters())")
        names = ", ".join(json.dumps(n) for n in slots["checked"])
        lines.append(f"    checked = ({names}{',' if len(slots['checked']) == 1 else ''})")
    lines.append("    result = {")
    lines.append(RUNTIME)
    lines += [f"        {json.dumps(name)}: {expr}," for name, expr in slots["outputs"].items()]
    if training:
        lines.append(TRAINING_RECORD)
    lines += ["    }", "    print(json.dumps(result))"]
    return "\n".join(lines)


def _section(heading, text, block, code):
    parts = [f"## {heading}"]
    if text:
        parts.append(text.strip())
    parts.append(f"```py runnable:{block}\n{code.strip(chr(10))}\n```")
    return "\n\n".join(parts)


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
        f"model = {model_class}.from_pretrained(model_id).to(device)",
        f"if variant:{HIDE}",
        f"    model = {model_class}.from_pretrained(model_id, **variant).to(device)",
        "",
        slots["inference"].strip("\n"),
    ]
    blocks = [[slots.get("heading", "Run the model"), slots.get("text", ""), "\n".join(first)]]
    for section in slots.get("sections", []):
        blocks.append([section["heading"], section.get("text", ""), section["code"].strip("\n")])
    training = slots.get("training")
    if training:
        step = TRAINING_STEP.format(loss=training["loss"], hide=HIDE)
        inputs = (training.get("inputs") or "").strip("\n")
        blocks.append(["One training step", training.get("text", ""), f"{inputs}\n\n{step}" if inputs else step])
    blocks[-1][2] += "\n\n" + _record(slots)
    sections = [
        _section(heading, text, block if i == 0 else f"{block}:{i + 1}", code)
        for i, (heading, text, code) in enumerate(blocks)
    ]
    intro = slots.get("intro") or INTRO.format(name=slots["name"], checkpoint=checkpoint)
    page = [f"# {slots['title']}", intro, *([slots["preamble"].strip()] if slots.get("preamble") else []), *sections]
    if slots.get("footer"):
        page.append(slots["footer"].strip())
    return "\n\n".join(page) + "\n"


def extract(usage):
    """Slots from an approved usage.md: the inverse of `render` for the shared scaffold.

    Raises ValueError for a page the template cannot express.
    """
    title = re.search(r"^# (.+)$", usage, re.M).group(1)
    fences = list(re.finditer(r"^## (.+?)\n\n(.*?)```py runnable:([\w-]+)[^\n]*\n(.*?)^```\n?", usage, re.M | re.S))
    if not fences:
        raise ValueError("no runnable sections")
    footer = usage[fences[-1].end() :].strip()
    name_match = re.search(r"Transformers contract of the (.+?) MIC", usage)
    head = usage[usage.index("\n", usage.index("# ")) : fences[0].start()].strip()
    paragraphs = [p.strip() for p in head.split("\n\n") if p.strip()]
    standard = name_match and paragraphs and paragraphs[0].startswith("This page is the Transformers contract")
    intro = None if standard else (paragraphs[0] if paragraphs else None)
    preamble = "\n\n".join(paragraphs[1:])
    blocks = [(m.group(1), m.group(2).strip(), m.group(3), m.group(4)) for m in fences]
    heading, text, block, code = blocks[0]
    lines = code.rstrip("\n").split("\n")

    def strip_hidden_plumbing(line):
        return line.strip() in (
            f"import json{HIDE}".strip(),
            f"import os{HIDE}".strip(),
            f'model_id = os.environ.get("MIC_MODEL", model_id){HIDE}'.strip(),
            f'variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{{}}")){HIDE}'.strip(),
        )

    lines = [line for line in lines if not strip_hidden_plumbing(line)]
    model_line = next(i for i, l in enumerate(lines) if re.match(r"model = (\w+)\.from_pretrained\(model_id\)", l))
    model_class = re.match(r"model = (\w+)\.", lines[model_line]).group(1)
    id_line = next(i for i, l in enumerate(lines) if l.startswith("model_id = "))
    device_line = next(i for i, l in enumerate(lines) if l.startswith("device = "))
    imports = "\n".join(lines[:id_line]).strip("\n")
    after = [l for l in lines[device_line + 1 : model_line] if not l.startswith("set_seed(")]
    setup = "\n".join(after).strip("\n")
    rest = lines[model_line + 1 :]
    # The hidden variant reload is the template's, wherever the page put it.
    cleaned, skip = [], 0
    for i, line in enumerate(rest):
        if skip:
            skip -= 1
            continue
        if line.startswith(f"if variant:{HIDE}") and i + 1 < len(rest) and f"model = {model_class}.from_pretrained(model_id, **variant)" in rest[i + 1]:
            skip = 1
            continue
        cleaned.append(line)
    inference = "\n".join(cleaned).strip("\n")

    sections, training, record_code = [], None, None
    for heading_i, text_i, _, code_i in blocks[1:]:
        if heading_i == "One training step":
            pre, _, post = code_i.partition("model.train()\n")
            loss = re.search(r"^loss = (.+)$", post, re.M).group(1)
            training = {"text": text_i, "inputs": pre.strip("\n"), "loss": loss}
            record_code = post
        else:
            sections.append({"heading": heading_i, "text": text_i, "code": code_i})
    if record_code is None:
        last = sections[-1] if sections else None
        source = last["code"] if last else inference
        body, _, record = source.partition(f'if os.environ.get("MIC_RUNNER"):{HIDE}')
        if last:
            last["code"] = body.strip("\n")
        else:
            inference = body.strip("\n")
        record_code = record
    else:
        record_code = record_code.partition(f'if os.environ.get("MIC_RUNNER"):{HIDE}')[2]
    if not record_code:
        raise ValueError("no MIC_RUNNER result record")
    checked = re.search(r"^\s+checked = \((.*)\)$", record_code, re.M)
    outputs = {}
    for match in re.finditer(r'^        "([\w.]+)": (.+),$', record_code, re.M):
        if match.group(1) not in ("runtime", "training", "device", "dtype", "attn_implementation", "loss", "gradients", "updated"):
            outputs[match.group(1)] = match.group(2)
    slots = {
        "title": title,
        "name": name_match.group(1) if name_match else title.split(":")[0],
        "intro": intro,
        "preamble": preamble,
        "block": block,
        "heading": heading,
        "text": text,
        "imports": imports,
        "setup": setup,
        "model_class": model_class,
        "inference": inference,
        "sections": sections,
        "training": training,
        "checked": [n.strip().strip("\"'") for n in checked.group(1).split(",") if n.strip()] if checked else [],
        "outputs": outputs,
        "footer": footer,
    }
    return slots
