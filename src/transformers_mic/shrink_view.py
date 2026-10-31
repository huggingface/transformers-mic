"""tmic shrink: show how a slim fixture was shrunk from its source checkpoint.

Both models are built on the meta device from their configs (no weights are
downloaded), so the comparison reads the modules Transformers actually builds:
which layer kinds each stack has, which module classes exist, and where the
parameters sit. Config changes and preserved properties come from the configs.
"""

from collections import Counter
import html
import json
from pathlib import Path
import re
import sys

# Module attributes that tell layers of the same classes apart (Gemma 3's sliding
# and full attention share one class).
KIND_ATTRIBUTES = (
    "attention_type",
    "layer_type",
    "is_sliding",
    "is_causal",
    "mixer_type",
)
TYPE_NAMES = ("attention_type", "layer_type", "mixer_type")
# Fields expected to survive shrinking unchanged.
KEPT_FIELDS = (
    "model_type",
    "vocab_size",
    "tie_word_embeddings",
    "hidden_act",
    "max_position_embeddings",
    "rope_theta",
    "image_size",
    "patch_size",
    "num_channels",
    "num_mel_bins",
    "sampling_rate",
)
NOISE = {
    "_name_or_path",
    "transformers_version",
    "architectures",
    "dtype",
    "torch_dtype",
}
PALETTE = (
    ("#2a78d6", "#3987e5"),
    ("#eb6834", "#d95926"),
    ("#1baf7a", "#199e70"),
    ("#eda100", "#c98500"),
    ("#e87ba4", "#d55181"),
    ("#008300", "#008300"),
    ("#6250d6", "#9085e9"),
    ("#e34948", "#e66767"),
)
OTHER = ("#8a8984", "#6f6e69")


def load(fixture, source=None):
    """The fixture's build record and the fixture and source configs."""
    from huggingface_hub import hf_hub_download, snapshot_download

    path = Path(fixture)
    if not path.is_dir():
        repo, _, revision = fixture.partition("@")
        path = Path(
            snapshot_download(
                repo,
                revision=revision or None,
                allow_patterns=["build.json", "config.json"],
            )
        )
    build = (
        json.loads((path / "build.json").read_text())
        if (path / "build.json").exists()
        else {}
    )
    if source:
        repo, _, revision = source.partition("@")
    elif build.get("built_from"):
        repo, revision = build["built_from"]["repo"], build["built_from"]["revision"]
    else:
        raise ValueError(
            "The fixture's build.json names no source checkpoint; pass --source REPO[@REVISION]"
        )
    original = config_dict(
        Path(hf_hub_download(repo, "config.json", revision=revision or None)).parent
    )
    shrunk = config_dict(path)
    return {
        "fixture": fixture,
        "source": f"{repo}@{revision}" if revision else repo,
        "build": build,
        "original": original,
        "shrunk": shrunk,
    }


def config_dict(directory):
    """The config as this Transformers reads it, so both sides use one schema
    (v5 moves rope_theta into rope_parameters; non-finite floats are encoded)."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(directory).to_dict()


def build_meta(values, class_name):
    import torch
    import transformers

    # A slim fixture is dense whatever the checkpoint's format.
    values = {k: v for k, v in values.items() if k != "quantization_config"}
    config = transformers.AutoConfig.for_model(**values)
    model_class = getattr(transformers, class_name, None) if class_name else None
    if model_class is None:
        model_class = transformers.AutoModel._model_mapping[type(config)]
    with torch.device("meta"):
        return model_class._from_config(config)


def stacks(model):
    """Outermost ModuleLists of blocks: name -> list of layers."""
    from torch import nn

    found = {}
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if any(name.startswith(f"{outer}.") for outer in found):
            continue
        if all(any(True for _ in layer.children()) for layer in module):
            found[name] = list(module)
    return found


def signature(layer):
    features = set()
    for module in layer.modules():
        features.add(type(module).__name__)
        for key in KIND_ATTRIBUTES:
            value = getattr(module, key, None)
            if isinstance(value, (str, bool)) and not isinstance(
                getattr(type(module), key, None), property
            ):
                features.add(f"{key}={value}")
    return frozenset(features)


def label_kinds(signatures, prefix):
    """Name each kind by what sets it apart from the stack's other kinds."""
    kinds = list(dict.fromkeys(signatures))
    common = frozenset.intersection(*kinds) if kinds else frozenset()
    labels = {}
    for kind in kinds:
        own = sorted(kind - common, key=lambda f: ("=" not in f, f))
        if len(kinds) == 1:
            own = [max(kind, key=lambda f: ("Layer" in f or "Block" in f, -len(f)))]
        elif not own:
            others = sorted(set().union(*kinds) - kind)
            own = [f"no {short(others[0], prefix)}"] if others else ["base"]
        if any(f.split("=")[0] in TYPE_NAMES for f in own):
            # A type name says it; is_sliding/is_causal only repeat it.
            own = [f for f in own if not f.endswith(("=True", "=False"))]
        own = [f.split("=", 1)[1] if f.split("=")[0] in TYPE_NAMES else f for f in own]
        labels[kind] = ", ".join(short(f, prefix) for f in own[:3]) + (
            " …" if len(own) > 3 else ""
        )
    return labels


def short(name, prefix):
    return (
        name[len(prefix) :]
        if prefix and name.startswith(prefix) and len(name) > len(prefix)
        else name
    )


def component(param, stack_names):
    for stack in stack_names:
        if param.startswith(f"{stack}."):
            return stack
    parts = param.split(".")[:-1]
    return ".".join(parts[:2]) or param


def nested(key, value):
    return (
        isinstance(value, dict)
        and key != "quantization_config"
        and ("model_type" in value or key.endswith("_config"))
    )


def flatten(config, prefix=""):
    """Leaf fields, with nested configs as dotted keys."""
    out = {}
    for key, value in config.items():
        if key in NOISE:
            continue
        if nested(key, value):
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def configs(config, prefix=""):
    yield prefix.rstrip(".") or "model", config
    for key, value in config.items():
        if nested(key, value):
            yield from configs(value, f"{prefix}{key}.")


def attention_style(heads, kv):
    if not (isinstance(heads, int) and isinstance(kv, int) and heads and kv):
        return None
    return "MHA" if kv >= heads else "MQA" if kv == 1 else "GQA"


def properties(original, shrunk):
    """Preserved-or-not checks the reader cares about, per (sub)config."""
    rows = []
    shrunk_configs = dict(configs(shrunk))
    for name, before in configs(original):
        after = shrunk_configs.get(name)
        if after is None:
            rows.append(
                {
                    "config": name,
                    "property": "sub-config",
                    "original": "present",
                    "fixture": "missing",
                    "kept": False,
                }
            )
            continue
        style = attention_style(
            before.get("num_attention_heads"), before.get("num_key_value_heads")
        )
        if style:
            new = attention_style(
                after.get("num_attention_heads"), after.get("num_key_value_heads")
            )
            rows.append(
                {
                    "config": name,
                    "property": "attention style",
                    "original": f"{style} ({before['num_attention_heads']} heads, {before['num_key_value_heads']} KV)",
                    "fixture": f"{new} ({after.get('num_attention_heads')} heads, {after.get('num_key_value_heads')} KV)",
                    "kept": new == style,
                }
            )
        experts = next(
            (
                k
                for k in ("n_routed_experts", "num_local_experts", "num_experts")
                if isinstance(before.get(k), int)
            ),
            None,
        )
        top = before.get("num_experts_per_tok")
        if experts and isinstance(top, int) and before[experts]:

            def routed(c):
                return c.get("num_experts_per_tok", 0) < c.get(experts, 0)

            rows.append(
                {
                    "config": name,
                    "property": "MoE routing",
                    "original": f"top-{top} of {before[experts]} experts",
                    "fixture": f"top-{after.get('num_experts_per_tok')} of {after.get(experts)} experts",
                    "kept": routed(before) == routed(after),
                }
            )
        rope, new_rope = (
            before.get("rope_parameters") or {},
            after.get("rope_parameters") or {},
        )
        if isinstance(rope, dict) and rope.get("rope_type"):
            rows.append(
                {
                    "config": name,
                    "property": "RoPE type",
                    "original": rope.get("rope_type"),
                    "fixture": new_rope.get("rope_type"),
                    "kept": rope.get("rope_type") == new_rope.get("rope_type"),
                }
            )
        for key in KEPT_FIELDS:
            if key in before:
                rows.append(
                    {
                        "config": name,
                        "property": key,
                        "original": before[key],
                        "fixture": after.get(key),
                        "kept": before[key] == after.get(key),
                    }
                )
    return rows


def changes(original, shrunk):
    before, after = flatten(original), flatten(shrunk)
    rows = []
    for key in sorted(set(before) | set(after)):
        old, new = before.get(key, "—"), after.get(key, "—")
        if old != new:
            rows.append({"field": key, "original": brief(old), "fixture": brief(new)})
    return rows, sum(before.get(k) == after.get(k) for k in before)


def brief(value):
    if isinstance(value, list) and len(value) > 6:
        counts = Counter(json.dumps(v) for v in value)
        if len(counts) <= 4:
            return f"{len(value)} entries: " + ", ".join(
                f"{n}× {json.loads(v)}" for v, n in counts.most_common()
            )
        return f"{len(value)} entries"
    text = json.dumps(value) if not isinstance(value, str) else value
    return text if len(text) <= 80 else text[:77] + "…"


def analyze(data):
    build = data["build"]
    class_name = (
        build.get("model_class") or (data["shrunk"].get("architectures") or [None])[0]
    )
    report = {
        "fixture": data["fixture"],
        "source": data["source"],
        "model_class": class_name,
        "weights": build.get("weights"),
        "build_shrink": build.get("shrink"),
        "overrides": (build.get("recipe") or {}).get("config_overrides"),
    }
    report["config_changes"], report["config_unchanged"] = changes(
        data["original"], data["shrunk"]
    )
    report["properties"] = properties(data["original"], data["shrunk"])
    try:
        original = build_meta(data["original"], class_name)
        shrunk = build_meta(data["shrunk"], class_name)
    except Exception as exc:  # the comparison still has the configs
        report["modules_error"] = f"{type(exc).__name__}: {exc}"[:300]
        return report
    prefix = re.split(r"For|Model", type(original).__name__)[0]
    report["prefix"] = prefix

    def classes(model):
        return Counter(type(m).__name__ for m in model.modules())

    before, after = classes(original), classes(shrunk)
    report["classes"] = [
        {"class": name, "original": before[name], "fixture": after.get(name, 0)}
        for name in sorted(
            set(before) | set(after), key=lambda n: (-before.get(n, 0), n)
        )
    ]

    stacks_before, stacks_after = stacks(original), stacks(shrunk)
    report["stacks"] = []
    for name in list(dict.fromkeys([*stacks_before, *stacks_after])):
        old = [signature(layer) for layer in stacks_before.get(name, [])]
        new = [signature(layer) for layer in stacks_after.get(name, [])]
        labels = label_kinds(old + new, prefix)
        order = list(dict.fromkeys(old + new))
        report["stacks"].append(
            {
                "name": name,
                "kinds": [
                    {
                        "label": labels[k],
                        "original": old.count(k),
                        "fixture": new.count(k),
                    }
                    for k in order
                ],
                "original": [order.index(k) for k in old],
                "fixture": [order.index(k) for k in new],
            }
        )

    def params(model, names):
        counts = Counter()
        for name, param in model.named_parameters():
            counts[component(name, names)] += param.numel()
        return counts

    p_before, p_after = (
        params(original, list(stacks_before)),
        params(shrunk, list(stacks_after)),
    )
    report["parameters"] = {
        "original": sum(p_before.values()),
        "fixture": sum(p_after.values()),
    }
    report["components"] = [
        {
            "name": name,
            "original": p_before.get(name, 0),
            "fixture": p_after.get(name, 0),
        }
        for name in sorted(
            set(p_before) | set(p_after), key=lambda n: -p_before.get(n, 0)
        )
    ]
    return report


def verdict(report):
    """Findings a reviewer should look at: what the original has and the fixture lost."""
    findings = []
    for row in report.get("classes", []):
        if row["original"] and not row["fixture"]:
            findings.append(f"module class {row['class']} is not in the fixture")
    for stack in report.get("stacks", []):
        for kind in stack["kinds"]:
            if kind["original"] and not kind["fixture"]:
                findings.append(
                    f"{stack['name']}: layer kind '{kind['label']}' is not in the fixture"
                )
            if kind["fixture"] and not kind["original"]:
                findings.append(
                    f"{stack['name']}: layer kind '{kind['label']}' exists only in the fixture"
                )
    for row in report.get("properties", []):
        if not row["kept"]:
            findings.append(
                f"{row['config']}: {row['property']} {row['original']} became {row['fixture']}"
            )
    return findings


def human(n):
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= size:
            return f"{n / size:.1f}{unit}"
    return str(n)


def text(report):
    lines = [f"{report['fixture']} <- {report['source']} ({report['model_class']})"]
    if report.get("parameters"):
        p = report["parameters"]
        lines.append(f"parameters: {human(p['original'])} -> {human(p['fixture'])}")
    for stack in report.get("stacks", []):
        kinds = "; ".join(
            f"{k['label']} {k['original']}->{k['fixture']}" for k in stack["kinds"]
        )
        lines.append(
            f"{stack['name']}: {len(stack['original'])} -> {len(stack['fixture'])} layers ({kinds})"
        )
    if report.get("modules_error"):
        lines.append(f"modules not compared: {report['modules_error']}")
    findings = verdict(report)
    lines.append(
        f"{len(findings)} finding(s)" + "".join(f"\n  - {f}" for f in findings)
    )
    return "\n".join(lines)


def render(report):
    def e(value):
        return html.escape(str(value))

    def color(i):
        return i if i < len(PALETTE) else len(PALETTE)

    findings = verdict(report)
    prefix = report.get("prefix", "")
    out = []

    if report.get("parameters"):
        p = report["parameters"]
        out.append(
            f'<div class="stats"><div class="stat"><span class="k">Original</span><span class="v">{human(p["original"])}</span>'
            f'<span class="k">parameters</span></div><div class="stat"><span class="k">Fixture</span>'
            f'<span class="v">{human(p["fixture"])}</span><span class="k">parameters</span></div>'
            f'<div class="stat"><span class="k">Layer stacks</span><span class="v">{len(report["stacks"])}</span>'
            f'<span class="k">compared</span></div><div class="stat"><span class="k">Findings</span>'
            f'<span class="v">{len(findings)}</span><span class="k">to review</span></div></div>'
        )
    if findings:
        out.append(
            '<section class="findings warn"><h2>⚠ To review</h2><ul>'
            + "".join(f"<li>{e(f)}</li>" for f in findings)
            + "</ul></section>"
        )
    else:
        out.append(
            '<section class="findings ok"><h2>✓ Same architecture, smaller</h2><p>Every module class, layer kind and '
            "checked property of the original is in the fixture.</p></section>"
        )
    if report.get("modules_error"):
        out.append(
            f'<section class="findings warn"><h2>⚠ Modules not compared</h2><p>{e(report["modules_error"])}</p></section>'
        )

    if report.get("stacks"):
        out.append(
            '<h2>Layer stacks</h2><p class="note">One square per layer, colored by kind: the layer\'s module classes '
            "and type attributes. Hover a square for its index and kind.</p>"
        )
        for stack in report["stacks"]:

            def cells(kinds):
                return "".join(
                    f'<i class="c{color(k)}" title="layer {i}: {e(stack["kinds"][k]["label"])}"></i>'
                    for i, k in enumerate(kinds)
                )

            legend = "".join(
                f'<tr><td><i class="c{color(i)}"></i>{e(k["label"])}</td><td>{k["original"]}</td><td>{k["fixture"]}</td>'
                f"<td>{'✓ kept' if k['fixture'] and k['original'] else '⚠ missing' if k['original'] else '⚠ new'}</td></tr>"
                for i, k in enumerate(stack["kinds"])
            )
            out.append(
                f'<div class="stack"><h3>{e(stack["name"])}</h3>'
                f'<div class="row"><span class="lab">Original · {len(stack["original"])}</span><div class="cells">{cells(stack["original"])}</div></div>'
                f'<div class="row"><span class="lab">Fixture · {len(stack["fixture"])}</span><div class="cells">{cells(stack["fixture"])}</div></div>'
                f'<table class="legend"><tr><th>Kind</th><th>Original</th><th>Fixture</th><th></th></tr>{legend}</table></div>'
            )

    if report.get("components"):
        rows = report["components"]
        shown = rows[: len(PALETTE) - 1]
        rest = rows[len(PALETTE) - 1 :]
        if rest:
            shown = shown + [
                {
                    "name": f"{len(rest)} others",
                    "original": sum(r["original"] for r in rest),
                    "fixture": sum(r["fixture"] for r in rest),
                }
            ]
        p = report["parameters"]

        def bar(key):
            total = p[key] or 1
            return "".join(
                f'<i class="c{color(i)}" style="flex:{r[key] / total:.6f}" title="{e(r["name"])}: '
                f'{human(r[key])} ({100 * r[key] / total:.1f}%)"></i>'
                for i, r in enumerate(shown)
                if r[key]
            )

        table = "".join(
            f'<tr><td><i class="c{color(i)}"></i>{e(r["name"])}</td><td>{human(r["original"])}</td>'
            f"<td>{100 * r['original'] / (p['original'] or 1):.1f}%</td><td>{human(r['fixture'])}</td>"
            f"<td>{100 * r['fixture'] / (p['fixture'] or 1):.1f}%</td></tr>"
            for i, r in enumerate(shown)
        )
        out.append(
            '<h2>Where the parameters are</h2><p class="note">Share of each model\'s parameters per component. '
            "Vocabulary and patch sizes are kept, so embeddings usually take a larger share of the fixture.</p>"
            f'<div class="row"><span class="lab">Original</span><div class="bar">{bar("original")}</div></div>'
            f'<div class="row"><span class="lab">Fixture</span><div class="bar">{bar("fixture")}</div></div>'
            '<table class="legend"><tr><th>Component</th><th>Original</th><th>Share</th><th>Fixture</th><th>Share</th></tr>'
            f"{table}</table>"
        )

    if report.get("properties"):
        rows = "".join(
            f'<tr class="{"" if r["kept"] else "bad"}"><td>{e(r["config"])}</td><td>{e(r["property"])}</td>'
            f"<td>{e(r['original'])}</td><td>{e(r['fixture'])}</td><td>{'✓ kept' if r['kept'] else '⚠ changed'}</td></tr>"
            for r in report["properties"]
        )
        out.append(
            "<h2>Preserved properties</h2><table><tr><th>Config</th><th>Property</th><th>Original</th>"
            f"<th>Fixture</th><th></th></tr>{rows}</table>"
        )

    if report.get("classes"):
        rows = "".join(
            f'<tr class="{"bad" if r["original"] and not r["fixture"] else ""}"><td>{e(short(r["class"], prefix))}</td>'
            f"<td>{r['original']}</td><td>{r['fixture']}</td></tr>"
            for r in report["classes"]
        )
        out.append(
            f'<h2>Module classes</h2><p class="note">Instances of each class in the built model'
            f"{f' (prefix {e(prefix)} omitted)' if prefix else ''}.</p><table><tr><th>Class</th><th>Original</th>"
            f"<th>Fixture</th></tr>{rows}</table>"
        )

    rows = "".join(
        f"<tr><td>{e(r['field'])}</td><td>{e(r['original'])}</td><td>{e(r['fixture'])}</td></tr>"
        for r in report["config_changes"]
    )
    out.append(
        f'<h2>Config changes</h2><p class="note">{len(report["config_changes"])} fields changed, '
        f"{report['config_unchanged']} unchanged.</p><table><tr><th>Field</th><th>Original</th><th>Fixture</th></tr>{rows}</table>"
    )

    swatches = "".join(
        f".c{i}{{background:var(--s{i})}}" for i in range(len(PALETTE) + 1)
    )
    light = "".join(f"--s{i}:{c[0]};" for i, c in enumerate((*PALETTE, OTHER)))
    dark = "".join(f"--s{i}:{c[1]};" for i, c in enumerate((*PALETTE, OTHER)))
    dark_tokens = f"--bg:#1a1a19;--fg:#f0efec;--muted:#c3c2b7;--line:#3a3936;--ok:#2f9e5a;--warn:#d9822b;{dark}"
    meta = (
        f"{e(report['fixture'])} ← {e(report['source'])} · {e(report['model_class'])}"
    )
    if report.get("weights"):
        meta += f" · {e(report['weights'])} weights"
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shrunk architecture</title>
<style>
:root{{--bg:#fcfcfb;--fg:#0b0b0b;--muted:#52514e;--line:#e4e3df;--ok:#1d7a43;--warn:#b35c00;{light}}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{{dark_tokens}}}}}
:root[data-theme="dark"]{{{dark_tokens}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:1040px;margin:0 auto;padding:24px 16px 64px}}
h1{{font-size:24px;margin:0 0 4px}}h2{{font-size:18px;margin:36px 0 8px}}h3{{font-size:14px;margin:0 0 8px;font-family:ui-monospace,monospace}}
.meta,.note{{color:var(--muted);margin:0 0 12px;overflow-wrap:anywhere}}
.stats{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:20px 0}}
.stat{{border:1px solid var(--line);border-radius:8px;padding:12px;display:flex;flex-direction:column}}
.stat .v{{font-size:26px;font-weight:600}}.stat .k{{color:var(--muted);font-size:13px}}
.findings{{border-radius:8px;padding:4px 16px;border:1px solid var(--line)}}
.findings.ok{{border-left:4px solid var(--ok)}}.findings.warn{{border-left:4px solid var(--warn)}}
.findings h2{{margin:12px 0 4px;font-size:16px}}
.stack{{border:1px solid var(--line);border-radius:8px;padding:12px;margin:12px 0}}
.row{{display:flex;align-items:center;gap:12px;margin:6px 0}}
.lab{{flex:0 0 110px;color:var(--muted);font-size:13px}}
.cells{{display:flex;flex-wrap:wrap;gap:2px}}
.cells i{{width:12px;height:12px;border-radius:2px;display:block}}
.bar{{flex:1;display:flex;gap:2px;height:16px}}.bar i{{display:block;border-radius:2px;min-width:2px}}
table{{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0;display:block;overflow-x:auto}}
th,td{{text-align:left;padding:4px 10px 4px 0;border-bottom:1px solid var(--line);vertical-align:top}}
th{{color:var(--muted);font-weight:500}}
td:first-child{{font-family:ui-monospace,monospace;overflow-wrap:anywhere}}
td i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}}
tr.bad td{{color:var(--warn)}}
{swatches}
</style></head><body><main>
<h1>Shrunk architecture</h1><p class="meta">{meta}</p>
{"".join(out)}
</main></body></html>"""


def run(args):
    from .workspace import Workspace

    fixture = args.fixture
    if not fixture:
        workspace = Workspace(args.workspace)
        name = workspace.fixture
        pin = workspace.lock["contracts"][workspace.state["contract"]]["fixtures"][name]
        if pin.get("path"):
            fixture = str((workspace.lock_path.parent / pin["path"]).resolve())
        else:
            fixture = f"{name}@{pin['revision']}"
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ValueError(
            "tmic shrink builds both models on the meta device: install transformers-mic[runtime]"
        )
    report = analyze(load(fixture, args.source))
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(text(report), file=sys.stderr)
    out = Path(args.html or f"shrink-{Path(fixture.split('@')[0]).name}.html")
    out.write_text(render(report))
    print(f"wrote {out}", file=sys.stderr)
    if args.open:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())
    return 0
