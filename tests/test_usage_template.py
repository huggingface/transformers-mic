import json
from pathlib import Path

import pytest

from transformers_mic import resources, usage_template

CATALOG = sorted((resources.catalog() / "hub").glob("*-integration"))


@pytest.mark.parametrize("folder", CATALOG, ids=lambda f: f.name)
def test_every_catalog_contract_round_trips_through_its_slots(folder):
    usage = (folder / "integrations/transformers/usage.md").read_text()
    checkpoint = next(iter(json.loads((folder / "integration.json").read_text())["checkpoints"]))
    slots = usage_template.extract(usage)
    rendered = usage_template.render(slots, checkpoint)
    # Rendering is stable: the slots of a rendered page render it again exactly.
    assert usage_template.render(usage_template.extract(rendered), checkpoint) == rendered
    for code in [slots["inference"], *(s["code"] for s in slots["sections"])]:
        assert code.strip() in rendered


def test_render_writes_the_scaffold():
    slots = {
        "title": "Tiny: fill-mask", "name": "Tiny", "block": "fill_mask", "heading": "Predict",
        "imports": "from transformers import AutoModelForMaskedLM, AutoTokenizer",
        "setup": "tokenizer = AutoTokenizer.from_pretrained(model_id)",
        "model_class": "AutoModelForMaskedLM",
        "inference": "logits = model(**tokenizer([\"a\"], return_tensors=\"pt\").to(device)).logits",
        "training": {"inputs": "labels = None", "loss": "model(**inputs, labels=labels).loss"},
        "checked": ["lm_head.weight"],
        "outputs": {"logits": "logits.float().tolist()"},
    }
    page = usage_template.render(slots, "org/tiny")
    assert page.count("```py runnable:fill_mask") == 2 and "```py runnable:fill_mask:2" in page
    for line in ('import torch', 'from transformers import set_seed', 'model_id = "org/tiny"',
                 'model = AutoModelForMaskedLM.from_pretrained(model_id, **variant).to(device)',
                 'checked = ("lm_head.weight",)', '"logits": logits.float().tolist(),', 'optimizer.step()'):
        assert line in page


def _plan(record, training="loss = model(**inputs).loss"):
    return f"""# T

## Run

```py runnable:x
import torch
from transformers import AutoModel
import json  # doc-builder: hide
import os  # doc-builder: hide

model_id = "org/m"
model_id = os.environ.get("MIC_MODEL", model_id)  # doc-builder: hide
variant = json.loads(os.environ.get("MIC_MODEL_KWARGS", "{{}}"))  # doc-builder: hide
device = "cpu"
model = AutoModel.from_pretrained(model_id, **{{"dtype": torch.float32, **variant}}).to(device)
out = model(**inputs)
```

## One training step

```py runnable:x:2
model.train()
optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
{training}
loss.backward()
before = {{n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}}  # doc-builder: hide
optimizer.step()
print(f"loss: {{loss.item():.4f}}")

{record}
```
"""


def test_planner_written_variations_extract_and_render():
    record = '''if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    named = dict(model.named_parameters())
    experts = next(n for n in named if n.endswith("experts.gate_up_proj"))
    checked = ("lm_head.weight", experts)
    result = {
        "runtime": {"device": model.device.type, "dtype": str(model.dtype), "attn_implementation": "x"},
        "probe": out.last_hidden_state[0, 0, :4].tolist(),
        "training": {"loss": loss.item(), "gradients": {}, "frozen": 1, "updated": True},
    }
    print(json.dumps(result))'''
    slots = usage_template.extract(_plan(record))
    assert slots["load_kwargs"] == '{"dtype": torch.float32}'
    assert slots["training"]["optimizer"] == "trainable" and slots["training"]["record"] == {"frozen": "1"}
    page = usage_template.render(slots, "org/m")
    # checked uses `experts`, which the record computes first.
    assert page.index("experts = next(") < page.index('checked = ("lm_head.weight", experts)')
    assert '**{"dtype": torch.float32, **variant}' in page and "del model  # doc-builder: hide" in page
    assert usage_template.render(usage_template.extract(page), "org/m") == page


def test_checked_defined_by_the_training_code_is_not_overwritten():
    record = '''if os.environ.get("MIC_RUNNER"):  # doc-builder: hide
    result = {"runtime": {}, "probe": 1, "training": {"loss": 0, "gradients": {}, "updated": True}}
    print(json.dumps(result))'''
    training = 'named = dict(model.named_parameters())\nchecked = ("lm_head.weight",)\nloss = model(**inputs).loss'
    page = usage_template.render(usage_template.extract(_plan(record, training)), "org/m")
    assert "checked = ()" not in page and page.count("checked = (") == 1


def test_slots_reply_expands_to_a_valid_plan():
    from transformers_mic import commands

    folder = next(f for f in CATALOG if f.name == "T5-integration")
    manifest = json.loads((folder / "integration.json").read_text())
    usage = (folder / "integrations/transformers/usage.md").read_text()
    model = next(iter(manifest["checkpoints"]))
    fixture = manifest["checkpoints"][model]["fixture"]
    reply = usage_template.as_slots({"name": "T5", "manifest": manifest, "usage": usage, "rationale": "", "historical_candidates": []}, fixture)
    reply["slots"]["outputs"]["training.loss"] = "loss.item()"  # reserved: the template writes it
    plan = usage_template.expand(reply, model, fixture)
    commands.validate_plan(plan, model, fixture)
    assert plan["manifest"]["outputs"]["training.loss"]["atol"] == 1e-4
    # A full page in slots mode is converted, not rejected.
    converted = usage_template.from_usage_reply({"name": "T5", "manifest": manifest, "usage": usage, "rationale": "", "historical_candidates": []}, fixture)
    commands.validate_plan(usage_template.expand(converted, model, fixture), model, fixture)


def test_a_runaway_nested_reply_is_not_a_draft():
    from types import SimpleNamespace

    from transformers_mic import planner

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[" * 100000), finish_reason="length")], usage=None)
    with pytest.raises(planner.MalformedDraft, match="nests too deeply"):
        planner.parse(response, "m", "p", None)


def test_slot_slips_from_planner_eval():
    import copy

    folder = next(f for f in CATALOG if f.name == "T5-integration")
    manifest = json.loads((folder / "integration.json").read_text())
    usage = (folder / "integrations/transformers/usage.md").read_text()
    model = next(iter(manifest["checkpoints"]))
    fixture = manifest["checkpoints"][model]["fixture"]
    reply = usage_template.as_slots({"name": "T5", "manifest": manifest, "usage": usage, "rationale": "", "historical_candidates": []}, fixture)
    # A suffix lookup inside the checked list is code (plbart, FlexOlmo).
    lookup = 'next(n for n in names if n.endswith("SelfAttention.q.weight"))'
    reply["slots"]["checked"] = [lookup, "lm_head.weight"]
    # Setup that reads the model runs after the load (Mistral-Large-4).
    reply["slots"]["setup"] += "\nvocab = model.config.vocab_size"
    reply["manifest"]["build"] = {"config_override": {"d_model": 64}}
    page = usage_template.expand(reply, model, fixture)
    assert f'checked = ({lookup}, "lm_head.weight")' in page["usage"]
    assert page["usage"].index("vocab = model.config") > page["usage"].index("model = AutoModelForSeq2SeqLM.from_pretrained(model_id)")
    assert page["manifest"]["fixtures"][fixture]["build"]["config_overrides"] == {"d_model": 64}
    # A statement in an output value is a draft error, not a crash.
    broken = copy.deepcopy(reply)
    broken["slots"]["outputs"]["logits"] = "x = 1"
    with pytest.raises(ValueError, match="one Python expression"):
        usage_template.expand(broken, model, fixture)
