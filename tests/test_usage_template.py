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
