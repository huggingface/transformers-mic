"""Chat templates see the runner's fixed date, not today's."""

import datetime
import importlib.util
import json
import os
import subprocess
import sys

import pytest

from transformers_mic import resources

BACKEND = resources.backend()
RENDER = (
    "from transformers.utils.chat_template_utils import _compile_jinja_template\n"
    "import datetime, json\n"
    "print(json.dumps([_compile_jinja_template(\"{{ strftime_now('%Y-%m-%d') }}\").render(),"
    " datetime.datetime.now().year]))\n"
)


def run(code, env=None):
    # A fresh interpreter: freezing installs an import hook and patches a module,
    # both process-wide.
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def runner():
    spec = importlib.util.spec_from_file_location("mic_template_runner", BACKEND / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def freeze_then(value, before=""):
    return (
        f"import sys\nsys.path.insert(0, {str(BACKEND)!r})\n{before}"
        f"import template_date\ntemplate_date.freeze({value!r})\n{RENDER}"
    )


# requires_runtime, not importorskip: example/transformers/ (the fleet lock folder) makes
# a bare `import transformers` succeed as a namespace package once example/ is on sys.path.
@pytest.mark.requires_runtime
@pytest.mark.parametrize("imported_first", [False, True])
def test_templates_see_the_frozen_date_whenever_transformers_is_imported(imported_first):
    before = "import transformers.utils.chat_template_utils\n" if imported_first else ""
    rendered, year = run(freeze_then("2026-01-01", before))
    assert rendered == "2026-01-01"
    # Only the template module's clock is frozen.
    assert year == datetime.datetime.now().year


def test_freezing_does_not_import_transformers():
    code = (
        f"import sys\nsys.path.insert(0, {str(BACKEND)!r})\n"
        "import template_date\ntemplate_date.freeze('2026-01-01')\n"
        "import json\nprint(json.dumps(['transformers' in sys.modules]))\n"
    )
    assert run(code) == [False]


def test_a_malformed_date_is_refused():
    sys.path.insert(0, str(BACKEND))
    try:
        import template_date

        with pytest.raises(ValueError):
            template_date.freeze("January 1st")
    finally:
        sys.path.remove(str(BACKEND))


@pytest.mark.requires_runtime
def test_isolated_contract_processes_freeze_the_date(tmp_path, monkeypatch):
    module = runner()
    # tmic points HERE at a workspace; the prelude must still find the runner's code.
    monkeypatch.setattr(module, "HERE", tmp_path / "workspace")
    prelude = module.patch_prelude([])
    assert str(module.CODE) in prelude and str(tmp_path) not in prelude
    env = {**os.environ, "MIC_TEMPLATE_DATE": "2026-01-01"}
    rendered, _ = run(prelude + RENDER, env)
    assert rendered == "2026-01-01"


def test_baselines_record_the_template_date():
    sys.path.insert(0, str(BACKEND))
    try:
        import template_date
    finally:
        sys.path.remove(str(BACKEND))
    schema = json.loads((BACKEND / "schemas" / "baseline.schema.json").read_text())
    recorded = schema["properties"]["recorded_with"]["properties"]
    assert "template_date" in recorded
    datetime.date.fromisoformat(template_date.TEMPLATE_DATE)
