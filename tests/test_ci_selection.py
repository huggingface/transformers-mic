"""Backend selection and hardware rules that Transformers CI relies on."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

from transformers_mic import resources


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"mic_{name}_test", resources.backend() / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_contracts = _load("select_contracts")
run = _load("run")


@pytest.mark.parametrize("container", ["dict", "ordered"])
def test_registry_only_addition_selects_affected_models(container):
    path = "src/transformers/models/auto/configuration_auto.py"
    if container == "dict":
        before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\n'
        after = (
            'CONFIG_MAPPING_NAMES = {"new_model": "NewConfig", "bert": "BertConfig"}\n'
        )
    else:
        before = 'CONFIG_MAPPING_NAMES = OrderedDict([("bert", "BertConfig")])\n'
        after = 'CONFIG_MAPPING_NAMES = OrderedDict([("new_model", "NewConfig"), ("bert", "BertConfig")])\n'
    dirs = {"New": ["new_model"], "BERT": ["bert"], "Whisper": ["whisper"]}
    selected = select_contracts.select([path], dirs, {path: (before, after)})
    assert set(selected) == {"New"}
    assert (
        select_contracts.select([path], {"BERT": ["bert"]}, {path: (before, after)})
        == {}
    )


@pytest.mark.parametrize(
    "after",
    [
        'CONFIG_MAPPING_NAMES = {"bert": "ChangedConfig", "new": "NewConfig"}',
        'CONFIG_MAPPING_NAMES = {"new": "NewConfig"}',
        'CONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig"}\ndef shared(): return 2',
        'import new_dependency\nCONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig"}\ndef shared(): return 1',
        'CONFIG_MAPPING_NAMES = {"bert": "BertConfig", "new": "NewConfig", "new": "OtherConfig"}\ndef shared(): return 1',
        "not valid Python!",
    ],
)
def test_registry_changes_with_runtime_edits_remain_full_fleet(after):
    path = "src/transformers/models/auto/configuration_auto.py"
    before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\ndef shared(): return 1'
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == set(
        dirs
    )


@pytest.mark.parametrize(
    "path,module",
    [
        ("src/transformers/models/__init__.py", "new_model"),
        ("src/transformers/__init__.py", "models.new_model"),
    ],
)
def test_type_checking_model_registration_is_selective(path, module):
    before = "if TYPE_CHECKING:\n    from .models.bert import BertModel\n"
    after = before + f"    from .{module} import NewModel\n"
    dirs = {"New": ["new_model"], "BERT": ["bert"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }


def test_unrelated_shared_lists_and_missing_diff_keep_full_fleet():
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    path = "src/transformers/generation/utils.py"
    assert set(
        select_contracts.select(
            [path],
            dirs,
            {
                path: (
                    'INPUTS = ["input_ids"]',
                    'INPUTS = ["input_ids", "padding_mask"]',
                )
            },
        )
    ) == set(dirs)
    assert set(
        select_contracts.select(["src/transformers/models/auto/modeling_auto.py"], dirs)
    ) == set(dirs)


@pytest.mark.parametrize(
    "value", ["register_and_mutate()", "(x := 1)", "[do_something() for _ in things]"]
)
def test_registry_entry_with_executable_side_effects_keeps_full_fleet(value):
    path = "src/transformers/models/auto/configuration_auto.py"
    before = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}'
    after = f'CONFIG_MAPPING_NAMES = {{"bert": "BertConfig", "new": {value}}}'
    dirs = {"BERT": ["bert"], "Whisper": ["whisper"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == set(
        dirs
    )


def test_registry_model_alias_and_backend_guard_select_dependencies():
    path = "src/transformers/models/auto/tokenization_auto.py"
    before = 'TOKENIZER_MAPPING_NAMES = {"bert": "BertTokenizer"}'
    after = 'TOKENIZER_MAPPING_NAMES = {"bert": "BertTokenizer", "new_alias": "NewTokenizer" if is_tokenizers_available() else None}'
    dirs = {"New": ["shared_tokenizer"], "BERT": ["bert"]}
    assert set(
        select_contracts.select(
            [path],
            dirs,
            {path: (before, after)},
            model_type_to_module=lambda name: "shared_tokenizer",
        )
    ) == {"New"}


def test_changed_sources_uses_merge_base_and_falls_back_when_missing(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    def git(*args):
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()

    git("init")
    git("config", "user.email", "mic-test@example.com")
    git("config", "user.name", "MIC test")
    path = "src/transformers/models/auto/configuration_auto.py"
    file = tmp_path / path
    file.parent.mkdir(parents=True)
    original = 'CONFIG_MAPPING_NAMES = {"bert": "BertConfig"}\n'
    file.write_text(original)
    git("add", ".")
    git("commit", "-m", "base")
    base = git("rev-parse", "HEAD")
    file.write_text(original.replace('"bert":', '"new": "NewConfig", "bert":'))
    git("add", ".")
    git("commit", "-m", "PR registration")
    head = git("rev-parse", "HEAD")
    git("checkout", "--detach", base)
    file.write_text(original + "# base branch advanced\n")
    git("add", ".")
    git("commit", "-m", "base advanced")
    advanced = git("rev-parse", "HEAD")
    git("checkout", "--detach", head)
    assert select_contracts.changed_sources([path], advanced) == {
        path: (original, file.read_text())
    }
    assert select_contracts.changed_sources([path], "missing-ref") == {}


@pytest.mark.parametrize("constructor", ["OrderedDict", "OrderedDict[str, str | None]"])
def test_registry_preserves_existing_duplicates_and_typed_constructor(constructor):
    path = "src/transformers/models/auto/modeling_auto.py"
    before = f'{constructor}([("sam3_tracker", "Sam3Tracker"), ("sam3_tracker", "Sam3Tracker")])'
    before = "MODEL_MAPPING_NAMES = " + before
    after = before.replace(
        '[("sam3_tracker"', '[("new", "NewModel"), ("sam3_tracker"', 1
    )
    dirs = {"New": ["new"], "SAM": ["sam3_tracker"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }
    changed = after.replace(
        '("sam3_tracker", "Sam3Tracker")', '("sam3_tracker", "Changed")', 1
    )
    assert set(select_contracts.select([path], dirs, {path: (before, changed)})) == set(
        dirs
    )


def test_registry_additions_preserve_composed_mapping_splices():
    path = "src/transformers/models/auto/modeling_auto.py"
    before = "MODEL_MAPPING_NAMES = OrderedDict([*list(BASE_MAPPING_NAMES.items())])"
    after = before.replace("])", ', ("new", "NewModel")])')
    dirs = {"New": ["new"], "BERT": ["bert"]}
    assert set(select_contracts.select([path], dirs, {path: (before, after)})) == {
        "New"
    }
    changed = after.replace("BASE_MAPPING_NAMES", "OTHER_MAPPING_NAMES")
    assert set(select_contracts.select([path], dirs, {path: (before, changed)})) == set(
        dirs
    )


def test_bfloat16_on_a_non_native_x86_host_is_reported_as_wrong_hardware(
    tmp_path, monkeypatch
):
    real_read = Path.read_text

    def cpuinfo(self, *a, **k):
        if str(self) == "/proc/cpuinfo":
            return "model name\t: AMD EPYC 7R13 Processor\nflags\t: avx2 avx512f\n"
        return real_read(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", cpuinfo)
    message = run.wrong_hardware({"dtype": "bfloat16"}, "cpu-x86_64")
    assert message.startswith("wrong hardware: AMD EPYC 7R13 Processor")
    assert "cpu-x86_64-bf16" in message
    assert run.wrong_hardware({"dtype": "float32"}, "cpu-x86_64") is None
    assert run.wrong_hardware({"dtype": "bfloat16"}, "cpu-x86_64-bf16") is None
    assert run.wrong_hardware({"dtype": "bfloat16"}, "cuda") is None



def test_check_passes_diff_base_and_run_declared_to_the_runner(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from transformers_mic import backend
    from transformers_mic.cli import main

    calls = []
    monkeypatch.setattr(
        backend,
        "invoke",
        lambda tool, root, options: (
            calls.append(options) or SimpleNamespace(returncode=0)
        ),
    )
    changes = tmp_path / "changes"
    lock = str(tmp_path / "fleet.lock")
    argv = ["check", "--lock", lock, "--changed-files", str(changes)]
    assert main([*argv, "--diff-base", "HEAD^1", "--run-declared"]) == 0
    options = calls[-1]
    assert options[options.index("--diff-base") + 1] == "HEAD^1"
    assert "--run-declared" in options
    assert main(argv) == 0
    assert "--diff-base" not in calls[-1] and "--run-declared" not in calls[-1]
