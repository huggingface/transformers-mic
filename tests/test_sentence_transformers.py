"""Complete ST fixture loading and nested artifact integrity."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("sentence_transformers")
transformers = pytest.importorskip("transformers")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
import build_fixtures
import run

from transformers_mic import commands as _commands
from transformers_mic import sentence_transformers as support
from transformers_mic.workspace import primary_integration

real_execute = _commands.execute


@pytest.mark.parametrize("kind", support.CLASSES)
def test_shrunk_st_pipeline_reloads_and_preserves_predictions(tmp_path, kind):
    from sentence_transformers.base.modules import Dense, Normalize
    from sentence_transformers.sentence_transformer.modules import Pooling
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

    torch.set_num_threads(2)
    original = tmp_path / "backbone"
    original.mkdir()
    config = BertConfig(
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        vocab_size=32,
        max_position_embeddings=128,
    )
    BertModel(config).save_pretrained(original)
    tokenizer = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[CLS]": 2,
                "[SEP]": 3,
                "[MASK]": 4,
                "cat": 5,
                "dog": 6,
                "outside": 7,
            },
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]",
        model_max_length=128,
    ).save_pretrained(original)
    model = getattr(st, kind)(str(original), device="cpu")
    if kind == "SparseEncoder":
        from sentence_transformers.base.modules import Transformer
        from sentence_transformers.sparse_encoder.modules import SpladePooling

        model = st.SparseEncoder(
            modules=[
                Transformer(str(original), transformer_task="fill-mask"),
                SpladePooling(),
            ]
        )
    if kind == "SentenceTransformer":
        model = st.SentenceTransformer(
            modules=[
                model[0],
                Pooling(32),
                Dense(32, 16, activation_function=torch.nn.Identity()),
                Normalize(),
            ]
        )
    source = tmp_path / "source"
    model.save_pretrained(source, create_model_card=False)
    for weights in [*source.rglob("*.safetensors"), *source.rglob("*.bin")]:
        weights.unlink()

    shrunk, backbone, changes = support.initialize(source, kind, 17, build_fixtures)
    fixture = tmp_path / "fixture"
    shrunk.save_pretrained(fixture, create_model_card=False)
    reloaded = getattr(st, kind)(str(fixture), device="cpu", local_files_only=True)
    texts = ["cat outside", "dog cat cat"]
    if kind == "CrossEncoder":
        expected = shrunk.predict([(texts[0], texts[1])], convert_to_tensor=True)
        actual = reloaded.predict([(texts[0], texts[1])], convert_to_tensor=True)
        torch.testing.assert_close(actual, expected)
    elif kind == "MultiVectorEncoder":
        for method in ("encode_query", "encode_document"):
            expected = getattr(shrunk, method)(texts)
            actual = getattr(reloaded, method)(texts)
            for left, right in zip(actual, expected):
                torch.testing.assert_close(left, right)
    else:
        expected = shrunk.encode(texts, convert_to_tensor=True).to_dense()
        actual = reloaded.encode(texts, convert_to_tensor=True).to_dense()
        torch.testing.assert_close(actual, expected)
    if kind == "SentenceTransformer":
        assert reloaded[1].get_embedding_dimension() == backbone.config.hidden_size
        assert reloaded[2].in_features == reloaded[1].get_embedding_dimension()
        assert changes


def st_modules(folder, *paths):
    entries = [{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.base.modules.Transformer"}]
    entries += [
        {"idx": i, "name": str(i), "path": path, "type": "sentence_transformers.sentence_transformer.modules.Pooling"}
        for i, path in enumerate(paths, 1)
    ]
    (folder / "modules.json").write_text(json.dumps(entries))


def test_nested_fixture_changes_invalidate_integrity(tmp_path):
    module = tmp_path / "1_Pooling"
    module.mkdir()
    config = module / "config.json"
    config.write_text('{"pooling_mode": "mean"}')
    st_modules(tmp_path, "1_Pooling")
    initial = run.file_hashes(tmp_path)
    assert "1_Pooling/config.json" in initial
    config.write_text('{"pooling_mode": "cls"}')
    assert run.file_hashes(tmp_path) != initial


def test_capture_supports_direct_forward_and_regular_calls():
    from transformers import BertConfig, BertForSequenceClassification

    existing = BertForSequenceClassification(
        BertConfig(
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            vocab_size=16,
        )
    )
    forward = type(existing).forward
    with build_fixtures.captured_calls(type(existing), method="forward") as calls:
        existing(torch.tensor([[1, 2]]))
        created = BertForSequenceClassification(existing.config)
        with torch.inference_mode():
            created.forward(torch.tensor([[3, 4]]))
    created.forward(torch.tensor([[5, 6]]))
    assert type(existing).forward is forward
    assert "forward" not in created.__dict__
    assert len(calls) == 2
    assert [call["class"] for call in calls] == [
        "BertForSequenceClassification",
        "BertForSequenceClassification",
    ]
    assert calls[0]["kwargs"]["input_ids"].tolist() == [[1, 2]]
    assert calls[1]["kwargs"]["input_ids"].tolist() == [[3, 4]]


@pytest.mark.parametrize(
    ("flags", "expected_stages"),
    [([], []), (["--build"], ["build"]), (["--record"], ["build", "record"])],
)
def test_st_creation_preserves_explicit_namespace_and_inference_outputs(
    tmp_path, monkeypatch, flags, expected_stages
):
    from types import SimpleNamespace

    from transformers_mic import commands
    from transformers_mic.cli import parser

    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "bert", "architectures": ["BertModel"]})
    )
    (source / "modules.json").write_text(
        json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                }
            ]
        )
    )
    monkeypatch.setattr(support, "source_files", lambda *args: source)
    monkeypatch.setattr(
        support,
        "HfApi",
        lambda: SimpleNamespace(
            model_info=lambda *args, **kwargs: SimpleNamespace(sha="a" * 40)
        ),
    )
    monkeypatch.setattr(
        support, "environment", lambda: {"sentence-transformers": "6.2.0"}
    )
    stages = []
    monkeypatch.setattr(commands, "build", lambda args: stages.append("build") or 0)
    monkeypatch.setattr(
        commands, "execute", lambda args, record: stages.append("record") or 0
    )
    args = parser().parse_args(
        [
            "create",
            "org/model",
            "--out",
            str(tmp_path / "work"),
            "--sentence-transformers",
            "--namespace",
            "tomaarsen",
            "--fixtures-only",
            *flags,
        ]
    )
    assert commands.create(args) == 0
    assert stages == expected_stages
    manifest = json.loads(
        (tmp_path / "work/example/hub/st-model-mic/integration.json").read_text()
    )
    assert set(manifest["fixtures"]) == {"tomaarsen/st-model-mic"}
    # Only the ST integration: Transformers lanes skip it, the ST nightly selects it.
    assert set(manifest["integrations"]) == {"sentence_transformers"}
    integration = manifest["integrations"]["sentence_transformers"]
    assert integration["entry_point"] == "integrations/sentence_transformers/usage.md"
    assert not integration["training"]
    assert {"scores", "query_embeddings", "document_embeddings"} <= set(
        integration["outputs"]
    )
    root = tmp_path / "work/example/hub/st-model-mic"
    assert (root / integration["entry_point"]).exists()
    assert not (root / "integrations/transformers").exists()
    lock = json.loads(
        (tmp_path / "work/example/transformers/tests/contracts/fleet.lock").read_text()
    )
    assert lock["contracts"]["st-model"]["integrations"] == ["sentence_transformers"]
    if not flags:
        # record/check/review default to the contract's only integration.
        invoked = []
        monkeypatch.setattr(
            commands.backend,
            "invoke",
            lambda command, cwd, options: invoked.append(options)
            or SimpleNamespace(returncode=1),
        )
        check = parser().parse_args(["--workspace", str(tmp_path / "work"), "check"])
        real_execute(check)
        assert invoked[0][invoked[0].index("--framework") + 1] == "sentence_transformers"
        review = parser().parse_args(["--workspace", str(tmp_path / "work"), "review"])
        commands.review(review)
        expectations = invoked[1][invoked[1].index("--expectations") + 1]
        assert expectations.endswith("integrations/sentence_transformers/expectations")


def test_resume_preserves_st_class_and_namespace():
    from transformers_mic.automation import resume_options
    from transformers_mic.cli import parser

    args = parser().parse_args(
        [
            "run",
            "org/model",
            "--sentence-transformers",
            "SparseEncoder",
            "--namespace",
            "tomaarsen",
            "--local-only",
        ]
    )
    resumed = parser().parse_args(["run", "org/model", *resume_options(args)])
    assert resumed.sentence_transformers == "SparseEncoder"
    assert resumed.namespace == "tomaarsen"
    assert resumed.local_only


def test_published_contract_and_baselines_are_separate_from_fixture_hashes(tmp_path):
    module = tmp_path / "1_Pooling"
    module.mkdir()
    (module / "config.json").write_text("{}")
    st_modules(tmp_path, "1_Pooling")
    initial = run.file_hashes(tmp_path)
    usage = tmp_path / "integrations/sentence_transformers/usage.md"
    usage.parent.mkdir(parents=True)
    usage.write_text("contract")
    (tmp_path / "integration.json").write_text("{}")
    (tmp_path / ".mic-repo.json").write_text(
        json.dumps(
            {
                "contract_files": [
                    "integration.json",
                    "integrations/sentence_transformers/usage.md",
                ]
            }
        )
    )
    baseline = tmp_path / "integrations/sentence_transformers/expectations/cpu-x86_64/model.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_text("{}")
    assert run.file_hashes(tmp_path) == initial
    baseline.write_text('{"scores": [1.0]}')
    assert run.file_hashes(tmp_path) == initial


def test_primary_integration_prefers_transformers_then_the_only_one():
    both = {"integrations": {"sentence_transformers": {}, "transformers": {}}}
    assert primary_integration(both) == "transformers"
    only = {"integrations": {"sentence_transformers": {}}}
    assert primary_integration(only) == "sentence_transformers"
    with pytest.raises(ValueError, match="--integration"):
        primary_integration({"integrations": {"a": {}, "b": {}}})

