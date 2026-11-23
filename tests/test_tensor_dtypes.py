"""Fixtures keep the per-tensor float dtypes their checkpoint stores."""

import json
import sys
from pathlib import Path

import pytest

from transformers_mic import tensor_dtypes

GLM = {
    "model.language_model.layers.3.mlp.gate.weight": "BF16",
    "model.language_model.layers.45.mlp.gate.weight": "BF16",
    "model.language_model.layers.3.mlp.gate.e_score_correction_bias": "F32",
    "model.language_model.layers.45.mlp.gate.e_score_correction_bias": "F32",
    "model.language_model.layers.3.mlp.experts.0.gate_proj.weight": "F8_E4M3",
    "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv": "F32",
    "model.language_model.layers.3.self_attn.o_proj.weight": "BF16",
    "model.language_model.layers.45.self_attn.o_proj.weight": "F8_E4M3",
    "model.language_model.embed_tokens.weight": "BF16",
}


def test_pattern_wildcards_numeric_path_components_only():
    assert tensor_dtypes.pattern("model.layers.45.mlp.experts.12.down_proj.weight") == (
        "model.layers.*.mlp.experts.*.down_proj.weight"
    )
    assert tensor_dtypes.pattern("h.0.attn.c_attn.weight") == "h.*.attn.c_attn.weight"
    assert tensor_dtypes.pattern("0.auto_model.layer3.weight") == "*.auto_model.layer3.weight"
    assert tensor_dtypes.pattern("encoder.conv1.weight") == "encoder.conv1.weight"


def test_a_pattern_takes_the_stored_dtype_only_when_unanimous_and_float():
    patterns = tensor_dtypes.pattern_dtypes(GLM)
    assert patterns["model.language_model.layers.*.mlp.gate.weight"] == "bfloat16"
    assert patterns["model.language_model.layers.*.mlp.gate.e_score_correction_bias"] == "float32"
    # BF16 in some layers, F8 in others: no single answer, the default applies.
    assert "model.language_model.layers.*.self_attn.o_proj.weight" not in patterns
    # Quantized storage is not mirrored by a dense fixture.
    assert "model.language_model.layers.*.mlp.experts.*.gate_proj.weight" not in patterns


def test_targets_map_a_shrunk_fixture_and_default_the_rest():
    names = [
        "model.language_model.layers.0.mlp.gate.weight",
        "model.language_model.layers.1.mlp.gate.e_score_correction_bias",
        "model.language_model.layers.1.mlp.experts.0.gate_proj.weight",
        "model.language_model.layers.1.self_attn.o_proj.weight",
        "model.language_model.layers.1.mlp.experts.gate_up_proj",  # fused by a conversion: not in the source
    ]
    assert tensor_dtypes.target_dtypes(names, GLM, "bfloat16") == {
        names[0]: "bfloat16",
        names[1]: "float32",
        names[2]: "bfloat16",
        names[3]: "bfloat16",
        names[4]: "bfloat16",
    }
    # Tensors in the checkpoint's main dtype (BF16 here) follow the default, whatever it is.
    assert tensor_dtypes.target_dtypes(names[:2], GLM, "float32") == {names[0]: "float32", names[1]: "float32"}


def test_the_main_dtype_is_weighted_by_parameters():
    stored = {
        "embed.weight": {"dtype": "BF16", "numel": 1000},
        **{f"layers.{i}.norm.weight": {"dtype": "F32", "numel": 8} for i in range(5)},
    }
    assert tensor_dtypes.main_dtype(stored) == "bfloat16"
    assert tensor_dtypes.overrides(stored, "bfloat16") == {"layers.*.norm.weight": "float32"}
    # Counted by tensors when sizes are unknown.
    assert tensor_dtypes.main_dtype({k: v["dtype"] for k, v in stored.items()}) == "float32"


def test_overrides_list_only_patterns_that_differ_from_the_default():
    assert tensor_dtypes.overrides(GLM, "bfloat16") == {
        "model.language_model.layers.*.mlp.experts.*.gate_proj.weight_scale_inv": "float32",
        "model.language_model.layers.*.mlp.gate.e_score_correction_bias": "float32",
    }
    assert tensor_dtypes.overrides({"a.0.w": "BF16", "a.1.w": "BF16", "b.w": "F32"}, "bfloat16") == {"b.w": "float32"}
    assert tensor_dtypes.overrides({}, "float32") == {}


def test_a_uniformly_stored_checkpoint_keeps_the_default():
    # SmolLM-135M stores every tensor F32 under a bfloat16 config: its fixture stays bfloat16.
    smollm = {"model.layers.0.mlp.up_proj.weight": "F32", "model.norm.weight": "F32"}
    assert not tensor_dtypes.mixed(smollm)
    assert tensor_dtypes.overrides(smollm, "bfloat16") == {}
    assert set(tensor_dtypes.target_dtypes(smollm, smollm, "bfloat16").values()) == {"bfloat16"}
    # F8 weights with F32 scales: one float dtype, so nothing to mirror either.
    assert not tensor_dtypes.mixed({"a.weight": "F8_E4M3", "a.weight_scale_inv": "F32"})
    assert tensor_dtypes.overrides({"a.weight": "F8_E4M3", "a.weight_scale_inv": "F32"}, "bfloat16") == {}
    assert tensor_dtypes.mixed(GLM)


def test_config_dtype_reads_sub_configs_when_the_top_level_has_none():
    assert tensor_dtypes.config_dtype({"dtype": "float16", "text_config": {"dtype": "bfloat16"}}) == "float16"
    assert tensor_dtypes.config_dtype({"torch_dtype": "bfloat16"}) == "bfloat16"
    assert (
        tensor_dtypes.config_dtype(
            {"text_config": {"model_type": "glm", "dtype": "bfloat16"}, "vision_config": {"dtype": "float32"}}
        )
        == "bfloat16"
    )
    assert tensor_dtypes.config_dtype(
        {"vision_config": {"dtype": "float16"}, "audio_config": {"dtype": "float16"}}
    ) == ("float16")
    assert (
        tensor_dtypes.config_dtype({"vision_config": {"dtype": "float16"}, "audio_config": {"dtype": "float32"}})
        is None
    )
    assert tensor_dtypes.config_dtype({"model_type": "bert"}) is None


def _write_safetensors(path, tensors):
    header = json.dumps(
        {name: {"dtype": dtype, "shape": [1], "data_offsets": [0, 0]} for name, dtype in tensors.items()}
    )
    path.write_bytes(len(header).to_bytes(8, "little") + header.encode())


def test_checkpoint_dtypes_read_local_headers_and_cache_per_revision(tmp_path, monkeypatch):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a.weight": "s1.safetensors", "b.weight": "s2.safetensors"}})
    )
    _write_safetensors(tmp_path / "s1.safetensors", {"a.weight": "BF16"})
    _write_safetensors(tmp_path / "s2.safetensors", {"b.weight": "F32"})
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(tensor_dtypes, "cache_path", lambda repo, revision: cache)
    revision = "0" * 40
    expected = {"a.weight": {"dtype": "BF16", "numel": 1}, "b.weight": {"dtype": "F32", "numel": 1}}
    assert tensor_dtypes.checkpoint_dtypes("org/model", revision, tmp_path) == expected
    (tmp_path / "s1.safetensors").unlink()  # the cache answers without the files
    assert tensor_dtypes.checkpoint_dtypes("org/model", revision, tmp_path) == expected


@pytest.mark.requires_runtime
def test_build_saves_per_tensor_dtypes_and_keeps_uniform_keys(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
    import build_fixtures
    from safetensors import safe_open
    from transformers import BertConfig, BertModel

    source = tmp_path / "source"
    source.mkdir()
    config = BertConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2, num_attention_heads=2, vocab_size=64)
    config.save_pretrained(source)
    data = json.loads((source / "config.json").read_text())
    data.pop("dtype", None)
    data.pop("torch_dtype", None)
    data["text_config"] = {"dtype": "bfloat16"}  # nested only, as GLM-5.3-Flash
    (source / "config.json").write_text(json.dumps(data))
    assert build_fixtures.source_dtype(source) == "bfloat16"

    torch.manual_seed(0)
    model = BertModel(config).eval()
    names = list(model.state_dict())
    uniform = {name: "BF16" for name in names}
    mixed = {**uniform, **{n: "F32" for n in names if n.endswith("attention.output.dense.weight")}}
    mixed["encoder.layer.1.intermediate.dense.weight"] = "F8_E4M3"  # disagrees with layer 0: default

    build_fixtures.save(
        model,
        tmp_path / "out",
        source,
        [],
        checkpoint={"keys": set(names), "binary": False, "dtype": "bfloat16", "dtypes": mixed},
    )
    with safe_open(tmp_path / "out" / "model.safetensors", "pt") as f:
        saved = {name: f.get_tensor(name) for name in f.keys()}
    dense = "encoder.layer.0.attention.output.dense.weight"
    assert saved[dense].dtype == torch.float32
    assert torch.equal(saved[dense], model.state_dict()[dense])  # not rounded through bfloat16 first
    assert saved["encoder.layer.0.intermediate.dense.weight"].dtype == torch.bfloat16
    assert saved["embeddings.word_embeddings.weight"].dtype == torch.bfloat16
    # The model was saved from float32; dtype="auto" must still load the fixture in its own dtype.
    assert json.loads((tmp_path / "out" / "config.json").read_text())["dtype"] == "bfloat16"
    # Integer tensors are never cast, whatever the header says.
    ints = build_fixtures.stored_like({"a.ids": torch.arange(3)}, {"a.ids": "F32"}, "bfloat16")
    assert ints["a.ids"].dtype == torch.int64

    plain = build_fixtures.fixture_key(source, BertModel, [])[0]
    assert build_fixtures.fixture_key(source, BertModel, [], tensor_dtypes=uniform)[0] == plain
    assert build_fixtures.fixture_key(source, BertModel, [], tensor_dtypes=mixed)[0] != plain


def test_config_with_dtype_sets_the_top_level_and_sub_configs_and_keeps_bytes_when_right(tmp_path):
    pytest.importorskip("torch")  # build_fixtures imports torch at module level
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
    import build_fixtures

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "dtype": "float32",
                "text_config": {"dtype": "float32", "hidden_size": 8},
                "vision_config": {"hidden_size": 4},
                "thinker_config": {"dtype": None, "audio_config": {"dtype": None}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    build_fixtures.config_with_dtype(path, "bfloat16")
    config = json.loads(path.read_text())
    assert config["dtype"] == config["text_config"]["dtype"] == "bfloat16"
    assert "dtype" not in config["vision_config"]  # a sub-config that named none stays as saved
    # A null dtype inherits the loading dtype (Qwen2.5-Omni's thinker): pinning it would break float32 loads.
    assert config["thinker_config"] == {"dtype": None, "audio_config": {"dtype": None}}

    right = path.read_bytes()
    stamp = path.stat().st_mtime_ns
    build_fixtures.config_with_dtype(path, "bfloat16")
    assert path.read_bytes() == right and path.stat().st_mtime_ns == stamp  # nothing to change, not rewritten
