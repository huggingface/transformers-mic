"""Regression coverage for non-language-model fixture inputs."""

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
safetensors = pytest.importorskip("safetensors.torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "example"))
import build_fixtures  # noqa: E402


def test_weight_named_processor_asset_is_copied(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "attention_mask.safetensors").write_bytes(b"processor tensor")
    (tmp_path / "model.safetensors").write_bytes(b"model weights")
    recipe = {"copy_assets": ["attention_mask.safetensors"]}

    assert "attention_mask.safetensors" in build_fixtures.copied_files(tmp_path, recipe)
    assert "model.safetensors" not in build_fixtures.copied_files(tmp_path, recipe)


def test_checkpoint_formats_and_oversized_files_are_not_copied(tmp_path, monkeypatch):
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "gemma-2b.gguf").write_bytes(b"weights")
    assert build_fixtures.copied_files(tmp_path) == ["tokenizer.json"]
    monkeypatch.setattr(build_fixtures, "MAX_COPIED_BYTES", 1)
    with pytest.raises(ValueError, match="tokenizer.json is too large"):
        build_fixtures.copied_files(tmp_path)

def test_batch_norm_calibration_recovers_evaluation_sensitivity_without_changing_weights():
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4, bias=False),
        torch.nn.BatchNorm1d(4, eps=1e-8),
        torch.nn.Dropout(0.9),
    ).eval()
    with torch.no_grad():
        model[0].weight.copy_(torch.eye(4) * 0.001)
    batch = {"input": torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])}
    weights = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    before = model(**batch)
    assert build_fixtures.calibrate_batch_norm(model, batch) == 1
    after = model(**batch)
    assert torch.max(torch.abs(after[0] - after[1])) > 100 * torch.max(
        torch.abs(before[0] - before[1])
    )
    assert torch.isfinite(after).all()
    assert all(
        torch.equal(weights[name], parameter)
        for name, parameter in model.named_parameters()
    )
    assert not model[1].training and not model[2].training
    assert model[1].momentum == 0.1
    assert torch.equal(model(**batch), after)


def test_calibration_does_not_forward_models_without_batch_norm():
    class Model(torch.nn.Module):
        def forward(self, **kw):
            raise AssertionError("No calibration forward is needed")

    assert build_fixtures.calibrate_batch_norm(Model(), {}) == 0


def test_selected_fixed_tensors_survive_random_initialization(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.geometry = torch.nn.Parameter(torch.zeros(2), requires_grad=False)
            self.weights = torch.nn.Parameter(torch.zeros(2))
            self.register_buffer("mesh_index", torch.tensor(0))
            self.register_buffer("scratch", torch.tensor(0), persistent=False)

    source = tmp_path / "source"
    source.mkdir()
    safetensors.save_file(
        {
            "geometry": torch.tensor([2.0, 3.0]),
            "weights": torch.tensor([8.0, 9.0]),
            "mesh_index": torch.tensor(7),
        },
        source / "model.safetensors",
    )
    monkeypatch.setattr(
        "huggingface_hub.hf_hub_download",
        lambda repo, filename, **kwargs: source / filename,
    )
    model = Model()
    hashes = build_fixtures.restore_frozen(
        model,
        source,
        "org/checkpoint",
        "pinned-sha",
        {"preserve_frozen_prefixes": ["geometry", "mesh_"]},
    )
    assert model.geometry.tolist() == [2.0, 3.0]
    assert model.weights.tolist() == [0.0, 0.0]
    assert model.mesh_index.item() == 7
    assert (
        hashes["mesh_index"]
        == hashlib.sha256(
            torch.tensor([7]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    assert (
        hashes["geometry"]
        == hashlib.sha256(
            torch.tensor([2.0, 3.0]).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="trainable"):
        build_fixtures.restore_frozen(
            model,
            source,
            "org/checkpoint",
            "pinned-sha",
            {"preserve_frozen_prefixes": ["weights"]},
        )
    with pytest.raises(ValueError, match="No frozen parameters or persistent buffers"):
        build_fixtures.restore_frozen(
            model,
            source,
            "org/checkpoint",
            "pinned-sha",
            {"preserve_frozen_prefixes": ["scratch"]},
        )


def test_shrink_resamples_longrope_factors_to_the_resized_head():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Phi3Config

    half = 3072 // 24 * 3 // 4 // 2
    config = Phi3Config(
        hidden_size=3072,
        num_attention_heads=24,
        partial_rotary_factor=0.75,
        max_position_embeddings=131072,
        original_max_position_embeddings=4096,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0 + i for i in range(half)],
            "long_factor": [2.0 + i for i in range(half)],
        },
    )
    rope = shrink(config).rope_parameters
    assert len(rope["short_factor"]) == len(rope["long_factor"]) < half
    assert rope["short_factor"][0] == 1.0 and rope["long_factor"][0] == 2.0


def test_shrink_keeps_stage_backbone_outputs_and_matches_encoder_channels():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import RTDetrConfig

    config = shrink(RTDetrConfig())
    backbone = config.backbone_config
    assert backbone.out_indices == [2, 3, 4]
    assert config.encoder_in_channels == [
        backbone.hidden_sizes[i - 1] for i in backbone.out_indices
    ]


def test_capture_names_positional_model_inputs():
    from transformers import BertConfig, BertModel

    model = BertModel(
        BertConfig(
            hidden_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=8,
        )
    )
    ids = torch.tensor([[1, 2, 3]])
    with build_fixtures.captured_calls() as calls:
        model(ids, attention_mask=torch.ones_like(ids))
    assert set(calls[0]["kwargs"]) == {"input_ids", "attention_mask"}
    assert torch.equal(calls[0]["kwargs"]["input_ids"], ids)


def test_overfit_fits_distinct_targets_when_the_model_has_no_loss():
    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 3)

        def forward(self, pixel_values):
            return {"pred_logits": self.proj(pixel_values)}

    torch.manual_seed(0)
    model = Head()
    batch = {"pixel_values": torch.randn(2, 4)}
    result = build_fixtures.overfit(model, batch, max_steps=400)
    assert result["converged"]
    first, second = model(**batch)["pred_logits"].detach()
    assert not torch.allclose(first, second, atol=0.1)


def test_shrink_drops_the_checkpoint_quantization_config():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import LlamaConfig

    config = LlamaConfig(
        quantization_config={"quant_method": "fp8", "weight_block_size": [128, 128]}
    )
    assert getattr(shrink(config), "quantization_config", None) is None


def test_shrink_resamples_longrope_factors_to_the_mla_rotary_width():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import AutoConfig

    config = AutoConfig.for_model(
        "minicpm3",
        qk_rope_head_dim=32,
        rope_scaling={
            "type": "longrope",
            "short_factor": [1.0 + i for i in range(16)],
            "long_factor": [2.0 + i for i in range(16)],
            "original_max_position_embeddings": 4096,
        },
    )
    shrunk = shrink(config)
    rope = shrunk.rope_parameters
    assert (
        len(rope["short_factor"])
        == len(rope["long_factor"])
        == shrunk.qk_rope_head_dim // 2
    )


def test_sensitivity_perturbs_list_contexts_and_never_the_forecast_target():
    import perturb

    class Forecaster(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(2.0))

        def forward(self, past_values, future_values=None):
            return {
                "mean_predictions": torch.stack(
                    [self.scale * row[-3:] for row in past_values]
                )
            }

    batch = {
        "past_values": [torch.arange(1.0, 9.0), torch.arange(10.0, 15.0)],
        "future_values": torch.zeros(2, 3),
    }
    result = perturb.sensitivity(Forecaster(), batch, threshold=1e-3)
    assert result["pass"]
    assert set(result["deltas"]) == {
        "past_values:roll_batch",
        "past_values:reverse",
        "past_values:halve",
    }
    assert result["no_perturbation_defined"] == []


def test_sensitivity_skips_inputs_only_absorbed_by_kwargs():
    import perturb

    class MaskedLM(torch.nn.Module):  # DistilBERT: the tokenizer adds token_type_ids
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return {"logits": self.embed(input_ids) * attention_mask[..., None]}

    batch = {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "token_type_ids": torch.zeros(2, 3, dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
    }
    result = perturb.sensitivity(MaskedLM(), batch, threshold=1e-3)
    assert result["pass"]
    assert result["not_a_forward_parameter"] == ["token_type_ids"]
    assert not any(key.startswith("token_type_ids") for key in result["deltas"])


def test_sensitivity_runs_a_decoder_only_model_without_its_labels():
    import perturb

    class CausalLM(torch.nn.Module):  # CSM: labels only valid for the unperturbed audio
        config = SimpleNamespace(is_encoder_decoder=False)

        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4)

        def forward(self, input_ids, labels=None):
            if labels is not None:
                raise IndexError("Target 128002 is out of bounds.")
            return {"logits": self.embed(input_ids)}

    ids = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert perturb.sensitivity(CausalLM(), {"input_ids": ids, "labels": ids}, threshold=1e-3)["pass"]


@pytest.mark.parametrize(
    "model_type,heads_key",
    [("pvt", "num_attention_heads"), ("swin", "num_heads")],
)
def test_shrink_keeps_per_stage_heads_dividing_per_stage_widths(model_type, heads_key):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import AutoConfig

    original = {"pvt": [1, 2, 5, 8], "swin": [3, 6, 12, 24]}[model_type]
    config = shrink(AutoConfig.for_model(model_type, **{heads_key: original}))
    heads = getattr(config, heads_key)
    widths = getattr(config, "hidden_sizes", None) or [
        config.embed_dim * 2**i for i in range(len(config.depths))
    ]
    assert all(w % h == 0 for w, h in zip(widths, heads))
    assert all(1 <= h <= o for h, o in zip(heads, original))


def test_max_delta_ignores_matching_infinities_but_keeps_nan():
    import perturb

    inf = float("inf")
    base = {"logits": torch.tensor([[1.0, -inf], [2.0, -inf]])}
    moved = {"logits": torch.tensor([[1.5, -inf], [2.0, -inf]])}
    assert perturb.max_delta(moved, base) == 0.5
    assert perturb.max_delta(base, base) == 0.0
    unmasked = {"logits": torch.tensor([[1.0, 0.0], [2.0, -inf]])}
    assert perturb.max_delta(unmasked, base) == inf
    broken = {"logits": torch.tensor([[float("nan"), -inf], [2.0, -inf]])}
    assert perturb.max_delta(broken, base) != perturb.max_delta(broken, base)


@pytest.mark.parametrize("fill,healthy", [(float("-inf"), True), (float("nan"), False)])
def test_numeric_health_accepts_masked_logits_but_not_nan(tmp_path, fill, healthy):
    class Output(dict):
        __getattr__ = dict.__getitem__

    class Masked(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 3)

        @classmethod
        def from_pretrained(cls, folder, dtype):
            return cls().to(dtype)

        def forward(self, x):
            logits = self.proj(x)
            loss = logits[:, :2].float().pow(2).mean()
            logits = torch.cat(
                [logits[:, :2], torch.full_like(logits[:, 2:], fill)], -1
            )
            return Output(logits=logits, loss=loss)

    rows = build_fixtures.numeric_health(Masked, tmp_path, {"x": torch.ones(2, 2)})
    assert all(row["finite"] is healthy for row in rows.values()), rows


def test_numeric_health_rejects_infinities_float32_does_not_have(tmp_path):
    class Output(dict):
        __getattr__ = dict.__getitem__

    class Overflow(torch.nn.Module):
        def __init__(self, dtype):
            super().__init__()
            self.dtype_ = dtype
            self.proj = torch.nn.Linear(2, 3)

        @classmethod
        def from_pretrained(cls, folder, dtype):
            return cls(dtype)

        def forward(self, x):
            logits = self.proj(x)
            if self.dtype_ == torch.float16:
                logits = logits + float("inf")
            return Output(logits=logits, loss=logits.float().clamp(max=1).pow(2).mean())

    rows = build_fixtures.numeric_health(Overflow, tmp_path, {"x": torch.ones(2, 2)})
    assert rows["float32"]["finite"] and rows["bfloat16"]["finite"]
    assert not rows["float16"]["finite"]


def test_shrink_keeps_vision_projections_matching_the_text_width():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import CLIPConfig, JanusConfig

    # As in Janus-Pro: both aligners project to the language model's width.
    config = shrink(JanusConfig(text_config={"hidden_size": 2048}))
    width = config.text_config.hidden_size
    assert config.vision_config.projection_dim == width
    assert config.vq_config.projection_dim == width
    # CLIP's projection is its own joint space, not the text width.
    clip = shrink(CLIPConfig())
    assert clip.projection_dim != clip.text_config.hidden_size


def test_shrink_keeps_blt_global_width_tied_to_the_local_models():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import BltConfig

    # As in itazap/blt-1b-hf: global = encoder width x cross_attn_k, named by both local models.
    local = {"hidden_size": 1024, "hidden_size_global": 2048}
    config = shrink(
        BltConfig(
            cross_attn_k=2,
            encoder_config=local,
            decoder_config=local,
            global_config={
                "hidden_size": 2048,
                "head_dim": 128,
                "num_attention_heads": 16,
            },
        )
    )
    width = config.global_config.hidden_size
    assert width == config.encoder_config.hidden_size * config.cross_attn_k
    assert config.encoder_config.hidden_size_global == width
    assert config.decoder_config.hidden_size_global == width
    assert config.global_config.encoder_cross_output_size is None
    assert (
        config.global_config.head_dim * config.global_config.num_attention_heads
        == width
    )


def test_shrink_keeps_every_block_kind_of_an_index_derived_pattern():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import RecurrentGemmaConfig

    # As in google/recurrentgemma-2b: 26 layers cycling (recurrent, recurrent, attention).
    config = shrink(RecurrentGemmaConfig(num_hidden_layers=26))
    assert config.layers_block_type == ["recurrent", "recurrent", "attention"]


def test_shrink_keeps_attention_layers_a_period_places():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import JambaConfig

    # layer_types is computed from attn_layer_period/offset, not stored, so it can't be re-indexed.
    config = shrink(JambaConfig(num_hidden_layers=32, attn_layer_period=8, attn_layer_offset=4))
    assert set(config.layer_types) == {"linear_attention", "full_attention"}


def test_config_without_token_types_accepts_their_perturbation():
    from transformers import BertConfig, DebertaV2Config

    batch = {"input_ids": torch.ones(1, 3, dtype=torch.long), "token_type_ids": torch.zeros(1, 3, dtype=torch.long)}
    assert set(build_fixtures.config_insensitive(DebertaV2Config(type_vocab_size=0), batch)) == {"token_type_ids:all_ones"}
    assert build_fixtures.config_insensitive(BertConfig(type_vocab_size=2), batch) == {}
    assert build_fixtures.config_insensitive(DebertaV2Config(type_vocab_size=0), {"input_ids": batch["input_ids"]}) == {}

def test_null_override_of_a_field_shrink_already_dropped_is_a_no_op(tmp_path):
    from transformers import BertConfig, BertModel

    BertConfig(num_hidden_layers=4).save_pretrained(tmp_path)
    fields = {"hidden_size": 64, "quantization_config": None}
    config = build_fixtures.shrunk_config(tmp_path, BertModel, {"config_overrides": {"fields": fields}})
    assert config.hidden_size == 64 and getattr(config, "quantization_config", None) is None
    fields["not_a_field"] = 1
    with pytest.raises(SystemExit, match="not_a_field"):
        build_fixtures.shrunk_config(tmp_path, BertModel, {"config_overrides": {"fields": fields}})

def test_shrink_keeps_siglip_text_projection_at_the_shared_width():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import SiglipConfig

    config = shrink(SiglipConfig())  # text projection_size defaults to its hidden_size (768)
    assert config.text_config.projection_size == config.text_config.hidden_size == config.vision_config.hidden_size

def test_shrink_keeps_an_audio_output_dim_at_the_text_width():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Qwen3ASRConfig

    # As in Qwen/Qwen3-ASR-0.6B-hf: the audio tower projects to the text hidden size.
    config = Qwen3ASRConfig(audio_config={"output_dim": 1024}, text_config={"hidden_size": 1024})
    config = shrink(config)
    assert config.audio_config.output_dim == config.text_config.hidden_size


def test_a_rebuild_reuses_the_recorded_config_even_when_the_rule_changes(tmp_path, monkeypatch):
    import json

    from transformers import BertConfig, BertModel

    BertConfig(num_hidden_layers=4).save_pretrained(tmp_path / "source")
    source = tmp_path / "source"
    recipe = {"seed": 17, "model_class": "BertModel"}
    key, config = build_fixtures.fixture_key(source, BertModel, [], recipe)
    out = tmp_path / "fixture"
    out.mkdir()
    fixture = {"build": recipe, "built_from": "org/model"}
    record = {"recipe": recipe, "built_from": {"repo": "org/model"}, "key_fields": config}
    (out / "build.json").write_text(json.dumps(record))
    assert build_fixtures.recorded_config(out, fixture) == json.loads(json.dumps(config))

    # Today's rule makes something else; the recorded config still reproduces the key.
    monkeypatch.setattr(build_fixtures, "shrink", lambda c: BertConfig(hidden_size=32, num_attention_heads=2))
    assert build_fixtures.fixture_key(source, BertModel, [], recipe)[0] != key
    recorded = {**recipe, "recorded_config": build_fixtures.recorded_config(out, fixture)}
    assert build_fixtures.fixture_key(source, BertModel, [], recorded)[0] == key

    # An edited recipe, or a record from before provenance, re-shrinks.
    assert build_fixtures.recorded_config(out, {**fixture, "build": {**recipe, "seed": 3}}) is None
    (out / "build.json").write_text(json.dumps({**record, "recipe": None}))
    assert build_fixtures.recorded_config(out, fixture) is None


def test_shrink_keeps_channels_feeding_a_sibling_encoder():
    # Unlimited-OCR's SAM tower ends at the CLIP encoder's width: its output is
    # the encoder's patch embedding.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import UnlimitedOcrConfig

    vision = shrink(UnlimitedOcrConfig()).vision_config
    assert vision.sam_config.downsample_channels[-1] == vision.encoder_config.hidden_size
    assert list(vision.sam_config.downsample_channels) == [32, 64]


def test_shrink_raises_a_vanishing_initializer_range():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import UnlimitedOcrConfig

    sam = shrink(UnlimitedOcrConfig()).vision_config.sam_config
    assert sam.initializer_range == 0.02


def test_shrink_keeps_projection_dim_equal_to_the_text_width():
    # OWLv2's class head contracts text query embeds (projection_dim) with a
    # projection of the image embeds to the text width.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Owlv2Config

    config = shrink(Owlv2Config())
    assert config.projection_dim == config.text_config.hidden_size


def test_shrink_keeps_one_dpt_layer_per_out_index():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import DPTConfig

    config = shrink(DPTConfig(backbone_out_indices=[5, 11, 17, 23], num_hidden_layers=24))
    assert config.num_hidden_layers == 4
    assert config.backbone_out_indices == [0, 1, 2, 3]
    assert len(config.neck_hidden_sizes) == 4


def test_save_accepts_a_missing_tied_alias(tmp_path):
    from transformers import BertConfig, BertForMaskedLM

    model = BertForMaskedLM(BertConfig(vocab_size=50, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32))
    model.config.save_pretrained(tmp_path / "src")
    build_fixtures.save(model, tmp_path / "out", tmp_path / "src", ["cls.predictions.decoder.bias"])


def test_shrink_keeps_clipseg_extract_layers_within_the_vision_depth():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import CLIPSegConfig

    config = shrink(CLIPSegConfig(extract_layers=[3, 6, 9]))
    assert config.extract_layers == [0, 1, 1]
    assert max(config.extract_layers) < config.vision_config.num_hidden_layers


def test_shrink_keeps_one_moe_layer_after_leading_dense_layers():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Lfm2MoeConfig

    config = shrink(Lfm2MoeConfig(num_dense_layers=2, num_hidden_layers=24))
    assert config.num_dense_layers < config.num_hidden_layers

def test_shrink_keeps_llama4_pixel_shuffle_and_projector_widths():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Llama4Config

    # As in Llama-4-Scout: the adapter MLP reads hidden_size / ratio**2 channels and
    # projects to the text width.
    config = shrink(Llama4Config(
        vision_config={"hidden_size": 768, "intermediate_size": 3072, "pixel_shuffle_ratio": 0.5,
                       "projector_input_dim": 2048, "projector_output_dim": 2048, "vision_output_dim": 2048},
        text_config={"hidden_size": 2048},
    ))
    vision, width = config.vision_config, config.text_config.hidden_size
    assert vision.intermediate_size == vision.hidden_size * 4
    assert vision.projector_input_dim == vision.projector_output_dim == vision.vision_output_dim == width

def test_shrink_keeps_csm_depth_decoder_reading_the_backbone_width():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import CsmConfig

    config = shrink(CsmConfig(hidden_size=2048, depth_decoder_config={"backbone_hidden_size": 2048}))
    assert config.depth_decoder_config.backbone_hidden_size == config.hidden_size
