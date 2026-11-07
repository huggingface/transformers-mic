"""Regression coverage for non-language-model fixture inputs."""

import copy
import hashlib
import json
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


def glm4v_source():
    from transformers import Glm4vConfig

    return Glm4vConfig(
        text_config={
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "rope_parameters": {
                "rope_type": "default",
                "mrope_section": [8, 12, 12],
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
            },
        }
    )


def test_config_overrides_that_resize_the_head_refit_mrope_sections(tmp_path, monkeypatch):
    # GLM-4.1V's planner overrode text_config.hidden_size to 128 after the rule had fitted
    # mrope_section to a 64-wide model: [1, 2, 1] split 8 rotary channels and the forward raised.
    from transformers import Glm4vForConditionalGeneration

    glm4v_source().save_pretrained(tmp_path)
    monkeypatch.setitem(build_fixtures.TESTER_OFF, str(tmp_path), {})
    recipe = {"config_overrides": {"fields": {"text_config.hidden_size": 128, "vision_config.out_hidden_size": 128}}}
    text = build_fixtures.shrunk_config(tmp_path, Glm4vForConditionalGeneration, recipe).text_config
    half = text.hidden_size // text.num_attention_heads * text.rope_parameters["partial_rotary_factor"] // 2
    assert sum(text.rope_parameters["mrope_section"]) == half == 8
    assert text.rope_parameters["mrope_section"] == [2, 3, 3]  # the checkpoint's 8:12:12, not [1, 2, 1] doubled

    unchanged = build_fixtures.shrunk_config(tmp_path, Glm4vForConditionalGeneration, None).text_config
    assert unchanged.rope_parameters["mrope_section"] == [1, 2, 1]


def test_fit_rope_leaves_the_rules_own_tables_unchanged():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import fit_rope_everywhere, shrink
    from transformers import Phi3Config

    half = 3072 // 24 * 3 // 4 // 2
    longrope = Phi3Config(
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
    for source in (glm4v_source(), longrope):
        original = source.to_dict()
        values = shrink(source).to_dict()
        assert fit_rope_everywhere(copy.deepcopy(values), original) == values


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


def test_shrink_keeps_voxtral_audio_frames_per_token():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import VoxtralConfig

    audio = shrink(VoxtralConfig()).audio_config
    assert audio.intermediate_size == 4 * audio.hidden_size


def _tiny_llava():
    from transformers import CLIPVisionConfig, LlamaConfig, LlavaConfig, LlavaForConditionalGeneration

    config = LlavaConfig(
        vision_config=CLIPVisionConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, image_size=28, patch_size=14),
        text_config=LlamaConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, vocab_size=64),
        image_token_id=63,
    )
    torch.manual_seed(0)
    return LlavaForConditionalGeneration(config).eval()


def test_fixture_keeps_the_checkpoint_legacy_names_and_reloads_exactly(tmp_path):
    from safetensors import safe_open
    from transformers import LlavaForConditionalGeneration
    from transformers.core_model_loading import revert_weight_conversion
    from transformers.conversion_mapping import get_model_conversion_mapping

    model = _tiny_llava()
    model._weight_conversions = get_model_conversion_mapping(model)  # legacy layout
    legacy = set(revert_weight_conversion(model, model.state_dict()))
    del model._weight_conversions
    assert any(k.startswith("vision_tower.vision_model.") for k in legacy)
    (tmp_path / "source").mkdir()
    build_fixtures.save(model, tmp_path / "out", tmp_path / "source", [], checkpoint={"keys": legacy, "binary": False})
    with safe_open(tmp_path / "out" / "model.safetensors", "pt") as f:
        assert set(f.keys()) <= legacy
    back = LlavaForConditionalGeneration.from_pretrained(tmp_path / "out")
    assert all(torch.equal(v, back.state_dict()[k]) for k, v in model.state_dict().items())


def test_fixture_keeps_a_head_checkpoint_prefix_and_writes_safetensors_for_a_bin_source(tmp_path):
    from safetensors import safe_open
    from transformers import BertConfig, BertModel

    torch.manual_seed(0)
    model = BertModel(BertConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, vocab_size=64)).eval()
    keys = {f"bert.{k}" for k in model.state_dict()}
    (tmp_path / "source").mkdir()
    build_fixtures.save(model, tmp_path / "out", tmp_path / "source", [], checkpoint={"keys": keys, "binary": True})
    # A pytorch_model.bin source still gets a safetensors fixture, with the source's names.
    assert not (tmp_path / "out" / "pytorch_model.bin").exists()
    with safe_open(tmp_path / "out" / "model.safetensors", "pt") as f:
        saved = set(f.keys())
    assert saved and all(k.startswith("bert.") for k in saved)
    back = BertModel.from_pretrained(tmp_path / "out")
    assert all(torch.equal(v, back.state_dict()[k]) for k, v in model.state_dict().items())


def test_shrink_renumbers_layer_indices_into_the_shrunk_depth():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Granite4VisionConfig, MllamaConfig

    mllama = shrink(MllamaConfig())
    vision_states = mllama.vision_config.num_hidden_layers + 1
    assert all(0 <= i < vision_states for i in mllama.vision_config.intermediate_layers_indices)
    text = mllama.text_config
    assert text.cross_attention_layers and all(0 < i < text.num_hidden_layers for i in text.cross_attention_layers)

    granite = Granite4VisionConfig(deepstack_layer_map=[(-19, 9), (-13, 6), (-7, 3), (-1, 0)], spatial_target_layers=[12, 15, 18, 21])
    granite.vision_config.num_hidden_layers = 27  # granite-vision-4.1-4b
    granite = shrink(granite)
    states, layers = granite.vision_config.num_hidden_layers + 1, granite.text_config.num_hidden_layers
    assert all(-states <= v < 0 and 0 <= t < layers for v, t in granite.deepstack_layer_map)
    assert granite.spatial_target_layers and all(0 <= t < layers for t in granite.spatial_target_layers)


def test_shrink_keeps_oneformer_pixel_decoder_width_equal_to_hidden_dim():
    # shi-labs/oneformer_*: conv_dim == mask_dim == hidden_dim (256). Shrinking only
    # hidden_dim made OneFormerTransformerModule build input projections it keeps in a
    # plain list, which from_pretrained leaves on the meta device: backward failed with
    # "expected device meta but got cpu".
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import OneFormerConfig, OneFormerForUniversalSegmentation

    config = shrink(OneFormerConfig(hidden_dim=256, conv_dim=256, mask_dim=256))
    assert config.conv_dim == config.mask_dim == config.hidden_dim < 256
    with torch.device("meta"):
        model = OneFormerForUniversalSegmentation(config)
    assert all(isinstance(p, torch.nn.Sequential) for p in model.model.transformer_module.input_projections)
    # A checkpoint that does project keeps its own widths.
    assert shrink(OneFormerConfig(hidden_dim=256, conv_dim=512, mask_dim=512)).conv_dim == 512


def test_shrink_keeps_projector_inputs_tied_to_the_shrunk_encoders():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import GraniteSpeechConfig, MllamaConfig

    vision = shrink(MllamaConfig()).vision_config
    assert vision.vision_output_dim == vision.hidden_size * (len(vision.intermediate_layers_indices) + 1)
    # granite-4.0-1b-speech: the projector cross-attends to the 1024-wide encoder.
    speech = shrink(GraniteSpeechConfig(projector_config={"model_type": "blip_2_qformer", "encoder_hidden_size": 1024}))
    assert speech.projector_config.encoder_hidden_size == speech.encoder_config.hidden_dim


def test_near_zero_batch_norm_scales_reset_to_one_before_overfitting():
    net = torch.nn.Sequential(torch.nn.Conv2d(3, 4, 3), torch.nn.BatchNorm2d(4), torch.nn.BatchNorm2d(4))
    torch.nn.init.normal_(net[1].weight, std=0.02)  # VitMatte's BatchNorm init
    assert build_fixtures.unit_batch_norm(net) == 1
    assert torch.equal(net[1].weight, torch.ones(4))


def test_overfit_reaches_targets_for_a_sigmoid_output_without_a_loss():
    class Matting(torch.nn.Module):  # VitMatte: alphas in [0, 1], no built-in loss
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 4)

        def forward(self, pixel_values):
            return {"alphas": torch.sigmoid(self.linear(pixel_values))}

    torch.manual_seed(0)
    result = build_fixtures.overfit(Matting(), {"pixel_values": torch.randn(2, 4)}, max_steps=500)
    assert result["converged"]


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
    # VITS: a perturbed input changes the waveform length.
    longer = {"waveform": torch.zeros(2, 7)}
    assert perturb.max_delta(longer, {"waveform": torch.zeros(2, 5)}) == inf


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


def test_shrink_keeps_mqa_and_the_resnet_identity_shortcut():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import GemmaConfig, LlamaConfig, ResNetConfig

    assert shrink(GemmaConfig(num_attention_heads=8, num_key_value_heads=1)).num_key_value_heads == 1
    assert shrink(LlamaConfig(num_attention_heads=32, num_key_value_heads=8)).num_key_value_heads == 2
    assert shrink(LlamaConfig(num_attention_heads=32, num_key_value_heads=32)).num_key_value_heads == 4
    resnet = shrink(ResNetConfig(embedding_size=64, hidden_sizes=[64, 128, 256, 512]))
    assert resnet.embedding_size == resnet.hidden_sizes[0]


def test_shrink_keeps_layer_kinds_the_config_derives_from_the_index():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import Llama4TextConfig
    from transformers_mic.shrink_view import kind_prefix

    # Scout: every layer MoE, every fourth without RoPE. moe_layers is an index list
    # as long as the depth, so re-indexing it as a per-layer list drops the MoE.
    config = Llama4TextConfig(num_hidden_layers=48, interleave_moe_layer_step=1)
    shrunk = shrink(config)
    assert kind_prefix(config, shrunk) is None
    assert shrunk.num_hidden_layers == 4


def test_test_checkpoints_are_never_mic_sources():
    from transformers_mic.hub import is_test_checkpoint

    assert is_test_checkpoint("optimum-intel-internal-testing/tiny-random-aya-base")
    assert is_test_checkpoint("peft-internal-testing/tiny-clip-text-2")
    assert is_test_checkpoint("hf-internal-testing/tiny-random-LlamaForCausalLM")
    assert is_test_checkpoint("tiny-random/mistral-3")
    assert not is_test_checkpoint("openai/whisper-tiny")
    assert not is_test_checkpoint("ai21labs/Jamba-tiny-dev")
    assert not is_test_checkpoint("tiiuae/Falcon-H1-Tiny-90M-Instruct")


def test_a_small_source_is_kept_whole_and_a_large_one_is_shrunk(tmp_path):
    from transformers import LlamaConfig, LlamaForCausalLM

    small = LlamaConfig(vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4)
    small.save_pretrained(tmp_path / "small")
    config = build_fixtures.shrunk_config(tmp_path / "small", LlamaForCausalLM)
    assert config.num_hidden_layers == 4 and config.hidden_size == 64
    assert build_fixtures.keeps_source(tmp_path / "small", LlamaForCausalLM, config)

    large = LlamaConfig(vocab_size=32000, hidden_size=1024, intermediate_size=4096, num_hidden_layers=8, num_attention_heads=16)
    large.save_pretrained(tmp_path / "large")
    config = build_fixtures.shrunk_config(tmp_path / "large", LlamaForCausalLM)
    assert config.num_hidden_layers < 8
    assert not build_fixtures.keeps_source(tmp_path / "large", LlamaForCausalLM, config)


def test_pixtral_vision_is_inert_to_pixel_scale():
    from transformers import Mistral3Config

    config = Mistral3Config()
    assert config.vision_config.model_type == "pixtral"
    assert "pixel_values:halve" in build_fixtures.config_insensitive(config, {"pixel_values": None})
    assert "pixel_values:halve" not in build_fixtures.config_insensitive(config, {"input_ids": None})



def test_layer_pattern_keeps_every_flagged_layer_kind_and_its_kv_source():
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    import layer_patterns

    # Mistral-Large-4 in 8 layers: dense 0, NoPE every 4th, a shared indexer from 5,
    # and layer 7 reading layer 6's KV cache.
    lists = {
        "layer_types": ["indexed_attention"] * 8,
        "mlp_layer_types": ["dense"] + ["sparse"] * 7,
        "no_rope_layers": [1, 1, 1, 0, 1, 1, 1, 0],
        "indexer_types": ["full"] * 5 + ["shared"] * 3,
        "kv_source_layers": [0, 1, 2, 3, 4, 5, 6, 6],
        "num_attention_heads_per_layer": [4, 4, 8, 8, 4, 4, 8, 8],  # a width, not a kind
    }
    config = SimpleNamespace(num_hidden_layers=8, first_k_dense_replace=1, **lists)
    config.to_dict = lambda: dict(lists, num_hidden_layers=8)
    kept = layer_patterns.pattern_indices(config)
    assert kept == [0, 1, 3, 5, 6, 7]  # 6 is kept as layer 7's KV source
    values = {}
    layer_patterns.reindex_references(values, config, kept)
    assert values["kv_source_layers"] == [0, 1, 2, 3, 4, 4]


def test_shrink_adopts_tester_sizes_the_rule_misses(monkeypatch):
    from transformers import Qwen3NextConfig, Qwen3NextForCausalLM

    from transformers_mic import tester

    original = Qwen3NextConfig(num_hidden_layers=4, vocab_size=128)
    shrunk = Qwen3NextConfig(**{**original.to_dict(), "hidden_size": 64, "intermediate_size": 128,
                                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                                "num_experts": 4, "moe_intermediate_size": 32, "shared_expert_intermediate_size": 128})
    monkeypatch.setattr(tester, "tester_values", lambda model_type, ref="main": (
        "tests/models/qwen3_next", {"linear_key_head_dim": 16, "linear_value_head_dim": 16, "vocab_size": 99}))
    adopted = build_fixtures.adopt_tester_sizes(Qwen3NextForCausalLM, original.to_dict(), shrunk)
    assert adopted.linear_key_head_dim == 16 and adopted.linear_value_head_dim == 16
    assert adopted.vocab_size == 128  # preserved: the fixture loads the real tokenizer
    assert adopted._mic_tester_sizes["fields"] == {"linear_key_head_dim": 16, "linear_value_head_dim": 16}
    # A tester size that does not build is dropped, keeping the rule's config.
    monkeypatch.setattr(tester, "tester_values", lambda model_type, ref="main": ("t", {"linear_key_head_dim": 3}))
    monkeypatch.setattr(Qwen3NextForCausalLM, "_from_config", classmethod(lambda cls, c: (_ for _ in ()).throw(ValueError("bad"))))
    assert build_fixtures.adopt_tester_sizes(Qwen3NextForCausalLM, original.to_dict(), shrunk) is shrunk


def test_tester_sizes_stay_out_of_the_config_and_fall_back_on_a_failed_build(monkeypatch):
    sizes = {"source": "t", "fields": {"encoder_hidden_dim": 32}}
    monkeypatch.setattr(build_fixtures, "TESTER_SIZES", {"src": dict(sizes)})
    monkeypatch.setattr(build_fixtures, "TESTER_OFF", {})
    calls = []

    def build(*args, **kwargs):
        calls.append(dict(build_fixtures.TESTER_OFF))
        if len(calls) == 1:
            raise SystemExit("the contract fails on the random fixture:\nRuntimeError: expected 256 channels, got 32")
        return {"ok": True}

    monkeypatch.setattr(build_fixtures, "_build_fixture", build)
    assert build_fixtures.build_fixture() == {"ok": True}
    assert calls[1]["src"]["fields"] == {"encoder_hidden_dim": 32}
    assert "256 channels" in calls[1]["src"]["build_failed"]
    # Without adopted sizes a failure is not retried.
    monkeypatch.setattr(build_fixtures, "TESTER_SIZES", {})
    monkeypatch.setattr(build_fixtures, "_build_fixture", lambda *a, **k: (_ for _ in ()).throw(SystemExit("boom")))
    with pytest.raises(SystemExit):
        build_fixtures.build_fixture()



def test_a_missing_package_is_reported_not_retried_at_other_sizes(monkeypatch, capsys):
    # Mistral-Large-4: PixtralProcessor needs torchvision, which no fixture size provides.
    monkeypatch.setattr(build_fixtures, "TESTER_SIZES", {"src": {"source": "t", "fields": {"hidden_size": 32}}})
    monkeypatch.setattr(build_fixtures, "TESTER_OFF", {})
    calls = []

    def build(*args, **kwargs):
        calls.append(1)
        raise SystemExit(
            "the contract fails on the random fixture:\nTraceback (most recent call last):\n"
            "ImportError: \nPixtralProcessor requires the Torchvision library but it was not found in your environment. "
            "Check out the instructions on the\ninstallation page: https://pytorch.org/get-started/locally/\n"
            "Please note that you may need to restart your runtime after installation.\n"
        )

    monkeypatch.setattr(build_fixtures, "_build_fixture", build)
    with pytest.raises(SystemExit) as raised:
        build_fixtures.build_fixture()
    assert calls == [1]  # no rebuild with the shrink rule's sizes
    assert str(raised.value).startswith(
        "ImportError: PixtralProcessor requires the Torchvision library but it was not found in your environment."
    )
    assert "Torchvision" in capsys.readouterr().err  # the full text stays in the log
    assert build_fixtures.missing_package(SystemExit("RuntimeError: expected 256 channels")) is None
    assert build_fixtures.missing_package(ModuleNotFoundError("No module named 'timm'")) == "No module named 'timm'"


def test_shrink_keeps_widths_that_equal_hidden_size_equal():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import ElectraConfig, OPTConfig, OPTForCausalLM

    # OPT-125m: word_embed_proj_dim == hidden_size means no project_in/out layers.
    opt = shrink(OPTConfig(hidden_size=768, word_embed_proj_dim=768, ffn_dim=3072, num_attention_heads=12))
    assert opt.word_embed_proj_dim == opt.hidden_size == 64
    with torch.device("meta"):
        names = set(OPTForCausalLM(opt).state_dict())
    assert not any("project_in" in n or "project_out" in n for n in names)
    # OPT-350m projects (512 vs 1024): the projection layers must survive the shrink.
    projected = shrink(OPTConfig(hidden_size=1024, word_embed_proj_dim=512, ffn_dim=4096, num_attention_heads=16))
    assert projected.word_embed_proj_dim != projected.hidden_size
    # ELECTRA-base: embedding_size == hidden_size means no embeddings_project.
    electra = shrink(ElectraConfig(hidden_size=768, embedding_size=768, intermediate_size=3072, num_attention_heads=12))
    assert electra.embedding_size == electra.hidden_size


def test_tied_tensor_keeps_the_name_the_source_serializes(tmp_path):
    from safetensors import safe_open
    from transformers import OPTConfig, OPTForCausalLM

    torch.manual_seed(0)
    model = OPTForCausalLM(OPTConfig(hidden_size=16, word_embed_proj_dim=16, ffn_dim=32, num_attention_heads=2, num_hidden_layers=1, vocab_size=64)).eval()
    # OPT-125m's pytorch_model.bin stores the tied embedding as lm_head.weight only.
    keys = {k for k in model.state_dict() if k != "model.decoder.embed_tokens.weight"}
    (tmp_path / "source").mkdir()
    build_fixtures.save(model, tmp_path / "out", tmp_path / "source", [], checkpoint={"keys": keys, "binary": True})
    with safe_open(tmp_path / "out" / "model.safetensors", "pt") as f:
        saved = set(f.keys())
    assert "lm_head.weight" in saved and "model.decoder.embed_tokens.weight" not in saved
    back, info = OPTForCausalLM.from_pretrained(tmp_path / "out", output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert all(torch.equal(v, back.state_dict()[k]) for k, v in model.state_dict().items())


def test_missing_parameters_in_dropped_layers_are_left_out():
    # A.X-K1: the real 61-layer checkpoint lacks every MoE layer's correction bias;
    # the shrunk fixture has two layers, so only layer 1's exists to omit.
    missing = [f"model.layers.{i}.mlp.gate.e_score_correction_bias" for i in range(1, 61)]
    names = {"model.layers.0.mlp.up_proj.weight", "model.layers.1.mlp.gate.e_score_correction_bias"}
    assert build_fixtures.in_fixture(missing, names) == ["model.layers.1.mlp.gate.e_score_correction_bias"]


def test_a_tensor_missing_only_in_a_dropped_layer_is_left_out():
    # Kimi K2.5: the real vision tower lacks layer 2's fc2; the fixture keeps layers 0 and 1, which have it.
    names = {"model.vision_tower.layers.0.mlp.fc2.weight", "model.vision_tower.layers.1.mlp.fc2.weight"}
    assert build_fixtures.in_fixture(["model.vision_tower.layers.2.mlp.fc2.weight"], names) == []


def test_a_missing_parameter_the_fixture_cannot_mirror_still_fails():
    names = {"model.layers.0.mlp.gate.weight"}
    with pytest.raises(SystemExit, match="cannot mirror"):
        build_fixtures.in_fixture(["model.layers.5.mlp.gate.e_score_correction_bias"], names)
    with pytest.raises(SystemExit, match="cannot mirror"):
        build_fixtures.in_fixture(["lm_head.bias"], names)


def test_sentence_transformer_module_configs_are_copied_and_hashed(tmp_path):
    (tmp_path / "modules.json").write_text(json.dumps([
        {"path": ""}, {"path": "1_Pooling"}, {"path": "2_Normalize"}
    ]))
    pooling = tmp_path / "1_Pooling"
    pooling.mkdir()
    (pooling / "config.json").write_text('{"embedding_dimension": 768}')
    (pooling / "model.safetensors").write_bytes(b"not a config")
    names = build_fixtures.copied_files(tmp_path)
    assert "1_Pooling/config.json" in names
    assert "1_Pooling/model.safetensors" not in names
    hashes = build_fixtures.run.file_hashes(tmp_path)
    assert "1_Pooling/config.json" in hashes
    (pooling / "config.json").write_text('{"embedding_dimension": 64}')
    assert build_fixtures.run.file_hashes(tmp_path) != hashes


def test_sentence_transformer_module_path_cannot_escape_source(tmp_path):
    (tmp_path / "modules.json").write_text('[{"path": "../outside"}]')
    with pytest.raises(ValueError, match="Invalid Sentence Transformers module path"):
        build_fixtures.copied_files(tmp_path)


def test_shrink_keeps_relations_the_widths_imply():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import shrink
    from transformers import GPTJConfig, MobileBertConfig, XLNetConfig

    # GPT-J rotates rotary_dim channels of a head it derives from n_embd / n_head.
    gptj = shrink(GPTJConfig(n_embd=4096, n_head=16, rotary_dim=64))
    assert gptj.rotary_dim == 64 * (gptj.n_embd // gptj.n_head) // 256
    # XLNet validates d_head == d_model // n_head: only that config is repaired.
    xlnet = shrink(XLNetConfig(d_model=1024, n_head=16, d_head=64))
    assert xlnet.d_head == xlnet.d_model // xlnet.n_head
    # MobileBERT's bottleneck widths stay the same fraction of hidden_size.
    mobile = shrink(MobileBertConfig())
    assert mobile.intra_bottleneck_size == mobile.true_hidden_size == mobile.hidden_size // 4


def test_input_layouts_keep_the_first_layout_and_add_what_it_lacks():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
    from sweep import input_layouts
    from transformers import VideoMAEConfig, VideoMAEModel

    config = VideoMAEConfig(hidden_size=32, num_hidden_layers=1, num_attention_heads=2, intermediate_size=64, num_frames=4, image_size=32, patch_size=16)
    layouts = input_layouts(VideoMAEModel(config), config)
    assert layouts[0]["pixel_values"].shape == (2, 3, 32, 32)  # what the sweep sent before
    assert layouts[1]["pixel_values"].shape == (2, 4, 3, 32, 32)  # frames, as the processor sends
