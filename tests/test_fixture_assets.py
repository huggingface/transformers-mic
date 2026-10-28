"""Regression coverage for non-language-model fixture inputs."""

import hashlib
import sys
from pathlib import Path

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
