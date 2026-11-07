"""Can the fixture recipe shrink every Transformers architecture? Configs only.

    python sweep.py prefetch            # resolve and download each doc checkpoint's config.json
    python sweep.py run [--jobs 6]      # shrink, build, forward, backward, one process per type
    python sweep.py summary

For every model type with a documentation checkpoint (`@auto_docstring(checkpoint=...)`
on its config class), load the real config, shrink it with one generic rule
(widths by field name in every sub-config, depth by layer pattern, MoE to 4
experts), build the model's primary head with random weights, and run forward
and backward on synthetic inputs built from the config. Each type stops at
the first stage that fails, with the error recorded. No weights or processors
are downloaded, so this measures the generic shrink rule, not processor paths.
"""

import argparse
import glob
import inspect
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).parent
SOURCES = HERE / "results" / "sweep_sources.json"
RESULTS = HERE / "results" / "sweep.json"

# Generic width rule: field name -> fixture value, applied in every (sub-)config.
WIDTHS = {
    "hidden_size": 64, "intermediate_size": 128, "d_model": 64, "d_ff": 128,
    "embed_dim": 64, "out_hidden_size": 64, "ffn_dim": 128, "encoder_ffn_dim": 128, "decoder_ffn_dim": 128,
    "num_attention_heads": 4, "num_heads": 4, "encoder_attention_heads": 4,
    "decoder_attention_heads": 4, "head_dim": 16, "d_kv": 16, "attention_head_dim": 16,
    # DeBERTa-v2 names its head size and pooler width explicitly.
    "attention_head_size": 16, "pooler_hidden_size": 64,
    # Per-layer embeddings (Qwen4-Exp) are vocab-sized: shrink their width too.
    "ple_embed_dim": 16,
    "moe_intermediate_size": 32, "shared_intermediate_size": 128,
    "shared_expert_intermediate_size": 128, "projection_dim": 32,
    "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mamba_d_state": 16, "state_size": 16, "ssm_state_size": 16, "mamba_n_heads": 8,
    "mamba_d_head": "auto", "mamba_chunk_size": 8, "chunk_size": 8,
    # GPT-2, XLM, DistilBERT, and others name the same widths differently.
    "n_embd": 64, "n_inner": 128, "n_head": 4, "emb_dim": 64, "dim": 64, "hidden_dim": 128,
    "n_heads": 4,
}
# Fields that, when equal to hidden_size in the checkpoint, must stay equal to it.
LINKED_TO_HIDDEN = ("projection_size", "word_embed_proj_dim", "embedding_size", "codebook_dim")
# The same for hidden_dim: OneFormer projects its pixel decoder's conv_dim features to
# hidden_dim only when they differ, and keeps those projections in a plain list that
# from_pretrained never materializes (they stay on the meta device; backward then fails).
LINKED_TO_HIDDEN_DIM = ("conv_dim", "mask_dim")
# The longest layer prefix a fixture keeps to build every layer kind.
MAX_PREFIX = 8
EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
# Experts per token: no more than the experts kept (Diffusion Gemma's top_k_experts 8, ERNIE 4.5's moe_k 6).
TOP_K = ("num_experts_per_tok", "top_k_experts", "moe_k")
DEPTHS = ("num_layers", "num_decoder_layers", "encoder_layers", "decoder_layers", "n_layer", "n_layers")
# Primary head, first match wins: the usage path a contract would cover first.
HEADS = (
    "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES",
    "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES",
    "MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES",
    "MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES",
    "MODEL_FOR_MASKED_LM_MAPPING_NAMES",
    "MODEL_FOR_CTC_MAPPING_NAMES",
    "MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES",
    "MODEL_MAPPING_NAMES",
)
# Fast integration tests in transformers#49103.
FAST_49103 = (
    "bert", "clip", "detr", "internvl", "llama", "llava", "musicgen", "paligemma", "qwen2",
    "qwen2_5_vl", "segformer", "smolvlm", "t5", "video_llama_3", "vit", "wav2vec2", "whisper",
)


def doc_checkpoints(transformers_src):
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES

    by_class = {name: model_type for model_type, name in CONFIG_MAPPING_NAMES.items()}
    found = {}
    pattern = re.compile(
        r'@auto_docstring\(\s*checkpoint="([^"]+)"[^)]*\)\s*(?:@[^\n]+\n\s*)*class (\w+)\('
    )
    for path in glob.glob(f"{transformers_src}/models/*/configuration_*.py"):
        for match in pattern.finditer(Path(path).read_text()):
            if match.group(2) in by_class:
                found[by_class[match.group(2)]] = match.group(1)
    return {t: found.get(t) for t in CONFIG_MAPPING_NAMES}


def prefetch(args):
    import transformers
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    sources = doc_checkpoints(Path(transformers.__file__).parent)

    def types_in(config):
        """model_type of a raw config.json and of every nested config in it."""
        found = {config.get("model_type")}
        for value in config.values():
            if isinstance(value, dict) and "model_type" in value:
                found |= types_in(value)
        return found

    def try_repo(model_type, repo):
        sha = api.model_info(repo).sha
        config = json.loads(Path(hf_hub_download(repo, "config.json", revision=sha)).read_text())
        if model_type not in types_in(config):
            raise ValueError(f"config.json has model_type {config.get('model_type')}")
        return sha

    from transformers_mic.hub import is_test_checkpoint

    def fetch(item):
        model_type, repo = item
        row = {"doc_checkpoint": repo}
        if repo is None:
            row["doc_status"] = "no_doc_checkpoint"
        elif is_test_checkpoint(repo):
            row["doc_status"] = "test_checkpoint"
        else:
            try:
                row.update(repo=repo, revision=try_repo(model_type, repo), source="doc", status="ok")
                row["doc_status"] = "ok"
                return model_type, row
            except Exception as exc:
                row["doc_status"] = (
                    "wrong_model_type" if isinstance(exc, ValueError) else type(exc).__name__
                )
        # Fall back to the most downloaded ungated, trained Transformers model with this type.
        try:
            candidates = api.list_models(
                filter=[model_type, "transformers"], sort="downloads", limit=10, expand=["gated"]
            )
            for candidate in candidates:
                if candidate.gated or is_test_checkpoint(candidate.id):
                    continue
                try:
                    sha = try_repo(model_type, candidate.id)
                except Exception:
                    continue
                row.update(repo=candidate.id, revision=sha, source="hub", status="ok")
                return model_type, row
        except Exception as exc:
            row["error"] = str(exc)[:200]
        row["status"] = "no_checkpoint"
        return model_type, row

    with ThreadPoolExecutor(16) as pool:
        rows = dict(pool.map(fetch, sources.items()))
    SOURCES.write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
    print({s: sum(r["status"] == s for r in rows.values()) for s in {r["status"] for r in rows.values()}})


def sub_configs(config):
    """The config and every nested config, depth first."""
    from transformers import PretrainedConfig

    yield config
    for key in getattr(config, "sub_configs", {}) or {}:
        value = getattr(config, key, None)
        if isinstance(value, PretrainedConfig):
            yield from sub_configs(value)


def subs_of(cfg):
    """The config's direct sub-configs by key."""
    return {key: getattr(cfg, key, None) for key in getattr(cfg, "sub_configs", {}) or {} if getattr(cfg, key, None) is not None}


def head_width(values):
    """The attention head width a config derives from its widths, or None."""
    width = values.get("hidden_size") or values.get("n_embd") or values.get("d_model")
    heads = values.get("num_attention_heads") or values.get("n_head")
    return width // heads if isinstance(width, int) and isinstance(heads, int) and heads else None


def repair(cfg, values):
    """Restore the relations a config validates between the generic rule's values, in place."""
    original = cfg.to_dict()
    # XLNet checks d_head == d_model // n_head.
    if original.get("d_head") and original["d_head"] == head_width(original):
        values["d_head"] = head_width(values)
    # GPT-Neo lists one attention kind per layer and a (pattern, repeats) summary of it.
    if isinstance(values.get("attention_types"), list) and isinstance(values.get("num_layers"), int):
        pattern = values["attention_types"][0][0]
        values["attention_types"] = [[pattern, max(1, values["num_layers"] // len(pattern))]]
        values["num_layers"] = len(pattern) * values["attention_types"][0][1]
        values["attention_layers"] = (pattern * values["num_layers"])[: values["num_layers"]]
    subs = subs_of(cfg)
    for key, sub in subs.items():
        if not isinstance(values.get(key), dict):
            continue
        # A field named after a sibling's width reads that sibling (Nemotron 3 Diarization's
        # head_config.audio_hidden_size is audio_config.hidden_size).
        for other, sibling in subs.items():
            field = f"{other.removesuffix('_config')}_hidden_size"
            width = values[other].get("hidden_size") if isinstance(values.get(other), dict) else None
            if other != key and width and getattr(sub, field, None) == sibling.to_dict().get("hidden_size", -1):
                values[key][field] = width
        # Gradium Phonon's flow head is conditioned on the backbone's hidden states.
        if getattr(sub, "condition_dim", None) == original.get("hidden_size", -1) and values.get("hidden_size"):
            values[key]["condition_dim"] = values["hidden_size"]
    # Fun-ASR-Nano's adaptor checks intermediate_size == text hidden_size // 4.
    adaptor, text = values.get("adaptor_config"), values.get("text_config")
    if isinstance(adaptor, dict) and isinstance(text, dict) and cfg.adaptor_config.intermediate_size == cfg.text_config.hidden_size // 4:
        adaptor["intermediate_size"] = text["hidden_size"] // 4
    # Molmo2's adapter reads vision layers by index: keep one per entry within the shrunk tower.
    adapter, vision = values.get("adapter_config"), values.get("vision_config")
    if isinstance(adapter, dict) and isinstance(adapter.get("vision_feature_layer"), list) and isinstance(vision, dict):
        old, new = cfg.vision_config.num_hidden_layers, vision["num_hidden_layers"]
        adapter["vision_feature_layer"] = [min(i * new // old, new - 1) if i >= 0 else i for i in adapter["vision_feature_layer"]]


def layer_pattern(config):
    """Indices to keep: the first layer of each distinct kind (see layer_patterns.py)."""
    sys.path.insert(0, str(HERE))
    from layer_patterns import pattern_indices

    return pattern_indices(config)


def shrink(config):
    """Apply the generic rule in place, innermost configs first, then rebuild each."""
    from transformers import PretrainedConfig

    def rebuilt(cfg, through=None):
        values = cfg.to_dict()
        # A slim fixture holds dense random weights, whatever the checkpoint's format.
        values.pop("quantization_config", None)
        # Random fixture weights need a signal: Unlimited-OCR's SAM tower inits with
        # 1e-10, which vanishes to 1e-37 over the tower and makes images inert.
        init = values.get("initializer_range")
        if isinstance(init, float) and 0 < init < 1e-3:
            values["initializer_range"] = 0.02
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if isinstance(sub, PretrainedConfig):
                values[key] = rebuilt(sub).to_dict()
        for key, value in WIDTHS.items():
            # 0 is a choice, not a width (GLM-5.3's NoPE qk_rope_head_dim).
            if key in values and values[key] not in (None, 0) and not isinstance(values[key], (list, dict)):
                values[key] = value
        if "num_key_value_heads" in values and values["num_key_value_heads"]:
            try:
                heads = cfg.num_attention_heads if hasattr(cfg, "num_attention_heads") else None
            except AttributeError:  # Step-3.5 lists head counts per layer and refuses a global read
                heads = None
            kv = values["num_key_value_heads"]
            # MHA stays MHA, MQA keeps its single KV head, GQA groups stay groups.
            values["num_key_value_heads"] = 4 if heads and kv >= heads else 1 if kv == 1 else 2
        # MLA models whose rotary tables are sized by head_dim (DeepSeek-V3, Kimi
        # K2.5) set it to qk_rope_head_dim; keep them equal after shrinking.
        if cfg.to_dict().get("head_dim") and cfg.to_dict().get("head_dim") == cfg.to_dict().get("qk_rope_head_dim"):
            values["head_dim"] = values["qk_rope_head_dim"]
        # Widths that equal hidden_size by design: equal means "no projection"
        # (OPT's project_in/out, ELECTRA's embeddings_project) or must meet another
        # tower (SigLIP's projection_size). Shrinking only hidden_size would add layers
        # the checkpoint does not have, so keep them equal.
        original = cfg.to_dict()
        for key in LINKED_TO_HIDDEN:
            if original.get(key) and original.get(key) == original.get("hidden_size"):
                values[key] = values["hidden_size"]
        # Aliases of a width (Kimi K2.5's vision tower repeats hidden_size as mm_hidden_size and
        # vt_hidden_size, intermediate_size as vt_intermediate_size) shrink with it.
        for key, value in original.items():
            width = next((w for w in ("hidden_size", "intermediate_size") if key in (f"mm_{w}", f"vt_{w}")), None)
            if width and isinstance(value, int) and value == original.get(width):
                values[key] = values[width]
        # A top-k sparse attention (Mistral-Large-4's indexer, 2048) at least as long as
        # the contract's sequences selects every token. Contract prompts can be 5 tokens
        # (DeepSeek-V3.2: top-k 8 left every output unchanged), so keep 2.
        if isinstance(values.get("index_topk"), int):
            values["index_topk"] = min(values["index_topk"], 2)
        if values.get("q_lora_rank"):
            values["q_lora_rank"] = 16
        for key in EXPERTS:
            if isinstance(values.get(key), int) and values[key] > 4:
                values[key] = 4
        for key in TOP_K:
            if isinstance(values.get(key), int):
                values[key] = min(values[key], 2)
        # GPT-J and CodeGen rotate rotary_dim channels of a head they derive from the widths
        # (no head_dim): keep the rotated fraction of the shrunk head.
        if isinstance(original.get("rotary_dim"), int) and "head_dim" not in original and head_width(original) and head_width(values):
            values["rotary_dim"] = max(2, original["rotary_dim"] * head_width(values) // head_width(original) // 2 * 2)
        # MobileBERT's bottleneck widths are a fraction of hidden_size: keep the fraction.
        if isinstance(original.get("intra_bottleneck_size"), int):
            for key in ("embedding_size", "intra_bottleneck_size", "true_hidden_size"):
                if isinstance(original.get(key), int):
                    values[key] = max(8, original[key] * values["hidden_size"] // original["hidden_size"])
        # EfficientNet derives its channels from width_coefficient; hidden_dim is the last of them.
        if "width_coefficient" in original and "hidden_dim" in original:
            values["hidden_dim"] = original["hidden_dim"]
        for key in LINKED_TO_HIDDEN_DIM:
            if isinstance(original.get(key), int) and original.get(key) == original.get("hidden_dim"):
                values[key] = values["hidden_dim"]
        # LayoutLMv3 concatenates 4 coordinate and 2 shape embeddings into hidden_size.
        if original.get("coordinate_size") and 4 * original["coordinate_size"] + 2 * original.get("shape_size", 0) == original.get("hidden_size"):
            values["coordinate_size"] = values["hidden_size"] // 8
            values["shape_size"] = (values["hidden_size"] - 4 * values["coordinate_size"]) // 2
        for key in ("n_group", "topk_group"):
            if isinstance(values.get(key), int):
                values[key] = 1
        for key in DEPTHS:
            if isinstance(values.get(key), int):
                values[key] = min(values[key], 2)
        if isinstance(values.get("hidden_sizes"), list):
            values["hidden_sizes"] = [16 * 2**i for i in range(len(values["hidden_sizes"]))]
            # ResNet's first stage keeps its identity shortcut when the stem width equals it.
            if cfg.to_dict().get("embedding_size") == cfg.hidden_sizes[0]:
                values["embedding_size"] = values["hidden_sizes"][0]
        if isinstance(values.get("depths"), list):
            values["depths"] = [1] * len(values["depths"])
        # Per-stage head counts must divide the shrunk per-stage widths (PVT's
        # hidden_sizes, Swin's embed_dim doubling per stage): keep the largest
        # head count, no more than the original, that does.
        # A Swin whose last stage is the model width (CLAP's audio tower:
        # patch_embeds_hidden_size doubled per stage == hidden_size) keeps that link.
        if isinstance(values.get("patch_embeds_hidden_size"), int) and isinstance(values.get("depths"), list):
            last = 2 ** (len(values["depths"]) - 1)
            if cfg.patch_embeds_hidden_size * last == cfg.hidden_size:
                values["patch_embeds_hidden_size"] = 16
                values["hidden_size"] = 16 * last
                if values.get("projection_hidden_size") == cfg.hidden_size:
                    values["projection_hidden_size"] = values["hidden_size"]
        stages = values.get("hidden_sizes")
        if not isinstance(stages, list) and isinstance(values.get("embed_dim"), int) and isinstance(values.get("depths"), list):
            stages = [values["embed_dim"] * 2**i for i in range(len(values["depths"]))]
        if not isinstance(stages, list) and isinstance(values.get("patch_embeds_hidden_size"), int) and isinstance(values.get("depths"), list):
            stages = [values["patch_embeds_hidden_size"] * 2**i for i in range(len(values["depths"]))]
        if isinstance(stages, list):
            for key in ("num_attention_heads", "num_heads"):
                heads = values.get(key)
                if isinstance(heads, list) and len(heads) == len(stages):
                    values[key] = [max(d for d in range(1, h + 1) if w % d == 0) for w, h in zip(stages, heads)]
        # Layer-indexed backbones name outputs after the real depth; let them default.
        # Stage-based ones (depths per stage) keep their stage count, so keep them.
        if not isinstance(values.get("depths"), list):
            for key in ("out_features", "out_indices"):
                if values.get(key) is not None:
                    values[key] = None
        # A detection encoder reads the backbone stages it was built for.
        backbone = values.get("backbone_config")
        if (
            isinstance(values.get("encoder_in_channels"), list)
            and isinstance(backbone, dict)
            and isinstance(backbone.get("hidden_sizes"), list)
            and backbone.get("out_indices")
        ):
            values["encoder_in_channels"] = [backbone["hidden_sizes"][i - 1] for i in backbone["out_indices"]]
        # Stage-based models (depths per stage) count stages in num_hidden_layers: keep them.
        if isinstance(values.get("num_hidden_layers"), int) and not isinstance(values.get("depths"), list):
            kept = layer_pattern(cfg) if hasattr(cfg, "num_hidden_layers") else [0, 1]
            kept = kept if len(kept) > 1 else list(range(min(2, values["num_hidden_layers"])))
            sys.path.insert(0, str(HERE))
            from layer_patterns import derived_types

            if derived_types(cfg):
                # Kinds follow the layer index, so only a prefix keeps them: up to each kind's first layer.
                kept = list(range(max(kept) + 1))
            if through:
                kept = list(range(through))
            for key, value in cfg.to_dict().items():
                if isinstance(value, list) and len(value) == values["num_hidden_layers"]:
                    values[key] = [value[i] for i in kept if i < len(value)]
            from layer_patterns import reindex_references

            reindex_references(values, cfg, kept)  # sources precede their layer, so a prefix holds them
            if getattr(cfg, "num_kv_shared_layers", None) is not None:
                values["num_kv_shared_layers"] = sum(i >= cfg.num_hidden_layers - cfg.num_kv_shared_layers for i in kept)
            # Leading dense layers: keep at least one MoE layer, or the fixture has no
            # experts at all (LFM2-MoE: 2 dense of 24, 2 layers kept).
            for key in ("first_k_dense_replace", "num_dense_layers"):
                if getattr(cfg, key, None) is not None:
                    values[key] = min(sum(i < getattr(cfg, key) for i in kept), len(kept) - 1)
            for key, value in cfg.to_dict().items():
                # Sapiens2 lists KV heads per layer: group them like num_key_value_heads.
                if key.startswith("num_key_value_heads_") and isinstance(values.get(key), list):
                    heads, old_heads = values.get("num_attention_heads") or 4, cfg.to_dict().get("num_attention_heads") or max(value)
                    values[key] = [heads if kv >= old_heads else 1 if kv == 1 else 2 for kv in values[key]]
            per_layer = values.get("per_layer_config")
            if isinstance(per_layer, dict):
                # Keys are layer indices, sometimes zero-padded (Gemma 4's "04").
                values["per_layer_config"] = {
                    str(n): value for n, i in enumerate(kept)
                    for key, value in per_layer.items() if int(key) == i
                }
            if getattr(cfg, "mlp_only_layers", None):
                values["mlp_only_layers"] = [n for n, i in enumerate(kept) if i in cfg.mlp_only_layers]
            # A layer-id list naming a layer that was dropped moves to the kept
            # layer of the same kind (Qwen4-Exp's PLE layer must stay linear attention).
            kinds = getattr(cfg, "layer_types", None)
            if isinstance(kinds, list) and len(kinds) == cfg.num_hidden_layers:
                def moved(ids):
                    same = {kinds[i]: n for n, i in reversed(list(enumerate(kept)))}
                    return sorted({kept.index(i) if i in kept else same[kinds[i]] for i in ids if kinds[i] in same})

                if isinstance(values.get("ple_layer_ids"), list):  # one-indexed
                    values["ple_layer_ids"] = [n + 1 for n in moved([i - 1 for i in values["ple_layer_ids"]])]
                linear = values.get("linear_attn_config")
                if isinstance(linear, dict):
                    # GLM-5.3 lists its linear and full attention layers by index.
                    for key in ("kda_layers", "full_attn_layers"):
                        if isinstance(linear.get(key), list):
                            linear[key] = moved(linear[key])
                    for key, value in (("num_heads", 4), ("head_dim", 16)):
                        if isinstance(linear.get(key), int):
                            linear[key] = value
            values["num_hidden_layers"] = len(kept)
        # CLIPSeg's decoder reads vision hidden states extract_layers (3, 6, 9): keep
        # one entry each, within the shrunk vision depth.
        vision = values.get("vision_config")
        if isinstance(values.get("extract_layers"), list) and isinstance(vision, dict) and vision.get("num_hidden_layers"):
            depth, top = vision["num_hidden_layers"], max(values["extract_layers"]) + 1
            values["extract_layers"] = [min(i * depth // top, depth - 1) for i in values["extract_layers"]]
        # Fields that name layers of the original depth must name layers of the shrunk
        # one, in order (Mllama's vision intermediate_layers_indices and text
        # cross_attention_layers, Granite 4 Vision's deepstack_layer_map of
        # (vision hidden state, text layer) pairs and spatial_target_layers). Hidden-state indices count the
        # embeddings too (depth + 1 entries) and keep their sign.
        def depth(key):
            sub = getattr(cfg, key, None) if key else cfg
            new = values.get(key) if key else values
            return getattr(sub, "num_hidden_layers", None), (new or {}).get("num_hidden_layers")

        def state(i, old, new):
            position = i if i >= 0 else old + 1 + i
            moved = min(position * new // old, new)
            return moved if i >= 0 else moved - (new + 1)

        old_v, new_v = depth("vision_config")
        old_t, new_t = depth("text_config")
        vision_values = values.get("vision_config")
        if isinstance(vision_values, dict) and old_v and new_v and old_v != new_v:
            original = getattr(cfg.vision_config, "intermediate_layers_indices", None)
            if isinstance(original, list):
                vision_values["intermediate_layers_indices"] = sorted({state(i, old_v, new_v) for i in original})
        text_values = values.get("text_config")
        if isinstance(text_values, dict) and old_t and new_t and old_t != new_t:
            original = getattr(cfg.text_config, "cross_attention_layers", None)
            if isinstance(original, list):
                # Keep layer 0 a self-attention layer when the original's is (Llama 3.2 Vision: [3, 8, ...]).
                first = 0 if 0 in original or new_t == 1 else 1
                kept = sorted({min(max(i * new_t // old_t, first), new_t - 1) for i in original if 0 <= i < old_t})
                text_values["cross_attention_layers"] = kept or [new_t - 1]
        targets = getattr(cfg, "spatial_target_layers", None)  # Granite 4 Vision: text layers
        if isinstance(targets, list) and old_t and new_t:
            values["spatial_target_layers"] = sorted({min(t * new_t // old_t, new_t - 1) for t in targets})
        mapping = getattr(cfg, "deepstack_layer_map", None)
        if isinstance(mapping, (list, tuple)) and old_v and new_v and old_t and new_t:
            values["deepstack_layer_map"] = [
                [state(v, old_v, new_v), min(t * new_t // old_t, new_t - 1)] for v, t in mapping
            ]
        # A top-level out_hidden_size that is a multiple of the vision width is the
        # merged patch width (Muse-Glimmer: 4 x 64): keep the multiple.
        vision = getattr(cfg, "vision_config", None)
        if (
            isinstance(values.get("out_hidden_size"), int)
            and isinstance(vision_values, dict)
            and getattr(vision, "hidden_size", None)
            and cfg.out_hidden_size % vision.hidden_size == 0
        ):
            values["out_hidden_size"] = vision_values["hidden_size"] * (cfg.out_hidden_size // vision.hidden_size)
        # DPT reads one hidden state per backbone_out_indices entry, one per neck
        # width: keep a layer per entry, renumbered, and shrink the neck.
        outs = values.get("backbone_out_indices")
        if isinstance(outs, list) and isinstance(values.get("num_hidden_layers"), int) and not values.get("is_hybrid"):
            values["num_hidden_layers"] = len(outs)
            values["backbone_out_indices"] = list(range(len(outs)))
            if isinstance(values.get("neck_hidden_sizes"), list):
                values["neck_hidden_sizes"] = [16 * 2 ** min(i, 2) for i in range(len(values["neck_hidden_sizes"]))]
            if isinstance(values.get("fusion_hidden_size"), int):
                values["fusion_hidden_size"] = 64
        if isinstance(values.get("intermediate_size"), list):
            values["intermediate_size"] = [min(size, 128) for size in values["intermediate_size"]]
        if values.get("upsample_groups") and values.get("hidden_size"):
            from math import gcd
            values["upsample_groups"] = gcd(values["upsample_groups"], values["hidden_size"])
        # These models need projection_dim == the text width (OWL's class head contracts
        # text query embeds against it); CLIP's projection is its own joint space.
        if values.get("model_type") in ("paligemma", "owlv2", "owlvit"):
            values["projection_dim"] = values["text_config"]["hidden_size"]
        # Voxtral's projector stacks intermediate_size / hidden_size consecutive audio
        # frames into one token (reshape(-1, intermediate_size)): keep that factor.
        if values.get("model_type") == "voxtral" and isinstance(values.get("audio_config"), dict):
            audio = values["audio_config"]
            audio["intermediate_size"] = audio["hidden_size"] * (cfg.audio_config.intermediate_size // cfg.audio_config.hidden_size)
        # Mllama's projector reads the vision tower's last state concatenated with
        # each intermediate one: vision_output_dim = hidden_size x (indices + 1).
        old_vision = getattr(cfg, "vision_config", None)
        if (
            isinstance(vision_values, dict)
            and isinstance(vision_values.get("vision_output_dim"), int)
            and isinstance(vision_values.get("intermediate_layers_indices"), list)
            and old_vision.vision_output_dim
            == old_vision.hidden_size * (len(old_vision.intermediate_layers_indices) + 1)
        ):
            vision_values["vision_output_dim"] = vision_values["hidden_size"] * (
                len(vision_values["intermediate_layers_indices"]) + 1
            )
        # Granite Speech's Q-Former projector cross-attends to the audio encoder's
        # hidden states: its encoder_hidden_size is the encoder's hidden_dim.
        projector, encoder = values.get("projector_config"), values.get("encoder_config")
        if (
            isinstance(projector, dict)
            and isinstance(encoder, dict)
            and "encoder_hidden_size" in projector
            and getattr(cfg.encoder_config, "hidden_dim", None) == cfg.projector_config.encoder_hidden_size
        ):
            projector["encoder_hidden_size"] = encoder["hidden_dim"]
        # BLT's local encoder/decoder name the global transformer's width
        # (hidden_size_global), and the checkpoint sizes the global transformer to
        # the encoder's cross_attn_k patch outputs so no extra projection is built.
        if values.get("model_type") == "blt":
            glob = values["global_config"]
            if cfg.global_config.hidden_size == cfg.encoder_config.hidden_size * cfg.cross_attn_k:
                glob["hidden_size"] = values["encoder_config"]["hidden_size"] * values["cross_attn_k"]
                if glob.get("head_dim"):  # attention splits hidden_size; rotary reads head_dim
                    glob["head_dim"] = glob["hidden_size"] // glob["num_attention_heads"]
            glob["encoder_cross_output_size"] = None  # recomputed from the widths on load
            for key in ("encoder_config", "decoder_config"):
                values[key]["hidden_size_global"] = glob["hidden_size"]
        # A sub-config projecting into the language model (Janus's vision aligner,
        # Qwen3-ASR's audio output_dim, Llama 4's vision projector) must keep
        # projecting to its resized hidden size.
        text = getattr(cfg, "text_config", None)
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            ratio = getattr(sub, "pixel_shuffle_ratio", None)
            if (
                ratio
                and isinstance(values.get(key), dict)
                and sub.intermediate_size == sub.hidden_size / ratio**2
            ):
                # Llama 4's pixel shuffle turns hidden_size channels into
                # hidden_size / ratio**2, which its adapter MLP reads as intermediate_size.
                values[key]["intermediate_size"] = int(values[key]["hidden_size"] / ratio**2)
            for field in ("projection_dim", "output_dim", "vision_output_dim", "projector_input_dim", "projector_output_dim"):
                if (
                    key != "text_config"
                    and isinstance(values.get(key), dict)
                    and getattr(sub, field, None) is not None
                    and getattr(sub, field) == getattr(text, "hidden_size", None)
                ):
                    # BART-style text configs (Florence-2) name their width d_model.
                    text_values = values["text_config"]
                    values[key][field] = text_values.get("hidden_size") or text_values["d_model"]
        # X-CLIP's video transformer (mit_hidden_size) runs in the projection space.
        for key, sub in subs_of(cfg).items():
            if not isinstance(values.get(key), dict):
                continue
            if getattr(sub, "mit_hidden_size", None) == getattr(cfg, "projection_dim", -1) and values.get("projection_dim"):
                values[key]["mit_hidden_size"] = values["projection_dim"]
                values[key]["mit_intermediate_size"] = 4 * values["projection_dim"]
            # SAM's vision neck outputs the prompt encoder's width.
            prompt = values.get("prompt_encoder_config")
            if getattr(sub, "output_channels", None) and isinstance(prompt, dict) and sub.output_channels == cfg.prompt_encoder_config.hidden_size:
                values[key]["output_channels"] = prompt["hidden_size"]
        # Kimi K2.5's projector reads the vision tower's width (projection_hidden_size).
        vision_values = values.get("vision_config")
        if isinstance(vision_values, dict) and getattr(cfg, "vision_config", None) is not None and isinstance(values.get("projection_hidden_size"), int):
            if cfg.to_dict()["projection_hidden_size"] == cfg.vision_config.to_dict().get("hidden_size"):
                values["projection_hidden_size"] = vision_values["hidden_size"]
        # CSM's depth decoder projects the backbone's hidden states (backbone_hidden_size).
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if (
                isinstance(values.get(key), dict)
                and getattr(sub, "backbone_hidden_size", None) == getattr(cfg, "hidden_size", None)
                and values.get("hidden_size")
            ):
                values[key]["backbone_hidden_size"] = values["hidden_size"]
        # Depth Anything's neck reads one backbone stage per width: keep that many stages.
        backbone = values.get("backbone_config")
        outs = getattr(getattr(cfg, "backbone_config", None), "out_indices", None)
        necks = values.get("neck_hidden_sizes")
        if (
            isinstance(backbone, dict)
            and isinstance(outs, (list, tuple))
            and isinstance(necks, list)
            and len(outs) == len(necks) > 1
            and isinstance(backbone.get("num_hidden_layers"), int)
            and not isinstance(backbone.get("depths"), list)
        ):
            backbone["num_hidden_layers"] = max(backbone["num_hidden_layers"], len(outs))
            backbone["out_indices"] = list(range(backbone["num_hidden_layers"] - len(outs) + 1, backbone["num_hidden_layers"] + 1))
            backbone["out_features"] = [f"stage{i}" for i in backbone["out_indices"]]
        # A sub-config's channel list that ends at a sibling's width feeds that sibling
        # (Unlimited-OCR's SAM downsample_channels [512, 1024] are the CLIP encoder's
        # patch embeddings): scale it to the sibling's resized width.
        subs = {key: getattr(cfg, key, None) for key in getattr(cfg, "sub_configs", {}) or {}}
        for key, sub in subs.items():
            channels = getattr(sub, "downsample_channels", None)
            if not (isinstance(channels, (list, tuple)) and channels and isinstance(values.get(key), dict)):
                continue
            for other, sibling in subs.items():
                width = getattr(sibling, "hidden_size", None)
                if other != key and width == channels[-1] and isinstance(values.get(other), dict):
                    resized = values[other]["hidden_size"]
                    values[key]["downsample_channels"] = [c * resized // width for c in channels]
        # Multimodal RoPE partitions must still cover half the resized head.
        rope = values.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("mrope_section"):
            sections = rope["mrope_section"]
            # GLM-4.1V splits only the rotated part (partial_rotary_factor) into sections, exactly;
            # Qwen3.5 interleaves them and keeps covering half the head.
            partial = 1.0 if rope.get("mrope_interleaved") else rope.get("partial_rotary_factor", values.get("partial_rotary_factor", 1.0))
            half_head = int(values.get("head_dim", values["hidden_size"] // values["num_attention_heads"]) * partial) // 2
            scaled = [size * half_head / sum(sections) for size in sections]
            kept = [int(size) for size in scaled]
            for i in sorted(range(len(kept)), key=lambda i: scaled[i] - kept[i], reverse=True)[:half_head - sum(kept)]:
                kept[i] += 1
            rope["mrope_section"] = kept
        # LongRoPE keeps one factor per rotary frequency; resample to the resized head.
        if isinstance(rope, dict) and isinstance(rope.get("short_factor"), list):
            # MLA models (MiniCPM3, DeepSeek) rotate only qk_rope_head_dim channels.
            head = (
                values.get("qk_rope_head_dim")
                or values.get("head_dim")
                or values["hidden_size"] // values["num_attention_heads"]
            )
            partial = rope.get("partial_rotary_factor", values.get("partial_rotary_factor", 1.0))
            half = int(head * partial) // 2
            for key in ("short_factor", "long_factor"):
                factors = rope.get(key)
                if isinstance(factors, list) and len(factors) != half:
                    rope[key] = [factors[i * len(factors) // half] for i in range(half)]
        try:
            shrunk = type(cfg).from_dict(values)
        except Exception:
            # Only for a config whose own validation refuses the values above: a config that
            # validates shrinks exactly as before, so published fixtures rebuild unchanged.
            repair(cfg, values)
            shrunk = type(cfg).from_dict(values)
        depth = getattr(cfg, "num_hidden_layers", None)
        if through is None and isinstance(depth, int) and isinstance(values.get("num_hidden_layers"), int) and values["num_hidden_layers"] < depth:
            # Kinds the config derives from the layer index (DeepSeek-V4's hash-routed
            # layers, GLM-5.3's indexer every few MoE layers, Llama 4's moe_layers) are
            # lost by re-indexing: build both stacks on the meta device and, if a kind
            # is missing, keep the prefix up to each kind's first layer.
            from transformers_mic.shrink_view import kind_prefix

            prefix = kind_prefix(cfg, shrunk)
            if prefix and prefix <= MAX_PREFIX:
                return rebuilt(cfg, through=prefix)
        return shrunk

    return rebuilt(config)


def primary_head(config):
    from transformers.models.auto import modeling_auto

    others = sorted(n for n in dir(modeling_auto) if n.endswith("_MAPPING_NAMES") and n not in HEADS)
    for name in (*HEADS, *others):
        mapping = getattr(modeling_auto, name, {})
        if not isinstance(mapping, dict):
            continue
        if config.model_type in mapping:
            classes = mapping[config.model_type]
            name_ = classes[0] if isinstance(classes, (list, tuple)) else classes
            if not name_.endswith("Config"):  # some *_MAPPING_NAMES map to configs
                return name, name_
    return None, None


def head_config(model_class, config):
    """The config the head is built from: a text-only head of a composite model
    (Qwen3.5's causal LM) takes its text config, as AutoModel would pass it."""
    expected = getattr(model_class, "config_class", None)
    if expected is None or isinstance(config, expected):
        return config
    text = config.get_text_config()
    return text if isinstance(text, expected) else config


def synthetic_inputs(model, config):
    """Inputs for main_input_name and its usual companions, sized from the config."""
    import torch

    def find(key, default=None):
        for cfg in sub_configs(config):
            value = getattr(cfg, key, None)
            if value is not None:
                return value
        return default

    name = model.main_input_name
    batch, length = 2, 16
    inputs = {}
    if name == "input_ids":
        vocab = find("vocab_size", 1000)
        inputs["input_ids"] = torch.randint(3, min(vocab, 1000), (batch, length))
        inputs["attention_mask"] = torch.ones(batch, length, dtype=torch.long)
        if config.is_encoder_decoder:
            inputs["decoder_input_ids"] = inputs["input_ids"].clone()
    elif name == "pixel_values":
        size = find("image_size", 224)
        size = size if isinstance(size, int) else (size.get("height") if isinstance(size, dict) else size[0])
        inputs["pixel_values"] = torch.randn(batch, find("num_channels", 3), size, size)
    elif name == "input_features":
        inputs["input_features"] = torch.randn(batch, find("num_mel_bins", 80), 3000)
        if config.is_encoder_decoder:
            inputs["decoder_input_ids"] = torch.randint(3, 100, (batch, 4))
    elif name == "input_values":
        inputs["input_values"] = torch.randn(batch, 16000)
    else:
        raise NotImplementedError(f"no synthetic inputs for {name}")
    # Dual encoders (CLIP-like) take text and pixels together; models that splice
    # image tokens into the text need a processor and get text only here.
    accepted = inspect.signature(model.forward).parameters
    spliced = any(find(k) is not None for k in ("image_token_id", "image_token_index"))
    vision = getattr(config, "vision_config", None)
    if name == "input_ids" and "pixel_values" in accepted and vision is not None and not spliced:
        size = getattr(vision, "image_size", 224)
        size = size if isinstance(size, int) else (size.get("height") if isinstance(size, dict) else size[0])
        inputs["pixel_values"] = torch.randn(batch, getattr(vision, "num_channels", 3), size, size)
    # Text-audio dual encoders (CLAP) take mel features beside the text, as the
    # feature extractor shapes them: (batch, channels, frames, mel bins).
    audio = getattr(config, "audio_config", None)
    if name == "input_ids" and "input_features" in accepted and audio is not None and getattr(audio, "spec_size", None):
        inputs["input_features"] = torch.randn(batch, 1, 1001, audio.num_mel_bins)
        if "is_longer" in accepted:  # fusion models flag clips longer than the window
            inputs["is_longer"] = torch.zeros(batch, 1, dtype=torch.bool)
    return inputs


def image_size(size, default=224):
    """(height, width) from an int, a {"height", "width"} dict or a [height, width] list."""
    if isinstance(size, dict):
        return size.get("height", default), size.get("width", default)
    if isinstance(size, (list, tuple)):
        return (size[0], size[-1]) if size else (default, default)
    return (size, size) if isinstance(size, int) else (default, default)


def input_layouts(model, config):
    """Candidate inputs, the usual layout first; the sweep runs the first one the model accepts.

    The config alone does not say everything synthetic_inputs needs: whether audio features are
    (batch, mel, frames) like Whisper's or (batch, frames, mel) like Parakeet's, whether a codec
    takes (batch, samples) or (batch, channels, samples), that a video model takes frames, or that
    a captioning head (BLIP) or an encoder-decoder with a pixel or waveform input needs text too.
    """
    import torch

    def find(key, default=None):
        for cfg in sub_configs(config):
            value = getattr(cfg, key, None)
            if value is not None:
                return value
        return default

    batch = 2
    try:
        first = synthetic_inputs(model, config)
    except NotImplementedError:
        first = {}
    accepted = inspect.signature(model.forward).parameters
    names = model.main_input_name if isinstance(model.main_input_name, list) else [model.main_input_name]
    full = dict(first)
    pretrained = find("pretrained_cfg") or {}
    height, width = image_size(pretrained.get("input_size", [None])[1:] or find("image_size", 224))
    frames = find("num_frames") or find("frames_per_clip")
    channels = find("num_channels", 3)
    for name in names:
        if name in ("pixel_values", "pixel_values_videos"):
            shape = (batch, channels, height, width)
            video = name == "pixel_values_videos" or (frames and name == names[0])
            full[name] = torch.randn(batch, frames, *shape[1:]) if video and frames else torch.randn(*shape)
        elif name == "flattened_patches":  # Pix2Struct: row and column ids, then the patch pixels
            patches = torch.randn(batch, 64, find("patch_embed_hidden_size", 768))
            rows = torch.arange(64).repeat(batch, 1)
            full[name] = torch.cat([(rows // 8 + 1)[..., None], (rows % 8 + 1)[..., None], patches], dim=-1).float()
        elif name == "task_inputs":  # OneFormer's tokenized task prompt
            full[name] = torch.randint(3, 1000, (batch, find("task_seq_len", 77)))
        elif name == "input_values" and find("audio_channels"):
            full[name] = torch.randn(batch, find("audio_channels"), 16000)
        elif name == "input_values" and find("num_mel_bins") and find("max_length"):
            full[name] = torch.randn(batch, find("max_length"), find("num_mel_bins"))  # AST's spectrogram
    pixels = full.get("pixel_values")
    if frames and pixels is not None and pixels.dim() == 4:  # X-CLIP: video beside the text
        full["pixel_values"] = pixels[:, None].expand(-1, frames, -1, -1, -1).contiguous()
    audio = getattr(config, "audio_config", None)
    if "input_features" in accepted and "input_features" not in full and getattr(audio, "num_mel_bins", None):
        # Voxtral Realtime streams one text token per audio_length_per_tok mel frames.
        per_token = getattr(config, "audio_length_per_tok", None)
        length = full["input_ids"].shape[1] * per_token if per_token and "input_ids" in full else 3000
        full["input_features"] = torch.randn(batch, audio.num_mel_bins, length)
    text = getattr(config, "text_config", None) or getattr(config, "decoder", None)
    if "input_ids" in accepted and "input_ids" not in full and text is not None:
        full["input_ids"] = torch.randint(3, min(find("vocab_size", 1000), 1000), (batch, 16))
    if config.is_encoder_decoder and "decoder_input_ids" in accepted and "decoder_input_ids" not in full:
        full["decoder_input_ids"] = torch.randint(3, 100, (batch, 4))
    layouts = [first] if first else []
    for inputs in (full,):
        if inputs is not first and inputs:
            layouts.append(inputs)
        features = inputs.get("input_features")
        if features is not None and features.dim() == 3:
            layouts.append({**inputs, "input_features": features.transpose(1, 2).contiguous()})
        values = inputs.get("input_values")
        if values is not None and values.dim() == 2:
            layouts.append({**inputs, "input_values": values[:, None]})
    if not layouts:
        raise NotImplementedError(f"no synthetic inputs for {model.main_input_name}")
    return layouts


def run_one(model_type):
    import torch
    import transformers
    from transformers import AutoConfig

    transformers.logging.set_verbosity_error()
    torch.manual_seed(0)
    source = json.loads(SOURCES.read_text())[model_type]
    row = {"model_type": model_type, "repo": source.get("repo"), "revision": source.get("revision")}
    stage = "load_config"
    try:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(source["repo"], "config.json", revision=source["revision"])
        config = AutoConfig.from_pretrained(str(Path(path).parent))
        row["config_model_type"] = config.model_type
        if config.model_type != model_type:
            # The doc checkpoint is a composite model; use its nested config of this type.
            nested = [c for c in sub_configs(config) if c.model_type == model_type]
            if not nested:
                row["stage"] = "sub_config"
                return row
            config = nested[0]
            row["from_parent"] = True
        stage = "shrink"
        small = shrink(config)
        stage = "build"
        head, class_name = primary_head(small)
        row["head"] = head
        if class_name is None:
            # Only reachable inside a composite model, not a usage path of its own.
            row["stage"] = "component"
            return row
        try:
            small = head_config(getattr(transformers, class_name), small)
            model = getattr(transformers, class_name)._from_config(small, dtype=torch.float32)
        except ValueError as exc:
            # A head the checkpoint does not have (w2v-BERT 2.0 is pretraining-only: no vocabulary
            # for a CTC head) or cannot pick (MusicGen's two text configs): measure the base model,
            # the class such a contract uses.
            from transformers.models.auto import modeling_auto

            base = modeling_auto.MODEL_MAPPING_NAMES.get(small.model_type)
            if not base or base == class_name:
                raise
            row["head"], row["head_error"], class_name = "MODEL_MAPPING_NAMES", str(exc)[:200], base
            model = getattr(transformers, class_name)._from_config(small, dtype=torch.float32)
        # A sub-config's own dtype (T5Gemma 2's bfloat16 towers) must not mix into the float32 build.
        model = model.float()
        row["parameters"] = sum(p.numel() for p in model.parameters())
        row["layers"] = getattr(small, "num_hidden_layers", None)
        stage = "inputs"
        layouts = input_layouts(model, small)
        inputs = layouts[0]
        row["main_input"] = model.main_input_name
        row["inputs"] = sorted(inputs)
        # Multimodal models get text only here; their image/audio path is not exercised.
        row["text_only_multimodal"] = "input_ids" in inputs and any(
            getattr(small, k, None) is not None for k in ("vision_config", "audio_config")
        )
        stage = "forward"
        start = time.perf_counter()
        first_error = None
        for n, candidate in enumerate(layouts):
            try:
                output = model(**candidate)
                break
            except Exception as exc:
                first_error = first_error or exc
                if n == len(layouts) - 1:
                    raise first_error from None
        if candidate is not inputs:
            row["inputs"], row["layout"] = sorted(candidate), n
        row["forward_seconds"] = time.perf_counter() - start
        tensors = [v for v in output.values() if isinstance(v, torch.Tensor) and v.is_floating_point()]
        row["finite"] = all(bool(torch.isfinite(t).all()) for t in tensors)
        stage = "backward"
        target = output.get("logits")
        target = target if target is not None else tensors[0]
        target.float().pow(2).mean().backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        row["finite"] = row["finite"] and bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
        row["stage"] = "ok"
    except Exception as exc:
        row["stage"] = stage
        row["error"] = f"{type(exc).__name__}: {exc}"[:400]
    return row


def run(args):
    sources = json.loads(SOURCES.read_text())
    types = args.types or sorted(t for t, s in sources.items() if s["status"] == "ok")

    def one(model_type):
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "one", model_type],
                capture_output=True, text=True, timeout=args.timeout,
            )
            line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else None
            if proc.returncode == 0 and line:
                return json.loads(line)
            return {"model_type": model_type, "stage": "crash", "error": proc.stderr[-400:]}
        except subprocess.TimeoutExpired:
            return {"model_type": model_type, "stage": "timeout"}

    # A run over named types updates those rows and keeps the rest.
    rows = json.loads(RESULTS.read_text())["rows"] if args.types and RESULTS.exists() else {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for row in pool.map(one, types):
            rows[row["model_type"]] = row
            print(f"{row['model_type']:40} {row['stage']:12} {row.get('error', '')[:100]}", flush=True)
    for model_type, source in sources.items():
        if model_type not in rows and not args.types:
            rows[model_type] = {"model_type": model_type, "stage": source["status"]}
    report = {"transformers": __import__("transformers").__version__, "rows": rows}
    RESULTS.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    summary(args)


def summary(args):
    rows = json.loads(RESULTS.read_text())["rows"]
    counts = {}
    for row in rows.values():
        counts[row["stage"]] = counts.get(row["stage"], 0) + 1
    print(json.dumps(dict(sorted(counts.items(), key=lambda kv: -kv[1])), indent=1))
    print("49103:", {t: rows.get(t, {}).get("stage") for t in FAST_49103})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prefetch", "run", "one", "summary"))
    parser.add_argument("types", nargs="*")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.command == "one":
        os.environ["HF_HUB_OFFLINE"] = "1"
        import torch

        torch.set_num_threads(1)
        print(json.dumps(run_one(args.types[0]), default=str))
    else:
        {"prefetch": prefetch, "run": run, "summary": summary}[args.command](args)


if __name__ == "__main__":
    main()
