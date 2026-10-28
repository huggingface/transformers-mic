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
    "moe_intermediate_size": 32, "shared_intermediate_size": 128,
    "shared_expert_intermediate_size": 128, "projection_dim": 32,
    "kv_lora_rank": 16, "qk_nope_head_dim": 16, "qk_rope_head_dim": 8, "v_head_dim": 16,
    "mamba_d_state": 16, "state_size": 16, "ssm_state_size": 16, "mamba_n_heads": 8,
    "mamba_d_head": "auto", "mamba_chunk_size": 8, "chunk_size": 8,
    # GPT-2, XLM, DistilBERT, and others name the same widths differently.
    "n_embd": 64, "n_inner": 128, "n_head": 4, "emb_dim": 64, "dim": 64, "hidden_dim": 128,
    "n_heads": 4,
}
EXPERTS = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
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

    def fetch(item):
        model_type, repo = item
        row = {"doc_checkpoint": repo}
        if repo is None:
            row["doc_status"] = "no_doc_checkpoint"
        else:
            try:
                row.update(repo=repo, revision=try_repo(model_type, repo), source="doc", status="ok")
                row["doc_status"] = "ok"
                return model_type, row
            except Exception as exc:
                row["doc_status"] = (
                    "wrong_model_type" if isinstance(exc, ValueError) else type(exc).__name__
                )
        # Fall back to the most downloaded ungated Transformers model with this type.
        try:
            candidates = api.list_models(
                filter=[model_type, "transformers"], sort="downloads", limit=10, expand=["gated"]
            )
            for candidate in candidates:
                if candidate.gated:
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


def layer_pattern(config):
    """Indices to keep: the first layer of each distinct kind (see layer_patterns.py)."""
    sys.path.insert(0, str(HERE))
    from layer_patterns import pattern_indices

    return pattern_indices(config)


def shrink(config):
    """Apply the generic rule in place, innermost configs first, then rebuild each."""
    from transformers import PretrainedConfig

    def rebuilt(cfg):
        values = cfg.to_dict()
        # A slim fixture holds dense random weights, whatever the checkpoint's format.
        values.pop("quantization_config", None)
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if isinstance(sub, PretrainedConfig):
                values[key] = rebuilt(sub).to_dict()
        for key, value in WIDTHS.items():
            if key in values and values[key] is not None and not isinstance(values[key], (list, dict)):
                values[key] = value
        if "num_key_value_heads" in values and values["num_key_value_heads"]:
            heads = cfg.num_attention_heads if hasattr(cfg, "num_attention_heads") else None
            values["num_key_value_heads"] = 4 if heads and values["num_key_value_heads"] >= heads else 2
        if values.get("q_lora_rank"):
            values["q_lora_rank"] = 16
        for key in EXPERTS:
            if isinstance(values.get(key), int) and values[key] > 4:
                values[key] = 4
        if isinstance(values.get("num_experts_per_tok"), int):
            values["num_experts_per_tok"] = min(values["num_experts_per_tok"], 2)
        for key in ("n_group", "topk_group"):
            if isinstance(values.get(key), int):
                values[key] = 1
        for key in DEPTHS:
            if isinstance(values.get(key), int):
                values[key] = min(values[key], 2)
        if isinstance(values.get("hidden_sizes"), list):
            values["hidden_sizes"] = [16 * 2**i for i in range(len(values["hidden_sizes"]))]
        if isinstance(values.get("depths"), list):
            values["depths"] = [1] * len(values["depths"])
        # Per-stage head counts must divide the shrunk per-stage widths (PVT's
        # hidden_sizes, Swin's embed_dim doubling per stage): keep the largest
        # head count, no more than the original, that does.
        stages = values.get("hidden_sizes")
        if not isinstance(stages, list) and isinstance(values.get("embed_dim"), int) and isinstance(values.get("depths"), list):
            stages = [values["embed_dim"] * 2**i for i in range(len(values["depths"]))]
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
        if isinstance(values.get("num_hidden_layers"), int):
            kept = layer_pattern(cfg) if hasattr(cfg, "num_hidden_layers") else [0, 1]
            kept = kept if len(kept) > 1 else list(range(min(2, values["num_hidden_layers"])))
            for key, value in cfg.to_dict().items():
                if isinstance(value, list) and len(value) == values["num_hidden_layers"]:
                    values[key] = [value[i] for i in kept if i < len(value)]
            if getattr(cfg, "num_kv_shared_layers", None) is not None:
                values["num_kv_shared_layers"] = sum(i >= cfg.num_hidden_layers - cfg.num_kv_shared_layers for i in kept)
            if getattr(cfg, "first_k_dense_replace", None) is not None:
                values["first_k_dense_replace"] = sum(i < cfg.first_k_dense_replace for i in kept)
            per_layer = values.get("per_layer_config")
            if isinstance(per_layer, dict):
                values["per_layer_config"] = {
                    str(n): per_layer[k] for n, i in enumerate(kept)
                    for k in (str(i), i) if k in per_layer
                }
            if getattr(cfg, "mlp_only_layers", None):
                values["mlp_only_layers"] = [n for n, i in enumerate(kept) if i in cfg.mlp_only_layers]
            values["num_hidden_layers"] = len(kept)
        if isinstance(values.get("intermediate_size"), list):
            values["intermediate_size"] = [min(size, 128) for size in values["intermediate_size"]]
        if values.get("upsample_groups") and values.get("hidden_size"):
            from math import gcd
            values["upsample_groups"] = gcd(values["upsample_groups"], values["hidden_size"])
        if values.get("model_type") == "paligemma":
            values["projection_dim"] = values["text_config"]["hidden_size"]
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
        # A sub-config projecting into the language model (Janus's vision aligner)
        # must keep projecting to its resized hidden size.
        text = getattr(cfg, "text_config", None)
        for key in getattr(cfg, "sub_configs", {}) or {}:
            sub = getattr(cfg, key, None)
            if (
                key != "text_config"
                and isinstance(values.get(key), dict)
                and getattr(sub, "projection_dim", None) is not None
                and sub.projection_dim == getattr(text, "hidden_size", None)
            ):
                values[key]["projection_dim"] = values["text_config"]["hidden_size"]
        # Multimodal RoPE partitions must still cover half the resized head.
        rope = values.get("rope_parameters")
        if isinstance(rope, dict) and rope.get("mrope_section"):
            sections = rope["mrope_section"]
            half_head = values.get("head_dim", values["hidden_size"] // values["num_attention_heads"]) // 2
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
        return type(cfg).from_dict(values)

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
    return inputs


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
        model = getattr(transformers, class_name)._from_config(small, dtype=torch.float32)
        row["parameters"] = sum(p.numel() for p in model.parameters())
        row["layers"] = getattr(small, "num_hidden_layers", None)
        stage = "inputs"
        inputs = synthetic_inputs(model, small)
        row["main_input"] = model.main_input_name
        row["inputs"] = sorted(inputs)
        # Multimodal models get text only here; their image/audio path is not exercised.
        row["text_only_multimodal"] = "input_ids" in inputs and any(
            getattr(small, k, None) is not None for k in ("vision_config", "audio_config")
        )
        stage = "forward"
        start = time.perf_counter()
        output = model(**inputs)
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
