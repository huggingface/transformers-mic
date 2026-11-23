"""The dtype each tensor of a checkpoint is stored in, read from safetensors headers.

Real checkpoints are not stored in one dtype: MoE routers keep `mlp.gate.weight`
or `e_score_correction_bias` in float32 beside bfloat16 weights, and quantized
experts are F8. A fixture saved in one dtype hides changes that depend on it (a
router kept in float32 on load is a lossless upcast of a bfloat16 fixture, and a
false difference for a float32 fixture of a bfloat16 router). Fixtures keep each
tensor's stored float dtype instead, matched by name pattern so a shrunk fixture
(fewer layers, fewer experts) still maps onto the checkpoint.

Only headers are read: the index plus each shard's 8-byte length and header JSON,
by HTTP range request or from local files, cached per revision.
"""

import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Stored float dtypes a fixture keeps; anything else (F8_*, I8, U8, packed
# quantized blocks, F64) is storage the dense fixture does not mirror.
FLOAT = {"F32": "float32", "BF16": "bfloat16", "F16": "float16"}


def pattern(name):
    """A tensor name with its numeric path components wildcarded: layers.45.mlp -> layers.*.mlp."""
    return re.sub(r"(?<=\.)\d+(?=\.|$)|^\d+(?=\.)", "*", name)


def pattern_dtypes(stored):
    """{pattern: torch dtype name} for patterns whose source tensors are unanimous in a float dtype.

    `stored` maps checkpoint tensor names to safetensors dtypes ("BF16", "F8_E4M3").
    A pattern whose tensors disagree (o_proj BF16 in some layers, F8 in others), or
    that is stored quantized or as integers, has no entry: the fixture default applies.
    """
    seen = collections.defaultdict(set)
    for name, dtype in stored.items():
        seen[pattern(name)].add(dtype)
    return {
        key: FLOAT[next(iter(dtypes))] for key, dtypes in seen.items() if len(dtypes) == 1 and dtypes <= FLOAT.keys()
    }


def overrides(stored, default):
    """The patterns whose stored float dtype differs from the fixture default: what a fixture key records."""
    return {key: dtype for key, dtype in sorted(pattern_dtypes(stored).items()) if dtype != default}


def target_dtypes(names, stored, default):
    """{fixture tensor name: torch dtype name} each floating fixture tensor is saved in.

    Names whose pattern the checkpoint does not have (renamed or fused by a
    conversion) take the default.
    """
    patterns = pattern_dtypes(stored)
    return {name: patterns.get(pattern(name), default) for name in names}


def config_dtype(config):
    """The dtype a config names: its own, else its sub-configs' (text_config first, else unanimous)."""
    own = config.get("dtype") or config.get("torch_dtype")
    if own:
        return own
    nested = {}
    for key, value in config.items():
        if isinstance(value, dict) and ("model_type" in value or key.endswith("_config")):
            found = config_dtype(value)
            if found:
                nested[key] = found
    if "text_config" in nested:
        return nested["text_config"]
    if len(set(nested.values())) == 1:
        return next(iter(nested.values()))
    return None


def _header(path):
    with open(path, "rb") as f:
        length = int.from_bytes(f.read(8), "little")
        return json.loads(f.read(length))


def _local_dtypes(source, files):
    stored = {}
    for name in files:
        for tensor, info in _header(Path(source) / name).items():
            if tensor != "__metadata__":
                stored[tensor] = info["dtype"]
    return stored


def weight_files(source):
    """The safetensors files a checkpoint loads from, from its local config/index, or None if it has none."""
    from .inspection import explicit_weights

    source = Path(source)
    explicit = explicit_weights(source)
    index = source / (explicit if explicit and explicit.endswith(".index.json") else "model.safetensors.index.json")
    if explicit and not explicit.endswith(".index.json"):
        return [explicit]
    if index.exists():
        return sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return ["model.safetensors"]


def cache_path(repo, revision):
    from huggingface_hub.constants import HF_HUB_CACHE

    return Path(HF_HUB_CACHE).parent / "tmic-tensor-dtypes" / repo.replace("/", "--") / f"{revision}.json"


def checkpoint_dtypes(repo, revision, source=None):
    """{tensor name: safetensors dtype} of a pinned checkpoint, from headers only.

    Empty for a checkpoint with no safetensors file (a pytorch_model.bin source):
    its fixture keeps the default dtype. Cached per revision, since a commit's
    headers never change.
    """
    cached = cache_path(repo, revision) if re.fullmatch(r"[0-9a-f]{40}", revision or "") else None
    if cached and cached.exists():
        return json.loads(cached.read_text())
    from huggingface_hub import HfApi, snapshot_download

    if source is None:  # the config and indexes only: they name the shards
        source = snapshot_download(repo, revision=revision, allow_patterns=["config.json", "*.index.json"])
    files = weight_files(source)
    if all((Path(source) / name).exists() for name in files):
        stored = _local_dtypes(source, files)
    else:
        api = HfApi()
        if files == ["model.safetensors"] and not api.file_exists(repo, "model.safetensors", revision=revision):
            stored = {}
        else:
            with ThreadPoolExecutor(8) as pool:
                headers = list(
                    pool.map(
                        lambda name: api.parse_safetensors_file_metadata(repo, name, revision=revision, timeout=60),
                        files,
                    )
                )
            stored = {name: info.dtype for header in headers for name, info in header.tensors.items()}
    if cached:
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(stored, sort_keys=True))
    return stored
