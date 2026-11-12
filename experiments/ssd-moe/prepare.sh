#!/usr/bin/env bash
# Download the pinned Qwen3.6-35B-A3B files once into the shared runner cache and build the BF16 GGUF from the
# same safetensors, so every engine reads the same weights. Writes $ROOT/paths.json for the engine jobs.
set -euo pipefail
ROOT=${ROOT:-/mnt/cache/ssd-moe}
LLAMA_TAG=b11977
mkdir -p "$ROOT" out/prepare
python3 - "$ROOT" <<'PY'
import json, sys
from huggingface_hub import hf_hub_download, snapshot_download
root = sys.argv[1]
paths = {
    "model": "Qwen/Qwen3.6-35B-A3B",
    "model_revision": "345d467b117d1fd5eda9b879a5988de9ae5f47a7",
    "gguf_repo": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF",
    "gguf_revision": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
}
paths["safetensors"] = snapshot_download(
    paths["model"], revision=paths["model_revision"],
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "tokenizer*", "vocab*", "merges*"],
)
paths["gguf_q8_0"] = hf_hub_download(paths["gguf_repo"], "Qwen3.6-35B-A3B-Q8_0.gguf", revision=paths["gguf_revision"])
json.dump(paths, open(f"{root}/paths.partial.json", "w"), indent=2)
PY
BF16="$ROOT/Qwen3.6-35B-A3B-345d467-bf16-llamacpp-$LLAMA_TAG.gguf"
if [ ! -f "$BF16.done" ]; then
  rm -rf /tmp/llama.cpp
  git clone -q --depth 1 --branch "$LLAMA_TAG" https://github.com/ggml-org/llama.cpp /tmp/llama.cpp
  python3 -m pip install -q /tmp/llama.cpp/gguf-py
  SRC=$(python3 -c "import json; print(json.load(open('$ROOT/paths.partial.json'))['safetensors'])")
  python3 /tmp/llama.cpp/convert_hf_to_gguf.py "$SRC" --outtype bf16 --outfile "$BF16.tmp" 2>&1 | tail -40 | tee out/prepare/convert.log
  mv "$BF16.tmp" "$BF16" && touch "$BF16.done"
fi
python3 - "$ROOT" "$BF16" "$LLAMA_TAG" <<'PY'
import json, os, sys
root, bf16, tag = sys.argv[1:]
paths = json.load(open(f"{root}/paths.partial.json"))
paths["gguf_bf16"] = bf16
paths["gguf_bf16_converter"] = f"ggml-org/llama.cpp {tag} convert_hf_to_gguf.py --outtype bf16 from the safetensors above"
paths["bytes"] = {
    "safetensors": sum(os.path.getsize(os.path.join(paths["safetensors"], f)) for f in os.listdir(paths["safetensors"])
                       if f.endswith(".safetensors")),
    "gguf_q8_0": os.path.getsize(paths["gguf_q8_0"]),
    "gguf_bf16": os.path.getsize(bf16),
}
json.dump(paths, open(f"{root}/paths.json", "w"), indent=2)
print(json.dumps(paths, indent=2))
PY
cp "$ROOT/paths.json" out/prepare/
