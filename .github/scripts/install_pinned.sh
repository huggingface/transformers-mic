#!/usr/bin/env bash
# Install a Transformers commit and the fleet lock's pinned environment.
#
#   PY=python FETCH=<sha or pull/<n>/head> [SHA=<expected sha>] .github/scripts/install_pinned.sh
#
# torch, torchvision, and torchaudio come from PyTorch's CPU index; packages
# the lock does not pin are left as Transformers resolves them.
set -euo pipefail

LOCK=example/transformers/tests/contracts/fleet.lock
requirements() { "$PY" -c 'import json, sys
env = json.load(open(sys.argv[1]))["environment"]
torch = {"torch", "torchvision", "torchaudio"}
for name, version in env.items():
    if name != "python" and (name in torch) == (sys.argv[2] == "torch"):
        print(f"{name}=={version}")' "$LOCK" "$1"; }

"$PY" -m pip install --quiet --upgrade pip
mapfile -t torch_pins < <(requirements torch)
mapfile -t other_pins < <(requirements other)
"$PY" -m pip uninstall --quiet -y torch torchvision torchaudio transformers 2> /dev/null || true
"$PY" -m pip install --quiet "${torch_pins[@]}" --index-url https://download.pytorch.org/whl/cpu
rm -rf /tmp/transformers && git init --quiet /tmp/transformers
git -C /tmp/transformers fetch --quiet --depth 1 https://github.com/huggingface/transformers "$FETCH"
git -C /tmp/transformers checkout --quiet FETCH_HEAD
if [[ -n "${SHA:-}" ]]; then test "$(git -C /tmp/transformers rev-parse HEAD)" = "$SHA"; fi
"$PY" -m pip install --quiet /tmp/transformers
"$PY" -m pip install --quiet git+https://github.com/huggingface/doc-builder jsonschema
"$PY" -m pip install --quiet "${other_pins[@]}"
