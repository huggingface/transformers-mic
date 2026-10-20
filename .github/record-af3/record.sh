#!/usr/bin/env bash
# Record AudioFlamingo3 CPU baselines the way PR CI runs the fleet, then check them.
set -uo pipefail
OUT=${OUT:?}
LOCK="$GITHUB_WORKSPACE/.github/record-af3/lock.json"
mkdir -p "$OUT/expectations"
{
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "cores: $(nproc)"
  echo "flags: $(grep -o -w -E 'avx2|avx512f|avx512_bf16|amx_bf16|amx_tile' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  python -c 'import torch, transformers; print("torch", torch.__version__, "threads", torch.get_num_threads(), "transformers", transformers.__version__, transformers.__file__)'
} | tee "$OUT/host.txt"
uv pip freeze > "$OUT/pip-freeze.txt" 2>/dev/null || python -m pip freeze > "$OUT/pip-freeze.txt"
for v in bfloat16 default eager; do
  run-model-contracts --fixtures-only --lock "$LOCK" --expectations "$OUT/expectations" \
    --variant "$v" --record --output "$OUT/record-$v.jsonl" 2> "$OUT/record-$v.log"
  tail -1 "$OUT/record-$v.log"
done
mkdir -p "$OUT/candidates"
cp -R "$OUT/expectations/." "$OUT/candidates/"
find "$OUT/expectations" -name '*.candidate.json' | while read -r f; do mv "$f" "${f%.candidate.json}.json"; done
for v in bfloat16 default eager; do
  run-model-contracts --fixtures-only --lock "$LOCK" --expectations "$OUT/expectations" \
    --variant "$v" --output "$OUT/check-$v.jsonl" 2> "$OUT/check-$v.log"
  tail -1 "$OUT/check-$v.log"
done
find "$OUT/candidates" -type f
