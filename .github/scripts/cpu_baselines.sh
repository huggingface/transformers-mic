#!/usr/bin/env bash
# Record and check cpu-x86_64 baselines with the fleet lock's pinned
# environment; shared by both jobs of cpu-baselines.yml.
#
#   PY=python3 SHA=<transformers sha> VARIANTS="default eager" OUT=out .github/scripts/cpu_baselines.sh
#
# Writes to $OUT: pip-freeze.txt, host.txt, record-*/check-*/golden-* .jsonl and
# .log, and candidates/ (the candidate baselines, at their repository paths).
set -euo pipefail

EXPECTATIONS=example/transformers/tests/contracts/expectations
mkdir -p "$OUT"
status() { "$PY" -c 'import json, sys
for line in sys.stdin:
    r = json.loads(line); print(r["target"], r["variant"] + ":", r["status"], *r.get("failures", [])[:1], flush=True)'; }

echo "::group::Install"
FETCH="$SHA" .github/scripts/install_pinned.sh
"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
echo "::endgroup::"
{
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "cores: $(nproc)"
  # bfloat16 kernels differ with native support (Zen 4/5, Sapphire Rapids): baselines may too.
  echo "flags: $(grep -o -w -E 'avx2|avx512f|avx512_bf16|amx_bf16|amx_tile' /proc/cpuinfo | sort -u | tr '\n' ' ')"
  echo "python: $("$PY" --version)"
  free -g | sed -n 2p
  df -h / | sed -n 2p
} | tee "$OUT/host.txt"

read -ra variants <<< "$VARIANTS"
for variant in "${variants[@]}"; do
  # --isolate: one process per target, so no process holds two models.
  "$PY" example/run.py --framework transformers --variant "$variant" --record --isolate \
    --output "$OUT/record-$variant.jsonl" 2> "$OUT/record-$variant.log" | status || true
  tail -1 "$OUT/record-$variant.log"
done
mkdir -p "$OUT/candidates"
find "$EXPECTATIONS" -name '*.candidate.json' -path '*/cpu-x86_64/*' | while read -r f; do
  mkdir -p "$OUT/candidates/$(dirname "$f")"
  cp "$f" "$OUT/candidates/$f"
  mv "$f" "${f%.candidate.json}.json"  # approved here only, so the checks below have baselines
done
echo "candidates: $(find "$OUT/candidates" -name '*.json' | wc -l)"

# The checks a shadow run makes, against the new baselines, and discrete outputs vs Apple M5.
for variant in "${variants[@]}"; do
  "$PY" example/run.py --framework transformers --variant "$variant" --isolate \
    --output "$OUT/check-$variant.jsonl" > /dev/null 2> "$OUT/check-$variant.log" || true
  tail -1 "$OUT/check-$variant.log"
done
"$PY" example/run.py --framework transformers --golden cpu-arm64 --isolate \
  --output "$OUT/golden-cpu-arm64.jsonl" > /dev/null 2> "$OUT/golden-cpu-arm64.log" || true
tail -1 "$OUT/golden-cpu-arm64.log"
