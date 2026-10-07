#!/usr/bin/env bash
# Record and check cpu-x86_64 baselines with the fleet lock's pinned
# environment; shared by both jobs of cpu-baselines.yml.
#
#   PY=python3 SHA=<transformers sha> VARIANTS="default eager" OUT=out .github/scripts/cpu_baselines.sh
#
# Writes to $OUT: pip-freeze.txt, host.txt, record-*/check-*/golden-* .jsonl and
# .log, and candidates/ (the candidate baselines, at their repository paths).
set -euo pipefail

LOCK=example/transformers/tests/contracts/fleet.lock
EXPECTATIONS=example/transformers/tests/contracts/expectations
mkdir -p "$OUT"
pin() { "$PY" -c "import json, sys; print(json.load(open('$LOCK'))['environment'][sys.argv[1]])" "$1"; }
status() { "$PY" -c 'import json, sys
for line in sys.stdin:
    r = json.loads(line); print(r["target"], r["variant"] + ":", r["status"], *r.get("failures", [])[:1], flush=True)'; }

echo "::group::Install"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet "torch==$(pin torch)" --index-url https://download.pytorch.org/whl/cpu
rm -rf /tmp/transformers && git init --quiet /tmp/transformers
git -C /tmp/transformers fetch --quiet --depth 1 https://github.com/huggingface/transformers "$SHA"
git -C /tmp/transformers checkout --quiet FETCH_HEAD
"$PY" -m pip uninstall --quiet -y transformers || true
"$PY" -m pip install --quiet /tmp/transformers
"$PY" -m pip install --quiet git+https://github.com/huggingface/doc-builder jsonschema accelerate Pillow
"$PY" -m pip install --quiet "tokenizers==$(pin tokenizers)" "numpy==$(pin numpy)"
"$PY" -m pip freeze > "$OUT/pip-freeze.txt"
echo "::endgroup::"
{
  echo "cpu: $(lscpu | sed -n 's/^Model name: *//p')"
  echo "cores: $(nproc)"
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
