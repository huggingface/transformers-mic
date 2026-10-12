#!/usr/bin/env bash
# Reviewed locally before committing; this job's approvals are temporary.
set -euo pipefail
for platform in cpu cuda; do
  if [[ "$platform" == cpu ]]; then export CUDA_VISIBLE_DEVICES=""; variants="default eager"; else export CUDA_VISIBLE_DEVICES=0; variants="default eager bfloat16"; fi
  for variant in $variants; do
    python3 example/run.py --framework transformers --contract WhisperSpeech --record --isolate \
      --variant "$variant" --output "out/record-$platform-$variant.jsonl" > "out/record-$platform-$variant.log" 2>&1
  done
done
python3 example/review_candidates.py --approve | tee out/review.txt
mkdir -p out/expectations
cp -r example/transformers/tests/contracts/expectations/WhisperSpeech out/expectations/
for platform in cpu cuda; do
  if [[ "$platform" == cpu ]]; then export CUDA_VISIBLE_DEVICES=""; variants="default eager"; else export CUDA_VISIBLE_DEVICES=0; variants="default eager bfloat16"; fi
  for variant in $variants; do
    python3 example/run.py --framework transformers --contract WhisperSpeech --require-all --isolate \
      --variant "$variant" --output "out/check-$platform-$variant.jsonl" > "out/check-$platform-$variant.log" 2>&1
  done
done
git -C /tmp/transformers fetch --quiet --depth 2 https://github.com/huggingface/transformers b6688043ff8e1c972c7fcac92db7942e1e41f42e
for platform in cpu cuda; do
  if [[ "$platform" == cpu ]]; then export CUDA_VISIBLE_DEVICES=""; else export CUDA_VISIBLE_DEVICES=0; fi
  TRANSFORMERS_REPO=/tmp/transformers python3 example/replay.py b6688043ff8e1c972c7fcac92db7942e1e41f42e \
    --contract WhisperSpeech --target whisper-tiny --variant eager --isolate > "out/replay-$platform.jsonl"
done
python3 - <<'PY'
import json
from pathlib import Path
for platform in ('cpu', 'cuda'):
    rows = [json.loads(line) for line in Path(f'out/replay-{platform}.jsonl').read_text().splitlines()]
    result = next(r for r in rows if r.get('target') == 'whisper-tiny')
    assert result['verdict'] == 'caught', result
    assert any('assisted_matches_greedy' in f for f in result['bad_failures']), result
    print(platform, result)
PY
