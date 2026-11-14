#!/bin/bash
# Qwen3.8-Flash-Next with moe-expert-streaming on one A10G: convert once into the shared cache, then benchmark.
# Inputs (env): STEPS (comma list: convert,kernel,decode,decode-bypass,decode-host,prefill,prefill-host,ppl), CACHE_GIB, PREFILL_CACHE_GIB, HOST_CACHE_GIB, IO_THREADS
set -euo pipefail
MODEL=/mnt/cache/moe-expert-streaming/qwen38
OUT=$PWD/out
mkdir -p "$OUT"
has() { [[ ",$STEPS," == *",$1,"* ]]; }
run() { echo "::group::$1"; shift; set -o pipefail; "$@" 2>&1 | tee -a "$OUT/log.txt"; echo "::endgroup::"; }

if has convert; then
  if [ -f "$MODEL/experts-q3/quantization.json" ] && [ -f "$MODEL/ngram/ngram.json" ]; then
    echo "converted model already in the cache: $MODEL"
  else
    run convert python pkg/scripts/convert_fp8.py "$MODEL" --bits 3 --device cuda
  fi
  du -sh "$MODEL"/* | tee "$OUT/model-size.txt"
fi
if has kernel; then
  run kernel python pkg/scripts/check_cuda_kernel.py
fi
common=(--io-threads "$IO_THREADS" --device cuda)
if has decode; then
  run decode python pkg/scripts/bench.py "$MODEL" --cache-gib "$CACHE_GIB" --skip-prefill "${common[@]}" \
    --output "$OUT/decode.json"
fi
if has decode-bypass; then
  run decode-bypass python pkg/scripts/bench.py "$MODEL" --cache-gib "$CACHE_GIB" --skip-prefill --bypass-page-cache \
    "${common[@]}" --output "$OUT/decode-bypass.json"
fi
if has decode-host; then
  run decode-host python pkg/scripts/bench.py "$MODEL" --cache-gib "$CACHE_GIB" --skip-prefill \
    --host-cache-gib "$HOST_CACHE_GIB" --host-preload "${common[@]}" --output "$OUT/decode-host.json"
fi
if has prefill-host; then
  run prefill-host python pkg/scripts/bench.py "$MODEL" --cache-gib "$PREFILL_CACHE_GIB" --skip-story \
    --prefill-chunk-size 1024 --host-cache-gib "$HOST_CACHE_GIB" --host-preload "${common[@]}" \
    --output "$OUT/prefill-host.json"
fi
if has prefill; then
  run prefill python pkg/scripts/bench.py "$MODEL" --cache-gib "$PREFILL_CACHE_GIB" --skip-story \
    --prefill-chunk-size 1024 "${common[@]}" --output "$OUT/prefill.json"
fi
if has ppl; then
  curl -sSfL -o philosophy.md https://raw.githubusercontent.com/huggingface/transformers/158f872ebb/docs/source/en/philosophy.md
  (cd "$OUT" && run ppl python "$OLDPWD/pkg/scripts/decode_ppl.py" "$MODEL" "$OLDPWD/philosophy.md" cuda-bf16 512 16 "$CACHE_GIB")
fi
