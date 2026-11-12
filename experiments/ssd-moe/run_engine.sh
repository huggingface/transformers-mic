#!/usr/bin/env bash
# Run one engine of the Qwen3.6-35B-A3B side-by-side under the same protocol:
# stage the pinned model files on local scratch, start the engine, wait until it is ready,
# drop the model files from the page cache, then send the same greedy chat request twice
# (run 1 = first request after eviction, run 2 = repeat). Usage: run_engine.sh ENGINE
set -euo pipefail
ENGINE=$1
HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${OUT:-$PWD/out/$ENGINE}
PATHS=${PATHS_JSON:-/mnt/cache/ssd-moe/paths.json}
MAX_TOKENS=${MAX_TOKENS:-32}
PY=${PY:-python3}
mkdir -p "$OUT"
# CPUs this pod may use: the cgroup quota, not nproc (the node has 96 CPUs, the pod about 22).
CPUS=$(awk '$1 != "max" {n = int($1 / $2)} END {print (n > 0 ? n : 0)}' /sys/fs/cgroup/cpu.max 2>/dev/null || echo 0)
[ "$CPUS" -gt 0 ] || CPUS=$(nproc)
echo "using $CPUS CPU threads" | tee "$OUT/cpus.txt"
field() { "$PY" -c "import json,sys; print(json.load(open('$PATHS'))[sys.argv[1]])" "$1"; }

{ nvidia-smi || true; df -h / /tmp || true; cat /proc/meminfo; nproc
  for f in memory.max memory.high cpu.max; do echo "cgroup $f: $(cat /sys/fs/cgroup/$f 2>/dev/null || echo n/a)"; done; } > "$OUT/environment.txt" 2>&1
cp "$PATHS" "$OUT/paths.json"
echo "memory limit: ${MEMORY_LIMIT:-none}" | tee "$OUT/memory-limit.txt"
cgroup_after() {  # peak RAM of this container (anon + page cache) and whether the OOM killer fired
  for f in memory.max memory.peak memory.events; do echo "== $f"; cat /sys/fs/cgroup/$f 2>/dev/null || echo n/a; done > "$OUT/cgroup-after.txt"
  grep -E "^(anon|file) " /sys/fs/cgroup/memory.stat >> "$OUT/cgroup-after.txt" 2>/dev/null || true
}
trap cgroup_after EXIT

stage() {  # stage SRC DST: copy (dereferencing HF cache symlinks) to container-local scratch, report throughput
  local src=$1 dst=$2 t0
  t0=$(date +%s)
  mkdir -p "$(dirname "$dst")"
  cp -rL "$src" "$dst"
  echo "staged $src -> $dst in $(( $(date +%s) - t0 ))s" | tee -a "$OUT/storage.txt"
  df -h "$dst" >> "$OUT/storage.txt"
}

disk_probe() {  # direct-I/O sequential read of 4 GiB of a staged file: the scratch disk's speed for context
  dd if="$1" of=/dev/null bs=16M count=256 iflag=direct 2>&1 | tail -1 | tee -a "$OUT/storage.txt" || true
}

bench() {  # bench STARTED_AT EXTRA_JSON API_KEY EVICT_PATHS...
  local started=$1 extra=$2 key=$3; shift 3
  "$PY" "$HERE/bench_client.py" --engine "$ENGINE" --prompt-file "$HERE/prompt.txt" --max-tokens "$MAX_TOKENS" \
    --started-at "$started" --extra "$extra" ${key:+--api-key "$key"} ${SERVER_PID:+--server-pid "$SERVER_PID"} --evict "$@" \
    --meta "{\"paths\": $(cat "$PATHS")}" --output "$OUT/result.json"
}

stop_server() { kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true; cgroup_after; }

case "$ENGINE" in
  transformers-streaming | transformers-offload)
    CKPT=/tmp/qwen36-bf16
    stage "$(field safetensors)" "$CKPT"
    disk_probe "$(ls "$CKPT"/*.safetensors | head -1)"
    git -C framework apply "$HERE/transformers.patch"
    "$PY" -m pip install -q -e framework
    "$PY" -m pip freeze > "$OUT/packages.txt"
    (cd framework && PYTHONPATH=src "$PY" -m unittest discover -s tests/integrations -p test_ssd_moe.py -v) > "$OUT/tests.log" 2>&1
    "$PY" - "$CKPT" <<'PY'
import os, sys
for d, _, fs in os.walk(sys.argv[1]):
    for f in fs:
        fd = os.open(os.path.join(d, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
    if [ "$ENGINE" = transformers-streaming ]; then
      mode_args=(--mode streaming --cache-mib "${CACHE_MIB:-16384}")
    else
      mode_args=(--mode offload --offload-gpu-mib 18432 --offload-cpu-mib "${OFFLOAD_CPU_MIB:-$([ -n "${MEMORY_LIMIT:-}" ] && echo 2048 || echo 40960)}" --offload-folder /tmp/offload)
    fi
    PYTHONPATH=framework/src "$PY" framework/examples/generate/ssd_moe.py "$CKPT" "${mode_args[@]}" \
      --chat --prompt-file "$HERE/prompt.txt" --device cuda --dtype bfloat16 --threads "$CPUS" \
      --max-new-tokens "$MAX_TOKENS" --runs 2 --output "$OUT/result.json" > "$OUT/engine.log" 2>&1
    ;;
  vllm)
    CKPT=/tmp/qwen36-bf16
    stage "$(field safetensors)" "$CKPT"
    disk_probe "$(ls "$CKPT"/*.safetensors | head -1)"
    "$PY" -m pip freeze > "$OUT/packages.txt"
    started=$(date +%s.%N)
    vllm serve "$CKPT" --served-model-name model --port 8080 --host 127.0.0.1 --dtype bfloat16 --seed 0 \
      --cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB:-50}" --gpu-memory-utilization 0.92 --max-model-len 4096 \
      --max-num-seqs 1 --enforce-eager --limit-mm-per-prompt '{"image": 0, "video": 0}' > "$OUT/engine.log" 2>&1 &
    SERVER_PID=$!; trap stop_server EXIT
    bench "$started" '{"ignore_eos": true}' "" "$CKPT"
    ;;
  llamacpp-bf16 | llamacpp-q8_0)
    if [ "$ENGINE" = llamacpp-bf16 ]; then GGUF=/tmp/gguf/model-bf16.gguf; stage "$(field gguf_bf16)" "$GGUF";
    else GGUF=/tmp/gguf/model-q8_0.gguf; stage "$(field gguf_q8_0)" "$GGUF"; fi
    disk_probe "$GGUF"
    # The runner's container hook does not keep the image's library path; the shared libraries live next to the binaries.
    export LD_LIBRARY_PATH=/app${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
    /app/llama-server --version > "$OUT/packages.txt" 2>&1 || true
    started=$(date +%s.%N)
    # --fit on: llama.cpp's own placement of layers/experts into the 24 GB GPU, rest of the experts on the CPU (mmap).
    /app/llama-server -m "$GGUF" --host 127.0.0.1 --port 8080 --alias model --fit on -c 4096 -np 1 \
      --jinja --seed 0 -t "$CPUS" -tb "$CPUS" > "$OUT/engine.log" 2>&1 &
    SERVER_PID=$!; trap stop_server EXIT
    bench "$started" '{"ignore_eos": true}' "" "$GGUF"
    ;;
  strata)
    PY=/opt/strata/terminal/python/bin/python3
    MODELDIR=/tmp/strata-models/Qwen3.6-35B-A3B-Q8_0
    stage "$(field gguf_q8_0)" "$MODELDIR/Qwen3.6-35B-A3B-Q8_0.gguf"
    disk_probe "$MODELDIR/Qwen3.6-35B-A3B-Q8_0.gguf"
    export XDG_STATE_HOME=/data/state XDG_CONFIG_HOME=/data/config HOME=/data/home STRATA_INSTALL_DIR=/data/install
    export STRATA_COMPONENTS_DIR=/opt/strata/components STRATA_IN_DOCKER=1
    mkdir -p /data/state /data/config /data/home /data/install /data/Strata-data
    KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
    export STRATA_LAUNCH_KEY=$KEY
    ls /opt/strata/components/engine > "$OUT/packages.txt"
    # Under a RAM cap, Strata's mapped mode: experts are read in place from the GGUF through the page cache (its SSD mode).
    # It sizes itself from /proc/meminfo (the host's RAM), so a container cap does not switch it on by itself.
    strata_set=(); [ -n "${MEMORY_LIMIT:-}" ] && strata_set=(--set mmap-experts=on)
    started=$(date +%s.%N)
    # One request at a time, no MTP/drafting, no pictures, no tuning: plain greedy decoding like the other engines.
    /opt/strata/terminal/bin/strata launch --no-tui --yes --access local --port 8080 --data-dir /data/Strata-data \
      --family qwen36 --model Q8_0 --model-path "$MODELDIR" --context 4096 --parallel 1 --vision off \
      --mtp off --drafting off --tune skip --api-key-env STRATA_LAUNCH_KEY ${strata_set[@]+"${strata_set[@]}"} > "$OUT/engine.log" 2>&1
    bench "$started" '{"ignore_eos": true}' "$KEY" "$MODELDIR" /data/Strata-data
    /opt/strata/terminal/bin/strata launch --status --output json > "$OUT/strata-status.json" 2>&1 || true
    /opt/strata/terminal/bin/strata launch --stop >> "$OUT/engine.log" 2>&1 || true
    ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
