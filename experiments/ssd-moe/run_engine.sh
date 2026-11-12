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
echo "memory limit: ${MEMORY_LIMIT:-none}, GPU limit: ${GPU_LIMIT:-none}" | tee "$OUT/memory-limit.txt"
cgroup_after() {  # peak RAM of this container (anon + page cache) and whether the OOM killer fired
  { echo "balloon alive: $([ -n "${BALLOON_PID:-}" ] && kill -0 "$BALLOON_PID" 2>/dev/null && echo yes || echo no)"
    for f in memory.max memory.peak memory.events; do echo "== $f"; cat /sys/fs/cgroup/$f 2>/dev/null || echo n/a; done; } > "$OUT/cgroup-after.txt"
  grep -E "^(anon|file) " /sys/fs/cgroup/memory.stat >> "$OUT/cgroup-after.txt" 2>/dev/null || true
  [ -z "${BALLOON_PID:-}" ] || kill "$BALLOON_PID" 2>/dev/null || true
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

bytes_of() { "$PY" -c "import sys; v=sys.argv[1].lower(); u={'k':1<<10,'m':1<<20,'g':1<<30}; print(int(float(v[:-1])*u[v[-1]]) if v[-1] in u else int(v))" "$1"; }

limit_memory() {  # with MEMORY_LIMIT / GPU_LIMIT (e.g. 12g), leave the engine only that much RAM / GPU memory; call after staging
  [ -n "${MEMORY_LIMIT:-}${GPU_LIMIT:-}" ] || return 0
  local ram=0 gpu=0
  [ -z "${MEMORY_LIMIT:-}" ] || ram=$(bytes_of "$MEMORY_LIMIT")
  [ -z "${GPU_LIMIT:-}" ] || gpu=$(bytes_of "$GPU_LIMIT")
  rm -f /tmp/balloon.ready
  "$PY" "$HERE/balloon.py" "$ram" "$gpu" /tmp/balloon.ready &
  BALLOON_PID=$!
  for _ in $(seq 600); do [ -s /tmp/balloon.ready ] && break; kill -0 "$BALLOON_PID" || { echo "balloon died" >&2; exit 1; }; sleep 1; done
  [ -s /tmp/balloon.ready ] || { echo "balloon not ready" >&2; exit 1; }
  { cat /tmp/balloon.ready; cat /proc/swaps; grep -E "^(anon|file) " /sys/fs/cgroup/memory.stat
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv; } 2>&1 | tee -a "$OUT/memory-limit.txt"
  GPU_BALLOON_BYTES=$(awk '$1 == "gpu" {print $5}' /tmp/balloon.ready); export GPU_BALLOON_BYTES=${GPU_BALLOON_BYTES:-0}
  echo 1000 > /proc/self/oom_score_adj  # the engine, not the balloon, is what the OOM killer takes
}

case "$ENGINE" in
  transformers-streaming | transformers-streaming-q8_0 | transformers-offload)
    CKPT=/tmp/qwen36-bf16
    stage "$(field safetensors)" "$CKPT"
    disk_probe "$(ls "$CKPT"/*.safetensors | head -1)"
    evict_paths=("$CKPT")
    if [ "$ENGINE" = transformers-streaming-q8_0 ]; then  # the same Q8_0 GGUF file llama.cpp and Strata read
      GGUF=/tmp/gguf/model-q8_0.gguf; stage "$(field gguf_q8_0)" "$GGUF"; evict_paths+=(/tmp/gguf)
    fi
    limit_memory
    git -C framework apply "$HERE/transformers.patch"
    "$PY" -m pip install -q -e framework "kernels>=0.18.2,<0.19"
    "$PY" -m pip freeze > "$OUT/packages.txt"
    (cd framework && PYTHONPATH=src "$PY" -m unittest discover -s tests/integrations -p test_ssd_moe.py -v) > "$OUT/tests.log" 2>&1
    if [ "$ENGINE" = transformers-streaming-q8_0 ]; then
      PYTHONPATH=framework/src "$PY" "$HERE/check_gguf_experts.py" "$CKPT" "$GGUF" "$OUT/gguf-check.json" 2>&1 | tee "$OUT/gguf-check.log"
    fi
    "$PY" - "${evict_paths[@]}" <<'PY'
import os, sys
for root in sys.argv[1:]:
    for d, _, fs in os.walk(root):
        for f in fs:
            fd = os.open(os.path.join(d, f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
PY
    # Expert cache: what the GPU budget leaves after the 4.6 GiB of BF16 dense weights, activations and the CUDA context.
    cache_mib=${CACHE_MIB:-$([ -n "${GPU_LIMIT:-}" ] && echo $(( $(bytes_of "$GPU_LIMIT") / 1048576 - 6656 )) || echo 16384)}
    if [ "$ENGINE" = transformers-streaming-q8_0 ]; then
      mode_args=(--mode streaming --cache-mib "$cache_mib" --experts-gguf "$GGUF")
    elif [ "$ENGINE" = transformers-streaming ]; then
      mode_args=(--mode streaming --cache-mib "$cache_mib")
    else
      mode_args=(--mode offload --offload-gpu-mib 18432 --offload-cpu-mib "${OFFLOAD_CPU_MIB:-$([ -n "${MEMORY_LIMIT:-}" ] && echo 2048 || echo 40960)}" --offload-folder /tmp/offload)
    fi
    # Hub kernels (Qwen3.6 linear attention, causal conv, RMSNorm) are fetched from the Hub, so this step is online.
    HF_HUB_OFFLINE=0 PYTHONPATH=framework/src "$PY" framework/examples/generate/ssd_moe.py "$CKPT" "${mode_args[@]}" --use-kernels \
      --chat --prompt-file "$HERE/prompt.txt" --device cuda --dtype bfloat16 --threads "$CPUS" \
      --max-new-tokens "$MAX_TOKENS" --runs 2 --output "$OUT/result.json" > "$OUT/engine.log" 2>&1
    ;;
  vllm)
    CKPT=/tmp/qwen36-bf16
    stage "$(field safetensors)" "$CKPT"
    disk_probe "$(ls "$CKPT"/*.safetensors | head -1)"
    limit_memory
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
    limit_memory
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
    limit_memory
    export XDG_STATE_HOME=/data/state XDG_CONFIG_HOME=/data/config HOME=/data/home STRATA_INSTALL_DIR=/data/install
    export STRATA_COMPONENTS_DIR=/opt/strata/components STRATA_IN_DOCKER=1
    mkdir -p /data/state /data/config /data/home /data/install /data/Strata-data
    KEY=$(head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n')
    export STRATA_LAUNCH_KEY=$KEY
    ls /opt/strata/components/engine > "$OUT/packages.txt"
    # Under a RAM cap, Strata's RAM-budget mode: the hottest N GiB of experts the GPU does not hold are copied into RAM, the
    # rest are read in place from the GGUF (its SSD mode). It sizes itself from /proc/meminfo (the host's RAM, hundreds of
    # GiB), so without an explicit budget its automatic RAM copy outgrows the cap and the pod is OOM-killed.
    strata_set=()
    if [ -n "${MEMORY_LIMIT:-}" ]; then
      strata_set=(--set "resident-budget-gib=${STRATA_RAM_BUDGET_GIB:-4}" --set "memory-limit-mib=$(( $(bytes_of "$MEMORY_LIMIT") / 1048576 ))")
    fi
    # Follow the engine log in the job log: an OOM kill takes the container, and the uploaded artifact, with it.
    tail -n +1 -F "$OUT/engine.log" 2>/dev/null &
    TAIL_PID=$!
    started=$(date +%s.%N)
    # One request at a time, no MTP/drafting, no pictures, no tuning: plain greedy decoding like the other engines.
    /opt/strata/terminal/bin/strata launch --no-tui --yes --access local --port 8080 --data-dir /data/Strata-data \
      --family qwen36 --model Q8_0 --model-path "$MODELDIR" --context 4096 --parallel 1 --vision off \
      --mtp off --drafting off --tune skip --api-key-env STRATA_LAUNCH_KEY ${strata_set[@]+"${strata_set[@]}"} > "$OUT/engine.log" 2>&1
    bench "$started" '{"ignore_eos": true}' "$KEY" "$MODELDIR" /data/Strata-data
    /opt/strata/terminal/bin/strata launch --status --output json > "$OUT/strata-status.json" 2>&1 || true
    /opt/strata/terminal/bin/strata launch --stop >> "$OUT/engine.log" 2>&1 || true
    kill "$TAIL_PID" 2>/dev/null || true
    ;;
  *) echo "unknown engine $ENGINE" >&2; exit 2 ;;
esac
