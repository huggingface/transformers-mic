"""Shrink this job to a smaller machine: hold the RAM and GPU memory beyond RAM_TARGET and GPU_TARGET bytes.

The runner's Kubernetes container hook ignores docker --memory, and a GPU cannot be split, so both limits are simulated:
- RAM: anonymous memory, touched and mlocked, cannot be reclaimed (no swap), so the engine and the page cache share
  what is left of the cgroup limit.
- GPU: one device allocation through the CUDA driver API, so engines that size themselves from the free GPU memory
  (llama.cpp --fit, Strata, vLLM) and the PyTorch caching allocator see only GPU_TARGET bytes.
Usage: balloon.py RAM_TARGET GPU_TARGET READY_FILE (0 = leave that one alone). Stays alive until killed.
"""

import ctypes
import mmap
import sys
import time


ram_target, gpu_target, ready = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
report = []

if gpu_target:
    cuda = ctypes.CDLL("libcuda.so.1")

    def check(result, what):
        if result:
            raise RuntimeError(f"{what} failed with CUDA error {result}")

    device, context = ctypes.c_int(), ctypes.c_void_p()
    free, total, pointer = ctypes.c_size_t(), ctypes.c_size_t(), ctypes.c_uint64()
    check(cuda.cuInit(0), "cuInit")
    check(cuda.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")
    check(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device), "cuDevicePrimaryCtxRetain")
    check(cuda.cuCtxSetCurrent(context), "cuCtxSetCurrent")
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    size = free.value - gpu_target
    if size <= 0:
        raise RuntimeError(f"only {free.value} bytes of GPU memory are free; cannot leave {gpu_target}")
    check(cuda.cuMemAlloc_v2(ctypes.byref(pointer), ctypes.c_size_t(size)), "cuMemAlloc")
    check(cuda.cuMemsetD8_v2(pointer, ctypes.c_ubyte(1), ctypes.c_size_t(size)), "cuMemsetD8")
    check(cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total)), "cuMemGetInfo")
    report.append(f"gpu total {total.value} balloon {size} free now {free.value} target {gpu_target}")

if ram_target:
    limit = int(open("/sys/fs/cgroup/memory.max").read().strip())
    size = limit - ram_target
    size -= size % mmap.PAGESIZE
    buffer = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    chunk = b"\1" * (64 << 20)
    done = 0
    while done < size:
        count = min(len(chunk), size - done)
        buffer.write(chunk[:count])
        done += count
    libc = ctypes.CDLL(None, use_errno=True)
    address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    locked = libc.mlock(ctypes.c_void_p(address), ctypes.c_size_t(size)) == 0
    report.append(
        f"ram cgroup limit {limit} balloon {size} target {ram_target} mlocked {locked} errno {ctypes.get_errno()}"
    )

with open(ready, "w") as handle:
    handle.write("\n".join(report or ["no limits"]) + "\n")
while True:
    time.sleep(3600)
