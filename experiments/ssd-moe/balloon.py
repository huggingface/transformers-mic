"""Hold all of this container's RAM except TARGET bytes, so an engine sees a small machine.

The runner's container hook ignores docker --memory, so the pod's cgroup limit cannot be lowered from the job. Anonymous
memory, touched and locked, cannot be reclaimed (no swap), so the page cache and the engine share what is left.
Usage: balloon.py TARGET_BYTES READY_FILE. Stays alive until killed.
"""

import ctypes
import mmap
import os
import sys
import time

target, ready = int(sys.argv[1]), sys.argv[2]
limit = int(open("/sys/fs/cgroup/memory.max").read().strip())
size = limit - target
size -= size % mmap.PAGESIZE
buf = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
chunk = b"\1" * (64 << 20)
done = 0
while done < size:
    n = min(len(chunk), size - done)
    buf.write(chunk[:n])
    done += n
libc = ctypes.CDLL(None, use_errno=True)
locked = libc.mlock(ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(buf))), ctypes.c_size_t(size)) == 0
with open(ready, "w") as f:
    f.write(f"cgroup limit {limit} balloon {size} target {target} mlocked {locked} errno {ctypes.get_errno()}\n")
while True:
    time.sleep(3600)
