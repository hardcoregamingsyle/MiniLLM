"""
Pin a GGUF's always-hot tensors in RAM so the kernel can never evict them.

The problem this solves, measured on a 32 GB / PCIe 3.0 laptop running
Qwen3.8-2.4T UD-IQ2_XXS at 30 s/token:

  Per token the model touches ~20-26 GB of always-hot weights (attention,
  shared expert, routers, norms, lm_head) plus ~11 GB of routed experts.
  That total exceeds usable RAM, so the page cache evicts the hot half to
  make room for experts -- and re-reads all of it on the very next token.
  Every token pays for the whole hot set again, at 4 KB-fault speed
  (~1 GB/s measured, against a device that does ~3 GB/s).

The fix: mmap the hot byte ranges in a separate process and mlock() them.
Page-cache pages are SHARED between processes mapping the same file, so
llama.cpp's own mmap finds those pages resident and never faults them. The
kernel cannot reclaim locked pages, so experts can churn through the rest of
RAM without ever costing the hot set again.

The routed-expert pool is deliberately left alone: it is 95% of the file and
the router picks ~2% of it per token. Locking that is neither possible nor
useful.

  sudo -E python3 tools/lock_hot.py --pattern Qwen3.8-2.4T  # lock and hold
     (-E preserves HF_HUB_CACHE; sudo strips it and the search fails)
  python3 tools/lock_hot.py --pattern Qwen3.8-2.4T --dry-run
  python3 tools/lock_hot.py --stop                        # release

Needs privilege to lock more than RLIMIT_MEMLOCK (commonly 8-64 MB by
default). Run under sudo, or raise the limit -- the script prints how.
Linux only: this relies on mlock plus a shared page cache.
"""

import argparse
import ctypes
import ctypes.util
import os
import signal
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warm_gguf import Gguf, EXPERT_RE, find_files, sibling_shards  # noqa: E402

GB = 1024 ** 3
MB = 1024 ** 2
PAGE = 4096
PROT_READ = 1
MAP_SHARED = 1
MAP_FAILED = ctypes.c_void_p(-1).value
ENOMEM = 12
PIDFILE = "/tmp/minillm-lock-hot.pid"

_LIBC = None


def libc():
    """Load libc on first use, not at import -- so --dry-run (which only parses
    GGUF headers and plans regions) works on any OS, including Windows and in
    CI lint, where there is no libc.so.6 to bind mmap/mlock against."""
    global _LIBC
    if _LIBC is None:
        _LIBC = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6",
                            use_errno=True)
        _LIBC.mmap.restype = ctypes.c_void_p
        _LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                               ctypes.c_int, ctypes.c_int, ctypes.c_int64]
        _LIBC.mlock.restype = ctypes.c_int
        _LIBC.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _LIBC.munlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        _LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return _LIBC


def coalesce(ranges, gap=1 * MB):
    """Merge hot ranges that are adjacent or nearly so.

    Far fewer, larger mmap+mlock calls; the small gaps between hot tensors
    cost less to include than the syscalls saved by not splitting.
    """
    ranges.sort()
    out = []
    for off, size in ranges:
        if out and off - (out[-1][0] + out[-1][1]) <= gap:
            p_off, p_size = out[-1]
            out[-1] = (p_off, off + size - p_off)
        else:
            out.append((off, size))
    return out


def memlock_limit():
    """Current RLIMIT_MEMLOCK soft limit, after trying to raise it to hard.
    Often the hard limit is already unlimited and only the soft one is small."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
        if soft != hard:
            try:
                resource.setrlimit(resource.RLIMIT_MEMLOCK, (hard, hard))
                soft = hard
            except (ValueError, OSError):
                pass
        return soft
    except Exception:
        return None


def plan(files):
    """-> (per_file {path: [(off,size)]}, hot_bytes, expert_bytes)"""
    per_file, hot_b, exp_b = {}, 0, 0
    for path in files:
        try:
            g = Gguf(path)
        except ValueError as e:
            print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        hot = []
        for name, off, size in g.tensors:
            if EXPERT_RE.search(name):
                exp_b += size
            else:
                hot.append((off, size))
                hot_b += size
        if hot:
            per_file[path] = coalesce(hot)
    return per_file, hot_b, exp_b


def explain_limit(lim, hot_b):
    print(f"\nRLIMIT_MEMLOCK is {lim / MB:.0f} MB but {hot_b / GB:.1f} GB must be locked.")
    print("Fix with EITHER:")
    print("  sudo -E python3 tools/lock_hot.py ...        (simplest; -E keeps HF_HUB_CACHE)")
    print("  or raise it permanently, then log out and back in:")
    print('    echo "$USER hard memlock unlimited" | sudo tee -a /etc/security/limits.conf')
    print('    echo "$USER soft memlock unlimited" | sudo tee -a /etc/security/limits.conf')


def lock_files(per_file, hot_b):
    held, locked_b = [], 0
    t0 = time.perf_counter()
    for path, ranges in per_file.items():
        fd = os.open(path, os.O_RDONLY)
        try:
            for off, size in ranges:
                a_off = off & ~(PAGE - 1)
                length = size + (off - a_off)
                addr = libc().mmap(None, length, PROT_READ, MAP_SHARED, fd, a_off)
                if addr == MAP_FAILED:
                    print(f"  mmap failed at {a_off}: "
                          f"{os.strerror(ctypes.get_errno())}", file=sys.stderr)
                    continue
                if libc().mlock(ctypes.c_void_p(addr), length) != 0:
                    err = ctypes.get_errno()
                    libc().munmap(ctypes.c_void_p(addr), length)
                    print(f"\nmlock failed after {locked_b / GB:.1f} GB: "
                          f"{os.strerror(err)}", file=sys.stderr)
                    if err == ENOMEM:
                        print("Out of lockable memory: run under sudo, or raise "
                              "RLIMIT_MEMLOCK.", file=sys.stderr)
                    unlock_all(held)
                    return None
                held.append((addr, length))
                locked_b += length
        finally:
            os.close(fd)   # the mapping holds its own reference
    dt = time.perf_counter() - t0
    print(f"\nlocked {locked_b / GB:.2f} GB in {dt:.1f}s "
          f"({locked_b / GB / max(dt, 0.001):.2f} GB/s)")
    return held


def unlock_all(held):
    for addr, length in held:
        libc().munlock(ctypes.c_void_p(addr), length)
        libc().munmap(ctypes.c_void_p(addr), length)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pattern")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop", action="store_true", help="release a running locker")
    ap.add_argument("--hold-seconds", type=int, default=0,
                    help="hold the lock for N seconds then release (0 = forever). "
                         "Lets a script verify the lock without backgrounding.")
    args = ap.parse_args()

    if args.stop:
        if not os.path.exists(PIDFILE):
            print("no locker running")
            return 0
        pid = int(open(PIDFILE).read().strip())
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped locker (pid {pid})")
        except ProcessLookupError:
            print("locker already gone")
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
        return 0

    files = []
    for f in args.files:
        files.extend(sibling_shards(f))
    if args.pattern:
        for f in find_files(args.pattern):
            files.extend(sibling_shards(f))
    files = sorted(set(files))
    if not files:
        print("No GGUF matched. Pass a path or --pattern.", file=sys.stderr)
        # sudo strips the environment by default, so HF_HUB_CACHE / HF_HOME are
        # gone and the search ran against the wrong directory as root. This is
        # the single most likely reason to land here.
        if os.environ.get("SUDO_USER") and not os.environ.get("HF_HUB_CACHE"):
            print(f"\nRunning under sudo without the cache path: searched\n"
                  f"  {find_files.__globals__['HF_CACHE']}\n"
                  f"Re-run preserving your environment:\n"
                  f"  sudo -E python3 tools/lock_hot.py --pattern ...\n"
                  f"or pass the GGUF path explicitly (no --pattern needed).",
                  file=sys.stderr)
        return 1

    per_file, hot_b, exp_b = plan(files)
    print(f"files          : {len(per_file)}")
    print(f"hot (to lock)  : {hot_b / GB:8.2f} GB in "
          f"{sum(len(v) for v in per_file.values())} regions")
    print(f"experts (skip) : {exp_b / GB:8.2f} GB   <- router picks ~2% per token")

    if args.dry_run:
        lim = memlock_limit()
        if lim is not None and lim != -1 and lim < hot_b:
            explain_limit(lim, hot_b)
        return 0

    if sys.platform != "linux":
        print("\nlock_hot.py needs Linux (mlock + shared page cache).", file=sys.stderr)
        return 1

    if os.path.exists(PIDFILE):
        old = open(PIDFILE).read().strip()
        if old.isdigit() and os.path.exists(f"/proc/{old}"):
            print(f"a locker is already running (pid {old}). --stop it first.")
            return 0

    lim = memlock_limit()
    if lim is not None and lim != -1 and lim < hot_b:
        explain_limit(lim, hot_b)
        return 1

    held = lock_files(per_file, hot_b)
    if held is None:
        return 1

    with open(PIDFILE, "w") as f:
        f.write(str(os.getpid()))

    def release(*_):
        unlock_all(held)
        if os.path.exists(PIDFILE):
            os.remove(PIDFILE)
        print("\nreleased.")
        sys.exit(0)

    signal.signal(signal.SIGTERM, release)
    signal.signal(signal.SIGINT, release)
    if args.hold_seconds > 0:
        print(f"holding for {args.hold_seconds}s (pid {os.getpid()}), then releasing.")
        sys.stdout.flush()
        time.sleep(args.hold_seconds)
        release()
    print(f"holding (pid {os.getpid()}). The hot set is now un-evictable.")
    print("Run the model in another terminal. Release with:")
    print("  python3 tools/lock_hot.py --stop")
    sys.stdout.flush()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
