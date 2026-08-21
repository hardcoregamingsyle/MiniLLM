"""
Pin a GGUF's always-hot tensors in RAM so the kernel can never evict them.

The problem this solves, measured on a 32 GB / PCIe 3.0 laptop running
Qwen3.8-2.4T UD-IQ2_XXS at 78.8 s/token:

  Per token the model touches ~34 GB of always-hot weights (attention,
  shared expert, routers, norms, lm_head) plus ~11 GB of routed experts.
  That total exceeds usable RAM, so the page cache evicts the hot half to
  make room for experts -- and re-reads all of it on the very next token.
  Every token pays for the whole hot set again, at 4 KB-fault speed
  (~0.58 GB/s measured, against a device that does ~3 GB/s).

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

On privilege: RLIMIT_MEMLOCK is commonly a few GB (this laptop reports
3977 MB) and locking 25 GB looks impossible against it. It is not. A root
process holds CAP_IPC_LOCK, and the kernel does not enforce RLIMIT_MEMLOCK
against a process that has it -- so under sudo the lock succeeds anyway.
Nothing in this file may refuse to call mlock() because a limit looks too
small; ask the kernel and let it answer. Linux only.
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
POSIX_FADV_WILLNEED = 3
PIDFILE = "/tmp/minillm-lock-hot.pid"

# mlock() walks and faults every page synchronously. Locking in slices lets us
# (a) report progress on a pin that takes minutes and (b) queue the NEXT
# slices with fadvise so the NVMe is never idle waiting for that page walk.
# Tunable only so tests can drive the failure path against a small
# RLIMIT_MEMLOCK; 64 MB is right for real use.
CHUNK = int(os.environ.get("MINILLM_LOCK_CHUNK_MB", "64")) * MB
LOOKAHEAD = 4 * CHUNK
PROGRESS_EVERY = 2 * GB

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
        try:
            _LIBC.posix_fadvise.restype = ctypes.c_int
            _LIBC.posix_fadvise.argtypes = [ctypes.c_int, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_int]
        except AttributeError:
            pass          # prefetch is an optimisation; locking works without
    return _LIBC


def fadvise(fd, off, length, advice=POSIX_FADV_WILLNEED):
    """Best-effort async prefetch. Never fatal: it only ever saves time."""
    try:
        libc().posix_fadvise(fd, off, length, advice)
    except (AttributeError, OSError):
        pass


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


def is_privileged():
    return hasattr(os, "geteuid") and os.geteuid() == 0


def raise_memlock(target):
    """Raise RLIMIT_MEMLOCK as far as allowed; return the resulting soft limit.

    Two separate mechanisms, and conflating them cost a full test cycle here:

      * CAP_SYS_RESOURCE (root) may raise its own HARD limit, so under sudo
        this usually just succeeds outright.
      * CAP_IPC_LOCK (also root) makes mlock() IGNORE RLIMIT_MEMLOCK entirely.
        So even when this function fails, root still locks as much as RAM
        allows.

    Returns -1 for unlimited, None if the limit is unreadable.
    """
    try:
        import resource
    except ImportError:
        return None
    INF = resource.RLIM_INFINITY
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    want = int(target) + 64 * MB          # headroom for page-alignment slop
    if hard == INF:
        # Never replace an unlimited hard limit with a finite one: lowering a
        # hard limit is irreversible for the process.
        attempts = [(INF, INF), (want, INF)]
    else:
        attempts = [(INF, INF), (want, max(want, hard)), (hard, hard)]
    for a in attempts:
        try:
            resource.setrlimit(resource.RLIMIT_MEMLOCK, a)
            break
        except (ValueError, OSError, OverflowError):
            continue
    return resource.getrlimit(resource.RLIMIT_MEMLOCK)[0]


def warn_limit(lim, want_b):
    """Report a too-small RLIMIT_MEMLOCK. Informational only -- NEVER fatal.

    Under sudo the limit does not apply (CAP_IPC_LOCK), and without sudo the
    kernel says so itself when mlock returns EPERM/ENOMEM. Refusing here on
    the strength of the number is exactly the bug this replaced: it printed
    "RLIMIT_MEMLOCK is 3977 MB but 25.0 GB must be locked" and exited without
    ever calling mlock, on a run that was already root and would have worked.
    """
    print(f"\nRLIMIT_MEMLOCK is {lim / MB:.0f} MB but {want_b / GB:.1f} GB is wanted.")
    if is_privileged():
        print("Running as root: CAP_IPC_LOCK means mlock() ignores this limit.")
        print("Proceeding -- the kernel decides, not the rlimit.")
        return
    print("NOT running as root, so this limit will be enforced and the pin will")
    print("stop early. Fix with EITHER:")
    print("  sudo -E python3 tools/lock_hot.py ...      (simplest; -E keeps HF_HUB_CACHE)")
    print("  or raise it permanently, then log out and back in:")
    print("    echo \"$USER hard memlock unlimited\" | sudo tee -a /etc/security/limits.conf")
    print("    echo \"$USER soft memlock unlimited\" | sudo tee -a /etc/security/limits.conf")


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


def lock_files(per_file, want_b):
    """mlock every range, in chunks, keeping whatever gets locked.

    Returns the list of held mappings, or None only if NOTHING was locked.
    A partial pin is not a failure: on a machine where the hot set does not
    fit, every pinned byte is a byte that stops being re-read every token.
    Throwing away 20 GB of successful locking because the 21st GB failed --
    which is what this used to do -- discards the entire benefit.
    """
    held, locked_b = [], 0
    next_note = PROGRESS_EVERY
    t0 = time.perf_counter()
    stopped = None
    for path, ranges in per_file.items():
        if stopped:
            break
        fd = os.open(path, os.O_RDONLY)
        try:
            for off, size in ranges:
                a_off = off & ~(PAGE - 1)
                length = size + (off - a_off)
                fadvise(fd, a_off, min(LOOKAHEAD, length))
                addr = libc().mmap(None, length, PROT_READ, MAP_SHARED, fd, a_off)
                if not addr or addr == MAP_FAILED:
                    print(f"  mmap failed at {a_off}: "
                          f"{os.strerror(ctypes.get_errno())}", file=sys.stderr)
                    continue
                # Keep the mapping even if the lock below stops part way: it
                # holds the chunks already locked.
                held.append((addr, length))
                pos = 0
                while pos < length:
                    n = min(CHUNK, length - pos)
                    ahead = pos + n
                    if ahead < length:
                        fadvise(fd, a_off + ahead, min(LOOKAHEAD, length - ahead))
                    if libc().mlock(ctypes.c_void_p(addr + pos), n) != 0:
                        stopped = os.strerror(ctypes.get_errno())
                        break
                    pos += n
                    locked_b += n
                    if locked_b >= next_note:
                        el = max(time.perf_counter() - t0, 1e-3)
                        print(f"  locked {locked_b / GB:6.2f} / {want_b / GB:.2f} GB"
                              f"   {locked_b / GB / el:.2f} GB/s")
                        sys.stdout.flush()
                        next_note = locked_b + PROGRESS_EVERY
                if stopped:
                    break
        finally:
            os.close(fd)   # the mapping holds its own reference
    dt = time.perf_counter() - t0

    if locked_b == 0:
        print(f"\nlocked nothing: {stopped or 'no lockable ranges'}", file=sys.stderr)
        print("Run under sudo (CAP_IPC_LOCK), or free RAM.", file=sys.stderr)
        unlock_all(held)
        return None

    print(f"\nlocked {locked_b / GB:.2f} GB in {dt:.1f}s "
          f"({locked_b / GB / max(dt, 0.001):.2f} GB/s)")
    if stopped:
        print(f"stopped early ({stopped}) -- KEEPING the {locked_b / GB:.2f} GB "
              f"already pinned.")
        print("Those bytes are now off every future token's read. That is the")
        print("whole point; it does not need to be all-or-nothing.")
    if locked_b < want_b:
        print(f"still streaming: {(want_b - locked_b) / GB:.2f} GB per token")
    # Self-verify against the kernel. VmLck in /proc/self/status is THIS
    # process's locked bytes -- unambiguous, unlike system-wide Mlocked, whose
    # accounting for file-backed shared pages varies (a GitHub runner reported
    # 44 MB while 1.79 GB was genuinely locked).
    vmlck = vmlck_bytes()
    if vmlck is not None:
        print(f"kernel VmLck   : {vmlck / GB:.2f} GB   <- confirmed unevictable")
        if vmlck < locked_b * 0.9:
            print(f"WARNING: kernel reports less locked than requested "
                  f"({vmlck / GB:.2f} vs {locked_b / GB:.2f} GB)", file=sys.stderr)
    else:
        print("kernel VmLck   : unavailable (no /proc/self/status)")
    sys.stdout.flush()
    return held


def vmlck_bytes():
    """This process's locked memory per the kernel, or None if unavailable."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmLck:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


def unlock_all(held):
    for addr, length in held:
        libc().munlock(ctypes.c_void_p(addr), length)
        libc().munmap(ctypes.c_void_p(addr), length)


def drop_pidfile():
    """Remove the pidfile, tolerating /tmp sticky-bit ownership mismatches."""
    try:
        os.remove(PIDFILE)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pattern")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stop", action="store_true", help="release a running locker")
    ap.add_argument("--max-gb", type=float, default=0.0,
                    help="pin at most N GB. When the hot set is LARGER than RAM, "
                         "pinning as much as fits is still a large win: every "
                         "pinned byte is one never re-read again. 0 = pin it all.")
    ap.add_argument("--hold-seconds", type=int, default=0,
                    help="hold the lock for N seconds then release (0 = forever). "
                         "Lets a script verify the lock without backgrounding.")
    args = ap.parse_args()

    if args.stop:
        if not os.path.exists(PIDFILE):
            print("no locker running")
            return 0
        try:
            pid = int(open(PIDFILE).read().strip())
        except (OSError, ValueError):
            print("unreadable pidfile; removing")
            drop_pidfile()
            return 0
        try:
            os.kill(pid, signal.SIGTERM)
            print(f"stopped locker (pid {pid})")
        except ProcessLookupError:
            print("locker already gone")
        except PermissionError:
            print(f"locker (pid {pid}) belongs to root -- retry with:\n"
                  f"  sudo python3 tools/lock_hot.py --stop", file=sys.stderr)
            return 1
        drop_pidfile()
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
    if args.max_gb > 0 and hot_b > args.max_gb * GB:
        # Trim to the budget. Any subset works: every hot tensor is touched
        # exactly once per token, so they are equally valuable to pin -- what
        # matters is only how many bytes stop being re-read.
        budget, trimmed, kept = args.max_gb * GB, {}, 0
        for path, ranges in per_file.items():
            keep = []
            for off, size in ranges:
                if kept >= budget:
                    break
                take = min(size, int(budget - kept))
                if take < PAGE:
                    break
                keep.append((off, take))
                kept += take
            if keep:
                trimmed[path] = keep
        print(f"budget {args.max_gb:.1f} GB: pinning {kept / GB:.2f} of "
              f"{hot_b / GB:.2f} GB hot; the remaining "
              f"{(hot_b - kept) / GB:.2f} GB still streams each token.")
        per_file, hot_b = trimmed, kept
    print(f"files          : {len(per_file)}")
    print(f"hot (to lock)  : {hot_b / GB:8.2f} GB in "
          f"{sum(len(v) for v in per_file.values())} regions")
    print(f"experts (skip) : {exp_b / GB:8.2f} GB   <- router picks ~2% per token")

    lim = raise_memlock(hot_b)
    if lim == -1:
        print("RLIMIT_MEMLOCK : unlimited")
    elif lim is not None and lim < hot_b:
        warn_limit(lim, hot_b)

    if args.dry_run:
        return 0

    if sys.platform != "linux":
        print("\nlock_hot.py needs Linux (mlock + shared page cache).", file=sys.stderr)
        return 1

    if os.path.exists(PIDFILE):
        old = open(PIDFILE).read().strip()
        if old.isdigit() and os.path.exists(f"/proc/{old}"):
            print(f"a locker is already running (pid {old}). --stop it first.")
            return 0

    held = lock_files(per_file, hot_b)
    if held is None:
        return 1

    try:
        with open(PIDFILE, "w") as f:
            f.write(str(os.getpid()))
        os.chmod(PIDFILE, 0o666)   # so a non-root --stop can clean up after
    except OSError as e:
        print(f"note: could not write {PIDFILE} ({e}); stop this pid with kill",
              file=sys.stderr)

    def release(*_):
        unlock_all(held)
        drop_pidfile()
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
