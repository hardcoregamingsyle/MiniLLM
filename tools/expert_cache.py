"""
A routing-aware expert cache, built entirely OUTSIDE the inference process.

The idea
--------
On a 32 GB box the Qwen3.8-2.4T expert pool is 569 GB and one token touches
11.1 GB of it. Lock the always-hot half (tools/lock_hot.py) and ~7-11 GB of
RAM is left over. Filling that with the RIGHT experts is worth a lot: at a 46%
hit rate the model reaches 2.0 s/token (0.5 tok/s) instead of 3.7.

llama.cpp has no notion of an expert cache -- the kernel's page cache evicts
4 KB pages by recency, blind to which expert they belong to. But MoE routing
is not uniform: some experts are picked far more often than others, and that
skew is stable for a given workload.

So: learn the skew by OBSERVATION, then pin the winners.

  1. mmap the expert tensors (address space only -- no reads, no I/O).
  2. While the model runs, call mincore() every few seconds. It reports which
     pages are currently resident WITHOUT touching them. A page is resident
     because something recently read it -- so residency is a free, zero-cost
     sample of which experts the router is choosing.
  3. Aggregate pages into per-expert buckets and count how often each expert
     shows up resident. That is a popularity histogram of real routing on the
     real prompt distribution.
  4. mlock the top experts that fit the budget. They can never be evicted, so
     every future hit on them costs zero I/O.

No llama.cpp changes, no instrumentation, no routing hooks. The page cache is
shared between processes mapping the same file, so pinning here is visible to
llama.cpp immediately.

  # learn for 3 minutes while the model generates, then pin 7 GB of winners
  sudo -E python3 tools/expert_cache.py --pattern Qwen3.8-2.4T --observe 180 --lock-gb 7

  # just look at the skew, no locking, no privilege
  python3 tools/expert_cache.py --pattern Qwen3.8-2.4T --observe 120 --report-only

  # reuse a saved histogram (skip observation)
  sudo -E python3 tools/expert_cache.py --pattern Qwen3.8-2.4T --load-stats stats.json --lock-gb 7

Linux only: needs mincore + mlock and a shared page cache.
"""

import argparse
import ctypes
import json
import os
import signal
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warm_gguf import Gguf, EXPERT_RE, find_files, sibling_shards  # noqa: E402
from lock_hot import (libc, coalesce, memlock_limit, explain_limit,  # noqa: E402
                      vmlck_bytes, unlock_all, PAGE, PROT_READ,
                      MAP_SHARED, MAP_FAILED, GB, MB)

PIDFILE = "/tmp/minillm-expert-cache.pid"
# Only the weight tensors are worth caching; the *_exps.bias tensors are tiny
# and get pulled in with the hot set anyway.
WEIGHT_SUFFIX = ".weight"


def expert_slices(files):
    """-> (slices, n_experts, bytes_per_expert)

    A GGUF stores every expert of a layer in ONE tensor:
    blk.N.ffn_gate_exps.weight is [n_expert, ...] laid out back to back. So
    expert e of that tensor is the slice [e*S, (e+1)*S) where S = size/n_expert.
    An "expert" for caching purposes is all of its slices (gate + up + down).
    """
    slices = defaultdict(list)          # (layer, expert) -> [(path, off, size)]
    n_experts = None
    for path in files:
        try:
            g = Gguf(path)
        except ValueError as e:
            print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        n = next((v for k, v in g.kv.items() if k.endswith(".expert_count")), None)
        if n:
            n_experts = n
        if not n_experts:
            continue
        for name, off, size in g.tensors:
            if not (EXPERT_RE.search(name) and name.endswith(WEIGHT_SUFFIX)):
                continue
            try:
                layer = int(name.split(".")[1])
            except (IndexError, ValueError):
                continue
            per = size // n_experts
            if per < PAGE:              # too small to be worth per-expert work
                continue
            for e in range(n_experts):
                slices[(layer, e)].append((path, off + e * per, per))
    total = sum(sz for v in slices.values() for _, _, sz in v)
    per_expert = total / max(len(slices), 1)
    return slices, n_experts, per_expert


class Observer:
    """Samples page residency of the expert regions via mincore()."""

    def __init__(self, files):
        self.maps = []          # (path, base_off, addr, length)
        for path in files:
            size = os.path.getsize(path)
            fd = os.open(path, os.O_RDONLY)
            try:
                # One mapping per file. This costs address space only -- no
                # reads, no page cache pressure, no I/O.
                addr = libc().mmap(None, size, PROT_READ, MAP_SHARED, fd, 0)
                if addr == MAP_FAILED:
                    print(f"  mmap failed for {os.path.basename(path)}: "
                          f"{os.strerror(ctypes.get_errno())}", file=sys.stderr)
                    continue
                self.maps.append((path, addr, size))
            finally:
                os.close(fd)
        libc().mincore.restype = ctypes.c_int
        libc().mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t,
                                   ctypes.POINTER(ctypes.c_ubyte)]

    def resident_fraction(self, path, off, size):
        """What fraction of this byte range is currently in the page cache."""
        for p, addr, msize in self.maps:
            if p != path:
                continue
            a_off = off & ~(PAGE - 1)
            length = size + (off - a_off)
            npages = (length + PAGE - 1) // PAGE
            vec = (ctypes.c_ubyte * npages)()
            if libc().mincore(ctypes.c_void_p(addr + a_off), length, vec) != 0:
                return 0.0
            return sum(1 for b in vec if b & 1) / npages
        return 0.0

    def close(self):
        for _, addr, size in self.maps:
            libc().munmap(ctypes.c_void_p(addr), size)


def observe(slices, files, seconds, interval):
    """Sample residency repeatedly -> popularity score per expert."""
    obs = Observer(files)
    counts = defaultdict(float)
    n = 0
    t_end = time.time() + seconds
    print(f"observing {len(slices)} experts for {seconds}s "
          f"(sample every {interval}s)...")
    print("Run the model NOW in another terminal so there is routing to learn from.")
    try:
        while time.time() < t_end:
            n += 1
            for key, parts in slices.items():
                # One representative slice per expert is enough and keeps the
                # sampling cheap: gate/up/down are read together.
                path, off, size = parts[0]
                counts[key] += obs.resident_fraction(path, off, size)
            hot = sum(1 for v in counts.values() if v > 0)
            print(f"  sample {n}: {hot}/{len(slices)} experts seen resident",
                  flush=True)
            time.sleep(max(0.0, min(interval, t_end - time.time())))
    except KeyboardInterrupt:
        print("\n(observation interrupted; using what we have)")
    finally:
        obs.close()
    for k in counts:
        counts[k] /= max(n, 1)
    return counts


def report(counts, per_expert, budget_b):
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    seen = [v for _, v in ranked if v > 0]
    print(f"\nexperts seen at least once : {len(seen)} / {len(ranked)}")
    if not seen:
        print("No expert residency observed. Was the model running?")
        return ranked
    fits = int(budget_b // per_expert) if per_expert else 0
    top = ranked[:fits]
    covered = sum(v for _, v in top)
    total = sum(v for _, v in ranked)
    print(f"per-expert size            : {per_expert / MB:.1f} MB")
    print(f"budget holds               : {fits} experts")
    if total > 0:
        print(f"share of observed activity : {covered / total:.1%}  "
              f"<- expected hit rate from pinning these")
    print("\ntop 10 by popularity:")
    for (layer, e), v in ranked[:10]:
        print(f"  layer {layer:>3}  expert {e:>4}   score {v:.3f}")
    return ranked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--pattern")
    ap.add_argument("--observe", type=int, default=120, help="seconds to sample")
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--lock-gb", type=float, default=0.0)
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--save-stats"); ap.add_argument("--load-stats")
    ap.add_argument("--hold-seconds", type=int, default=0)
    ap.add_argument("--stop", action="store_true")
    args = ap.parse_args()

    if args.stop:
        if not os.path.exists(PIDFILE):
            print("no expert cache running"); return 0
        pid = int(open(PIDFILE).read().strip())
        try:
            os.kill(pid, signal.SIGTERM); print(f"stopped (pid {pid})")
        except ProcessLookupError:
            print("already gone")
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
        if os.environ.get("SUDO_USER") and not os.environ.get("HF_HUB_CACHE"):
            print("Under sudo the cache path is stripped: use sudo -E.", file=sys.stderr)
        return 1

    slices, n_experts, per_expert = expert_slices(files)
    if not slices:
        print("No routed-expert tensors found (is this an MoE GGUF?)", file=sys.stderr)
        return 1
    print(f"files    : {len(files)}")
    print(f"experts  : {len(slices)} ({n_experts} per layer)")
    print(f"per expert: {per_expert / MB:.1f} MB")

    if args.load_stats:
        raw = json.load(open(args.load_stats))
        counts = {(int(k.split(":")[0]), int(k.split(":")[1])): v
                  for k, v in raw.items()}
        print(f"loaded {len(counts)} scores from {args.load_stats}")
    else:
        if sys.platform != "linux":
            print("Observation needs Linux (mincore).", file=sys.stderr)
            return 1
        counts = observe(slices, files, args.observe, args.interval)

    if args.save_stats:
        json.dump({f"{l}:{e}": v for (l, e), v in counts.items()},
                  open(args.save_stats, "w"))
        print(f"saved scores -> {args.save_stats}")

    budget_b = args.lock_gb * GB
    ranked = report(counts, per_expert, budget_b)

    if args.report_only or args.lock_gb <= 0:
        return 0

    fits = int(budget_b // per_expert)
    chosen = [k for k, v in ranked[:fits] if v > 0]
    if not chosen:
        print("\nNothing worth pinning (no observed activity).", file=sys.stderr)
        return 1

    jobs = defaultdict(list)
    for key in chosen:
        for path, off, size in slices[key]:
            jobs[path].append((off, size))
    per_file = {p: coalesce(v) for p, v in jobs.items()}
    want_b = sum(sz for v in per_file.values() for _, sz in v)

    lim = memlock_limit()
    if lim is not None and lim != -1 and lim < want_b:
        explain_limit(lim, want_b)
        return 1

    from lock_hot import lock_files
    print(f"\npinning {len(chosen)} experts ({want_b / GB:.2f} GB)...")
    held = lock_files(per_file, want_b)
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
        print(f"holding for {args.hold_seconds}s...", flush=True)
        time.sleep(args.hold_seconds)
        release()
    print(f"holding (pid {os.getpid()}). Release with:")
    print("  python3 tools/expert_cache.py --stop")
    sys.stdout.flush()
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    sys.exit(main())
