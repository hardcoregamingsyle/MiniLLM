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

  # ACCUMULATE across sessions -- run this a few times over a few evenings.
  # Each round pins better experts, which speeds generation, which means the
  # next round observes more tokens in the same wall-clock time.
  sudo -E python3 tools/expert_cache.py --pattern Qwen3.8-2.4T \n       --merge-stats ~/qwen-experts.json --observe 14400 --lock-gb 7

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
from lock_hot import (libc, coalesce, raise_memlock, warn_limit,  # noqa: E402
                      vmlck_bytes, unlock_all, PAGE, PROT_READ,
                      MAP_SHARED, MAP_FAILED, GB, MB)

PIDFILE = "/tmp/minillm-expert-cache.pid"
# Only the weight tensors are worth caching; the *_exps.bias tensors are tiny
# and get pulled in with the hot set anyway.
WEIGHT_SUFFIX = ".weight"
# Pages probed per expert per sample. 16 pages = 64 KB is plenty to tell
# "this expert was recently read" from "it was not", and keeps a full sample
# of 47k experts at ~0.1 s instead of 3-5 s.
PROBE_PAGES = 16


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
        """Is this expert currently in the page cache? Sampled, not exhaustive.

        Checking every page of every expert costs 150 M Python operations per
        sample on a 2.4T model -- 3-5 s of CPU, which on a 4-core laptop steals
        a whole core from inference and makes the thing we are measuring
        slower. An expert is read as a unit, so a small probe window answers
        the same question ~200x cheaper.
        """
        for p, addr, msize in self.maps:
            if p != path:
                continue
            # Probe a third of the way in: avoids readahead spill from the
            # previous tensor at the start and truncation at the end.
            probe_off = (off + size // 3) & ~(PAGE - 1)
            npages = min(PROBE_PAGES, max(1, size // PAGE))
            length = npages * PAGE
            if probe_off + length > off + size:
                probe_off = (off + size - length) & ~(PAGE - 1)
            vec = (ctypes.c_ubyte * npages)()
            if libc().mincore(ctypes.c_void_p(addr + probe_off), length, vec) != 0:
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
          f"(sample every {interval}s, {PROBE_PAGES} pages probed per expert)...")
    print("Run the model NOW in another terminal so there is routing to learn from.")
    print("NOTE: this steals some CPU. Do not trust a tok/s measured while it runs --")
    print("      measure clean afterwards, or run it under `nice -n 19`.")
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
    # Return RAW sums plus the sample count, not an average. Keeping them
    # separate is what lets several sessions merge correctly: averages cannot
    # be re-averaged without their weights.
    return counts, n


def load_stats(path):
    """-> (raw counts, n_samples). Accepts the old flat {key: score} format."""
    d = json.load(open(path))
    if isinstance(d, dict) and "counts" in d and "samples" in d:
        raw, n = d["counts"], int(d["samples"])
    else:                                   # legacy: scores were pre-averaged
        raw, n = d, 1
    return ({(int(k.split(":")[0]), int(k.split(":")[1])): float(v)
             for k, v in raw.items()}, n)


def save_stats(path, counts, n):
    json.dump({"samples": n,
               "counts": {f"{l}:{e}": v for (l, e), v in counts.items()}},
              open(path, "w"))


def merge_stats(a, na, b, nb):
    """Combine two observation sessions by weight, not by averaging averages."""
    out = defaultdict(float)
    for k, v in a.items():
        out[k] += v
    for k, v in b.items():
        out[k] += v
    return out, na + nb


def report(counts, per_expert, budget_b):
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    seen = [v for _, v in ranked if v > 0]
    print(f"\nexperts seen at least once : {len(seen)} / {len(ranked)}")
    if not seen:
        print("No expert residency observed. Was the model running?")
        return ranked

    # Residency reveals routing PREFERENCE only when the page cache is under
    # pressure. If the whole model fits in RAM every page stays resident, every
    # score saturates at 1.0, and there is no skew to learn -- pinning would buy
    # nothing because nothing is being evicted in the first place.
    saturated = sum(1 for v in seen if v > 0.98) / len(ranked)
    if saturated > 0.9:
        print(f"\nNOTE: {saturated:.0%} of experts are ~always resident, so the page")
        print("cache is NOT under pressure -- the model fits in RAM here. There is")
        print("no skew to learn and nothing to gain from pinning. This tool is for")
        print("the case where the model is much larger than RAM.")
        return ranked
    spread = (max(seen) - min(seen)) if len(seen) > 1 else 0.0
    if spread < 0.05:
        print(f"\nNOTE: popularity is nearly flat (spread {spread:.3f}). Either the")
        print("sample is too short or routing is close to uniform; pinning the top")
        print("experts would be little better than pinning random ones. Observe")
        print("across more generated tokens (--observe).")
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
    ap.add_argument("--save-stats", help="write the histogram here")
    ap.add_argument("--load-stats", help="use a saved histogram, skip observing")
    ap.add_argument("--merge-stats", metavar="FILE",
                    help="accumulate: load FILE if present, observe, add, save back. "
                         "Run this repeatedly to build a histogram over several "
                         "short sessions instead of one marathon.")
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

    prior, prior_n = {}, 0
    stats_path = args.merge_stats or args.load_stats
    if stats_path and os.path.exists(stats_path):
        prior, prior_n = load_stats(stats_path)
        print(f"loaded {len(prior)} scores ({prior_n} prior samples) from {stats_path}")

    if args.load_stats:
        counts, n_samples = prior, max(prior_n, 1)
        if not counts:
            print(f"{args.load_stats} not found or empty", file=sys.stderr)
            return 1
    else:
        if sys.platform != "linux":
            print("Observation needs Linux (mincore).", file=sys.stderr)
            return 1
        fresh, fresh_n = observe(slices, files, args.observe, args.interval)
        if prior:
            counts, n_samples = merge_stats(prior, prior_n, fresh, fresh_n)
            print(f"merged: {prior_n} prior + {fresh_n} new = {n_samples} samples")
        else:
            counts, n_samples = fresh, fresh_n

    out_path = args.merge_stats or args.save_stats
    if out_path:
        save_stats(out_path, counts, n_samples)
        print(f"saved {len(counts)} scores ({n_samples} samples) -> {out_path}")

    # Report on averages; the file keeps raw sums so sessions can accumulate.
    counts = {k: v / max(n_samples, 1) for k, v in counts.items()}

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

    # Ask the kernel, do not pre-judge: a root process has CAP_IPC_LOCK and
    # mlock() then ignores RLIMIT_MEMLOCK entirely. Refusing here on the
    # strength of the reported limit skipped locks that would have worked.
    lim = raise_memlock(want_b)
    if lim is not None and lim != -1 and lim < want_b:
        warn_limit(lim, want_b)

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
