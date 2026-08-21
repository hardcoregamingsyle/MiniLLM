"""
Record and attribute kernel I/O counters across a llama.cpp run.

Why this exists: a run has three phases -- load, prompt eval, generation --
and only the third one matters for tokens/second. llama.cpp reports the TIME
of each phase separately but says nothing about BYTES, and the whole question
on a machine smaller than the model is how many bytes each token costs. A
whole-run byte count mixes in the 7-minute load and answers nothing.

So: sample /proc/vmstat on a timer during the run, then slice the trace using
the phase durations llama.cpp itself printed. The counters are:

  pgpgin      kilobytes paged in from block devices (system-wide, exact)
  pgmajfault  faults that had to go to disk; bytes/fault reveals readahead
  pswpin      pages read back from SWAP -- not model bytes, pure waste

  python3 tools/iotrace.py sample results/io_trace.tsv --hz 4
  python3 tools/iotrace.py report results/io_trace.tsv results/run.log
"""

import argparse
import os
import re
import signal
import sys
import time

GB = 1024 ** 3
KEYS = ("pgpgin", "pgmajfault", "pswpin")


def read_vmstat():
    out = {}
    try:
        with open("/proc/vmstat") as f:
            for line in f:
                k, _, v = line.partition(" ")
                if k in KEYS:
                    out[k] = int(v)
    except OSError:
        pass
    return tuple(out.get(k, 0) for k in KEYS)


def cmd_sample(args):
    """Write 'epoch pgpgin pgmajfault pswpin' until killed."""
    stop = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.append(1))
    period = 1.0 / max(args.hz, 0.1)
    os.makedirs(os.path.dirname(os.path.abspath(args.path)) or ".", exist_ok=True)
    with open(args.path, "w", buffering=1) as f:
        f.write("# epoch\t" + "\t".join(KEYS) + "\n")
        nxt = time.time()
        while not stop:
            f.write("%.3f\t%d\t%d\t%d\n" % ((time.time(),) + read_vmstat()))
            nxt += period
            time.sleep(max(0.0, nxt - time.time()))
    return 0


def load_trace(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split()
            if len(p) == 4:
                rows.append((float(p[0]), int(p[1]), int(p[2]), int(p[3])))
    return rows


def window(rows, t_start, t_end):
    """Counter deltas over [t_start, t_end], using the samples that bracket it."""
    inside = [r for r in rows if t_start <= r[0] <= t_end]
    if len(inside) < 2:
        return None
    a, b = inside[0], inside[-1]
    return (b[0] - a[0], (b[1] - a[1]) * 1024, b[2] - a[2], (b[3] - a[3]) * 4096)


# llama.cpp --perf, e.g. "eval time =  236422.77 ms /     3 runs"
# Not anchored: llama.cpp prefixes every line with "llama_perf_context_print:".
# "prompt eval time" must precede "eval time" in the alternation or the longer
# name is never reached.
PERF_RE = re.compile(
    r"(load time|prompt eval time|eval time)\s*=\s*([\d.]+)\s*ms"
    r"(?:\s*/\s*(\d+)\s*(?:tokens|runs))?")


def parse_perf(log_text):
    phases = {}
    for m in PERF_RE.finditer(log_text):
        name = " ".join(m.group(1).split())
        phases[name] = (float(m.group(2)) / 1000.0,
                        int(m.group(3)) if m.group(3) else 0)
    return phases


def show(label, w, ntok, note=""):
    if not w:
        print(f"  {label:<16} (no samples in window)")
        return
    dur, read_b, faults, swap_b = w
    rate = read_b / GB / dur if dur > 0 else 0.0
    print(f"  {label:<16} {dur:7.1f} s   {read_b / GB:7.2f} GB   {rate:6.3f} GB/s", end="")
    if ntok:
        print(f"   {read_b / GB / ntok:7.2f} GB/tok   {dur / ntok:7.2f} s/tok", end="")
    print(f"  {note}")
    if faults > 0:
        kb = read_b / faults / 1024
        flag = "  <- readahead is doing NOTHING" if kb < 16 else ""
        print(f"  {'':<16} {faults:>9,} major faults = {kb:.1f} kB/fault{flag}")
    if swap_b > 0:
        print(f"  {'':<16} {swap_b / GB:7.2f} GB SWAPPED IN <- wasted, not model bytes")


def cmd_report(args):
    rows = load_trace(args.trace)
    if len(rows) < 2:
        print("trace too short -- was the sampler running?", file=sys.stderr)
        return 1
    perf = parse_perf(open(args.log, encoding="utf-8", errors="replace").read())
    if not perf:
        print("no --perf block in the log; showing whole-run totals only\n",
              file=sys.stderr)

    t_start, t_end = rows[0][0], rows[-1][0]
    print("\n---------------- I/O by phase ----------------")
    print(f"  {'phase':<16} {'time':>7}     {'read':>7}      {'rate':>6}"
          f"      {'per token':>9}")

    # The run ends with generation, so walk the phases backwards from t_end.
    gen_s, gen_n = perf.get("eval time", (0.0, 0))
    pp_s, pp_n = perf.get("prompt eval time", (0.0, 0))
    if gen_s > 0:
        g0 = t_end - gen_s
        show("generation", window(rows, g0, t_end), gen_n, "<- THE number")
        if pp_s > 0:
            # llama.cpp's "load time" overlaps prompt eval, so bound the prompt
            # window by its own duration ending where generation began.
            show("prompt eval", window(rows, max(t_start, g0 - pp_s), g0), pp_n)
            show("load", window(rows, t_start, max(t_start, g0 - pp_s)), 0)
    show("WHOLE RUN", window(rows, t_start, t_end), 0)

    if gen_s > 0 and gen_n:
        w = window(rows, t_end - gen_s, t_end)
        if w:
            _, read_b, _, _ = w
            print(f"\n  Generation costs {read_b / GB / gen_n:.2f} GB per token at "
                  f"{read_b / GB / gen_s:.3f} GB/s.")
            print("  Cut either number and the token rate moves. Nothing else matters.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="record counters until killed")
    s.add_argument("path")
    s.add_argument("--hz", type=float, default=4.0)
    s.set_defaults(fn=cmd_sample)
    r = sub.add_parser("report", help="attribute the trace to llama.cpp phases")
    r.add_argument("trace")
    r.add_argument("log")
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
