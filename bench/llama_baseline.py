"""
Baseline: what does the best existing runtime do on this hardware?

llama.cpp mmaps the GGUF and lets the OS page cache decide which expert blocks
stay resident. That is a real expert cache -- just a routing-blind one, evicting
4 KB pages by LRU with no idea a router exists. Whatever number this produces is
what a routing-aware cache has to beat. If it cannot, the thesis is wrong.

  python bench/llama_baseline.py --list
  python bench/llama_baseline.py                     # full sweep
  python bench/llama_baseline.py --quick             # one config
  python bench/llama_baseline.py --generate          # real generation sample
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
IS_WIN = sys.platform == "win32"
EXE = ".exe" if IS_WIN else ""

# Where llama.cpp binaries live. Override with MINILLM_LLAMA_BIN.
#   Windows: the prebuilt zip extracted to D:\minillm\llamacpp
#   Linux  : ~/llama.cpp/build/bin after `cmake --build build`
LLAMA = os.environ.get("MINILLM_LLAMA_BIN") or (
    r"D:\minillm\llamacpp" if IS_WIN else os.path.expanduser("~/llama.cpp/build/bin"))

# Hugging Face cache. Honors HF_HUB_CACHE, then HF_HOME/hub, then a per-OS default.
HF_CACHE = os.environ.get("HF_HUB_CACHE") or (
    os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else
    (r"D:\minillm\hf\hub" if IS_WIN else os.path.expanduser("~/minillm/hf/hub")))


SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)


def find_gguf(pattern):
    """Find the GGUF to hand llama.cpp for a model matching `pattern`.

    A large quant is split into shards ("-00001-of-00015.gguf"); llama.cpp
    must be given the FIRST shard, which is often the smallest (metadata) --
    so "pick the largest match" is wrong for split models. Prefer shard 00001,
    then fall back to the largest single file.
    """
    single, first_shards = [], []
    for dp, _, fs in os.walk(HF_CACHE):
        for f in fs:
            if not (f.endswith(".gguf") and pattern.lower() in f.lower()):
                continue
            p = os.path.join(dp, f)
            m = SHARD_RE.search(f)
            if m:
                if int(m.group(1)) == 1:
                    first_shards.append(p)
            else:
                try:
                    single.append((os.path.getsize(p), p))
                except OSError:
                    pass
    if first_shards:
        return sorted(first_shards)[0]
    single.sort(reverse=True)
    return single[0][1] if single else None


# Per-machine defaults. Override with --model / --draft-model, or these env vars.
DEFAULT_MODEL_PATTERN = os.environ.get("MINILLM_MODEL_PATTERN") or (
    "gpt-oss-120b-MXFP4" if IS_WIN else "Qwen3.8-2.4T-A95B-UD-IQ2_XXS")
DEFAULT_DRAFT_PATTERN = os.environ.get("MINILLM_DRAFT_PATTERN") or (
    "eagle3" if IS_WIN else "MTP-ONLY")


def gb(n):
    return n / 1024 ** 3


def total_ram_gb():
    """Physical RAM in GB, or None if it cannot be determined. No psutil dep."""
    try:
        if IS_WIN:
            import ctypes
            class MEMSTAT(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = MEMSTAT(); s.dwLength = ctypes.sizeof(MEMSTAT)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return gb(s.ullTotalPhys)
        return gb(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def run(cmd, timeout, stdin_eof=False):
    t0 = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                           errors="replace",
                           stdin=subprocess.DEVNULL if stdin_eof else None)
        return p.returncode, (p.stdout or "") + (p.stderr or ""), time.perf_counter() - t0
    except subprocess.TimeoutExpired as e:
        got = (e.stdout or b"")
        if isinstance(got, bytes):
            got = got.decode("utf-8", "replace")
        return -9, got + "\n[TIMEOUT]", time.perf_counter() - t0


TG_RE = re.compile(r"\|\s*tg(\d+)\s*\|\s*([0-9.]+)\s*±", re.I)
PP_RE = re.compile(r"\|\s*pp(\d+)\s*\|\s*([0-9.]+)\s*±", re.I)

# llama-cli in build b10437+ is a chat frontend on top of llama-server. It does
# NOT print the classic "eval time = ... tokens per second" line -- that only
# appears with --perf, which defaults to off. What it prints after each turn is:
#     [ Prompt: 0.1 t/s | Generation: 0.0 t/s ]
# and then drops to an interactive ">" prompt even with -no-cnv, so a scripted
# run must feed it EOF on stdin or it will block forever.
CLI_RATE_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")


def parse_bench(out):
    """llama-bench prints one markdown table row per test:
         | model | size | params | backend | threads | [lm] | test | t/s |
    The optional `lm` column appears when --load-mode was passed. Match on the
    test name and the ± figure so column count does not matter."""
    res = {}
    for m in PP_RE.finditer(out):
        res[f"pp{m.group(1)}"] = float(m.group(2))
    for m in TG_RE.finditer(out):
        res[f"tg{m.group(1)}"] = float(m.group(2))
    return res


def cmd_bench(args, model):
    exe = os.path.join(LLAMA, "llama-bench" + EXE)
    configs = ([{"t": args.threads, "n": args.n_gen, "p": args.n_prompt}] if args.quick else
               [{"t": 2, "n": args.n_gen, "p": args.n_prompt},
                {"t": 4, "n": args.n_gen, "p": args.n_prompt},
                {"t": 8, "n": args.n_gen, "p": args.n_prompt}])

    # llama-bench (b10437) ALWAYS repacks: it has no --no-repack flag and it
    # ignores LLAMA_ARG_REPACK. Repacking needs a real allocation of the whole
    # model, so llama-bench cannot run anything larger than RAM. Verified:
    #   done_getting_tensors: ... cannot be used with preferred buffer type CPU_REPACK
    #   alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 60914073600
    # It is still the right tool for models that DO fit (clean table output).
    # For models larger than RAM use --generate (llama-cli honors --no-repack).
    model_gb = gb(os.path.getsize(model))
    ram_gb = total_ram_gb()
    if ram_gb and model_gb > ram_gb * 0.85:
        print(f"REFUSING llama-bench: model is {model_gb:.1f} GB, RAM is {ram_gb:.1f} GB.")
        print(f"llama-bench always repacks (no --no-repack) and will fail to allocate.")
        print(f"Use --generate, which runs llama-cli --no-repack --perf instead.")
        return [{"error": "model larger than RAM; llama-bench cannot run it",
                 "model_gb": model_gb, "ram_gb": ram_gb}]
    rows = []
    for c in configs:
        cmd = [exe, "-m", model, "-t", str(c["t"]), "-p", str(c["p"]),
               "-n", str(c["n"]), "-r", str(args.reps), "-ngl", "0"]
        if args.load_mode:
            cmd += ["-lm", args.load_mode]
        print(f"\n>>> threads={c['t']} pp={c['p']} tg={c['n']} reps={args.reps} lm={args.load_mode}")
        print(f"    {' '.join(cmd)}")
        rc, out, dt = run(cmd, args.timeout)
        parsed = parse_bench(out)
        print(f"    exit {rc} in {dt/60:.1f} min -> {parsed or 'no numbers parsed'}")
        if not parsed:
            tail = "\n".join(out.strip().splitlines()[-12:])
            print(f"    --- tail ---\n{tail}")
        rows.append({"threads": c["t"], "n_prompt": c["p"], "n_gen": c["n"],
                     "load_mode": args.load_mode, "exit": rc, "wall_s": dt,
                     "parsed": parsed, "tail": out[-3000:]})
    return rows


def cmd_generate(args, model):
    """Real end-to-end generation via llama-cli. Prefer `--bench` (llama-bench)
    for the tok/s number: it is purpose-built, prints a stable table, and has
    no chat mode to fall into. This mode exists to confirm the model actually
    produces sensible text, and to exercise speculative decoding."""
    exe = os.path.join(LLAMA, "llama-cli" + EXE)
    prompt = args.prompt
    cmd = [exe, "-m", model, "-t", str(args.threads), "-n", str(args.n_gen),
           "-c", str(args.ctx), "-ngl", "0", "--no-warmup", "-p", prompt,
           "--perf"]  # without --perf there is no per-token timing at all
    # Repacking rewrites quantized weights into a SIMD-friendly layout, which
    # needs a full in-RAM copy of the model and defeats mmap. Off by default.
    if not args.repack:
        cmd.append("--no-repack")
    if args.load_mode:
        cmd += ["--load-mode", args.load_mode]

    draft_path = None
    if args.draft or args.draft_model:
        draft_path = args.draft_model or find_gguf(DEFAULT_DRAFT_PATTERN)
        if not draft_path:
            print(f"ERROR: --draft requested but no GGUF matching "
                  f"'{DEFAULT_DRAFT_PATTERN}' found under {HF_CACHE}.")
            print("       Pass --draft-model PATH, or set MINILLM_DRAFT_PATTERN.")
            return {"mode": "generate", "exit": 2, "error": "draft model not found"}
        # b10437 flag names. --draft-max/--draft-min were REMOVED upstream.
        cmd += ["-md", draft_path, "--spec-draft-n-max", "4", "--spec-draft-n-min", "1"]
        print(f"    draft model: {draft_path}")

    print(f">>> generating {args.n_gen} tokens, threads={args.threads}, ctx={args.ctx}")
    # This build's llama-cli drops into an interactive ">" prompt after the
    # response even with -no-cnv. Closing stdin makes it exit cleanly instead
    # of hanging until timeout.
    rc, out, dt = run(cmd, args.timeout, stdin_eof=True)

    m = CLI_RATE_RE.search(out)
    pp_rate = float(m.group(1)) if m else None
    tg_rate = float(m.group(2)) if m else None
    print(f"    exit {rc} in {dt/60:.1f} min")
    if m:
        print(f"    prompt {pp_rate:.2f} t/s | generation {tg_rate:.2f} t/s")
        if tg_rate == 0.0:
            print("    (llama-cli rounds to 1 decimal; 0.0 means < 0.05 t/s -- "
                  "use --bench for a precise figure)")
    else:
        print("    no '[ Prompt: X t/s | Generation: Y t/s ]' line found; tail:")
        print("\n".join(out.strip().splitlines()[-20:]))
    return {"mode": "generate", "threads": args.threads, "ctx": args.ctx,
            "n_gen": args.n_gen, "draft_model": draft_path, "exit": rc,
            "wall_s": dt, "prompt_tok_s": pp_rate, "gen_tok_s": tg_rate,
            "tail": out[-4000:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--n-gen", type=int, default=16)
    ap.add_argument("--n-prompt", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=5400)
    ap.add_argument("--quick", action="store_true",
                    help="bench: one thread config instead of the 2/4/8 sweep")
    ap.add_argument("--repack", action="store_true",
                    help="generate: allow weight repacking (needs a full in-RAM copy)")
    ap.add_argument("--load-mode", default="mmap",
                    choices=["auto", "mmap", "mlock", "mmap+mlock", "dio"],
                    help="mmap streams via the page cache. NOTE: dio and mlock "
                         "load the whole model into RAM and cannot run a model "
                         "larger than RAM -- they are not a 'no cache' bracket.")
    ap.add_argument("--generate", action="store_true",
                    help="run llama-cli for real text (default: llama-bench for tok/s)")
    ap.add_argument("--draft", action="store_true",
                    help=f"speculative decoding; finds a GGUF matching "
                         f"'{DEFAULT_DRAFT_PATTERN}' unless --draft-model is given")
    ap.add_argument("--draft-model", default=None, help="explicit draft GGUF path")
    ap.add_argument("--pattern", default=DEFAULT_MODEL_PATTERN,
                    help="substring to locate the model GGUF when --model is not given")
    ap.add_argument("--prompt", default="Explain why mixture-of-experts models "
                                        "are easier to run on small machines than dense models.")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for dp, _, fs in os.walk(HF_CACHE):
            for f in fs:
                if f.endswith(".gguf"):
                    p = os.path.join(dp, f)
                    print(f"  {gb(os.path.getsize(p)):8.2f} GB  {p}")
        return 0

    model = args.model or find_gguf(args.pattern)
    if not model:
        print(f"No GGUF matching '{args.pattern}' found under {HF_CACHE}.")
        print("Still downloading? Use --list to see what is present, or pass --model PATH.")
        return 1

    exe_check = os.path.join(LLAMA, ("llama-cli" if args.generate else "llama-bench") + EXE)
    if not os.path.exists(exe_check):
        print(f"llama.cpp binary not found: {exe_check}")
        print("Set MINILLM_LLAMA_BIN to the directory containing llama-cli / llama-bench.")
        return 1

    size = os.path.getsize(model)
    print(f"model : {model}")
    print(f"size  : {gb(size):.2f} GB")
    print(f"llama : {LLAMA}")
    print(f"NOTE  : if the model is larger than RAM, every token pages experts")
    print(f"        from disk. First token is cold; later tokens are the number.\n")

    os.makedirs(RESULTS, exist_ok=True)
    out = (cmd_generate(args, model) if args.generate else
           {"mode": "bench", "rows": cmd_bench(args, model)})
    out["model"] = model
    out["model_bytes"] = size

    # Tag by hostname so laptop and server results coexist in the same tree.
    import platform
    host = platform.node().split(".")[0].lower() or "unknown"
    out["host"] = host
    out["platform"] = sys.platform
    path = os.path.join(RESULTS, f"baseline_{'gen' if args.generate else 'bench'}_{host}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
