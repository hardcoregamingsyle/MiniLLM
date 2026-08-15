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

# Which binary does generation. In build b10437+ `llama-cli` became a chat
# frontend on top of llama-server: it ignores -no-cnv, prints only a rounded
#     [ Prompt: 0.1 t/s | Generation: 0.0 t/s ]
# summary, and on EOF from stdin it prints ">" and loops FOREVER (46k prompt
# lines in one log here) instead of exiting. It is unusable for scripting.
# `llama-completion` is the classic non-chat binary: honors -no-cnv and
# --no-repack, exits when done, and with --perf prints the real breakdown:
#     llama_perf_context_print: prompt eval time = ... / N tokens ( ... ms per token, X tokens per second)
#     llama_perf_context_print:        eval time = ... / N runs   ( ... ms per token, X tokens per second)
GEN_BIN = "llama-completion"
CLI_RATE_RE = re.compile(r"\[\s*Prompt:\s*([0-9.]+)\s*t/s\s*\|\s*Generation:\s*([0-9.]+)\s*t/s\s*\]")
PERF_PROMPT_RE = re.compile(r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens.*?([0-9.]+)\s*tokens per second", re.I)
PERF_EVAL_RE = re.compile(r"(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs.*?([0-9.]+)\s*tokens per second", re.I)
PERF_LOAD_RE = re.compile(r"load time\s*=\s*([0-9.]+)\s*ms", re.I)


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
    # For models larger than RAM use --generate (llama-completion honors --no-repack).
    model_gb = gb(os.path.getsize(model))
    ram_gb = total_ram_gb()
    if ram_gb and model_gb > ram_gb * 0.85:
        print(f"REFUSING llama-bench: model is {model_gb:.1f} GB, RAM is {ram_gb:.1f} GB.")
        print(f"llama-bench always repacks (no --no-repack) and will fail to allocate.")
        print(f"Use --generate, which runs llama-completion --no-repack --perf instead.")
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
    """End-to-end generation via llama-completion, with --perf timings.

    For a model larger than RAM this is THE measurement on b10437: llama-bench
    cannot run it (unconditional repack) and llama-cli cannot be scripted (chat
    loop). llama-completion is the classic binary and does both correctly."""
    # Speculative decoding lives only in llama-cli / llama-server on b10437;
    # llama-completion has none of the --spec-* flags. llama-cli normally loops
    # forever on stdin EOF, but -st (single-turn) makes it answer once and exit
    # -- verified. So: draft runs use llama-cli -st, plain runs llama-completion.
    use_draft = bool(args.draft or args.draft_model)
    binary = "llama-cli" if use_draft else GEN_BIN
    exe = os.path.join(LLAMA, binary + EXE)
    prompt = args.prompt
    cmd = [exe, "-m", model, "-t", str(args.threads), "-n", str(args.n_gen),
           "-c", str(args.ctx), "-ngl", "0", "--no-warmup", "-p", prompt,
           "--perf"]  # --perf is off by default; no timings without it
    cmd += ["-st"] if use_draft else ["-no-cnv"]
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
        # An MTP-ONLY GGUF (a4lg/*-MTP-ONLY-GGUF) is not a standalone draft
        # model -- it is the target model's own multi-token-prediction head,
        # and llama.cpp must be told so or it will try to run it as a full
        # model and fail. Per the a4lg README: --spec-type draft-mtp.
        # A conventional draft (EAGLE3, a small full model) must NOT get this.
        if "mtp" in os.path.basename(draft_path).lower():
            cmd += ["--spec-type", "draft-mtp"]
        print(f"    draft model: {draft_path}")

    print(f">>> generating {args.n_gen} tokens, threads={args.threads}, ctx={args.ctx}")
    # stdin closed defensively: llama-completion exits on its own, but if
    # someone points GEN_BIN at llama-cli this at least avoids a blocking read.
    rc, out, dt = run(cmd, args.timeout, stdin_eof=True)

    # Prefer the exact --perf breakdown; fall back to llama-cli's rounded summary.
    pp = PERF_PROMPT_RE.search(out)
    ev = PERF_EVAL_RE.search(out)
    ld = PERF_LOAD_RE.search(out)
    summary = CLI_RATE_RE.search(out)

    load_s = float(ld.group(1)) / 1000 if ld else None
    pp_rate = float(pp.group(3)) if pp else (float(summary.group(1)) if summary else None)
    pp_n = int(pp.group(2)) if pp else None
    tg_rate = float(ev.group(3)) if ev else (float(summary.group(2)) if summary else None)
    tg_n = int(ev.group(2)) if ev else None
    tg_ms = float(ev.group(1)) / tg_n if (ev and tg_n) else None

    print(f"    exit {rc} in {dt/60:.1f} min")
    if load_s is not None:
        print(f"    load          : {load_s:8.1f} s   (mmap + first-touch of the hot set)")
    if pp_rate is not None:
        n = f" over {pp_n} tokens" if pp_n else ""
        print(f"    prompt eval   : {pp_rate:8.3f} tok/s{n}")
    if tg_rate is not None:
        n = f" over {tg_n} tokens" if tg_n else ""
        per = f"  ({tg_ms/1000:.2f} s/token)" if tg_ms else ""
        print(f"    GENERATION    : {tg_rate:8.3f} tok/s{n}{per}   <-- the number")
    if pp_rate is None and tg_rate is None:
        print("    no perf timings found in output; tail:")
        print("\n".join(out.strip().splitlines()[-20:]))
    return {"mode": "generate", "binary": binary, "threads": args.threads,
            "ctx": args.ctx, "n_gen": args.n_gen, "draft_model": draft_path,
            "exit": rc, "wall_s": dt, "load_s": load_s,
            "prompt_tok_s": pp_rate, "prompt_n": pp_n,
            "gen_tok_s": tg_rate, "gen_n": tg_n, "gen_ms_per_tok": tg_ms,
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
                    help="run llama-completion with --perf (the measurement for models > RAM)")
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

    exe_check = os.path.join(LLAMA, (GEN_BIN if args.generate else "llama-bench") + EXE)
    if not os.path.exists(exe_check):
        print(f"llama.cpp binary not found: {exe_check}")
        print("Set MINILLM_LLAMA_BIN to the directory containing llama-completion / llama-bench.")
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
