# MiniLLM — running trillion-parameter MoE models on machines with 8–32 GB of RAM

[![ci](https://github.com/hardcoregamingsyle/MiniLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/hardcoregamingsyle/MiniLLM/actions/workflows/ci.yml)

CPU inference for Mixture-of-Experts models far larger than RAM, streaming
experts from disk. The bet: **a router-aware expert cache beats the OS page
cache**, because the OS evicts 4 KB pages by LRU and has no idea what a router is.

## Quick start (Linux Mint / Ubuntu / Debian)

One command does the whole setup — packages, clone, Python env, HF token,
llama.cpp built for your CPU, hardware calibration, and the model download
(250 GB default, 680 GB for the 2.4T) started in a `tmux` session. **Every push runs this exact installer
on a fresh Ubuntu VM** (see the badge): cold install → re-run and assert
nothing changed → build llama.cpp → calibrate → fetch a 12 GB MoE → prove
`run.sh`, `--draft`, and the harness end-to-end. Green means the commands
below work on a machine nobody has touched.

```bash
curl -fsSL https://raw.githubusercontent.com/hardcoregamingsyle/MiniLLM/main/install.sh | bash
```

It asks before anything needing `sudo`, is **safe to re-run** (each step skips
itself if already done, the download resumes), and stops with a clear message
on the first failure. Then, once the download finishes:

```bash
cd ~/MiniLLM
bash run.sh                # generate 32 tokens, print tok/s
bash run.sh --draft        # same, with MTP speculative decoding
bash run.sh --chat         # talk to the model
bash ~/minillm/download.sh qwen3.8-2.4t   # fetch the other model too (they fit side by side)
```

Options are env vars: `MINILLM_MODEL=qwen3.8-2.4t` (the 2.4T instead of the
default 397B), `MINILLM_DATA=/mnt/big` (where the model goes), `MINILLM_QUANT=...`,
`MINILLM_SKIP_DOWNLOAD=1`,
`MINILLM_YES=1` (unattended). Prefer to see every step? The same sequence,
explained, is in **[SETUP_LINUX.md](SETUP_LINUX.md)**.

`run.sh` knows things that cost a day to learn: it uses `llama-completion`
(not `llama-cli`, which is now a chat loop, nor `llama-bench`, which cannot run
a model larger than RAM), passes `--no-repack` (mandatory above RAM, not the
default), finds shard `00001` of a split GGUF through the Hugging Face cache's
symlinks, and refuses to start until the last shard exists.

## What is in here

| Path | What it does |
| --- | --- |
| **`install.sh`** | One-shot Linux setup: packages → clone → venv → token → build llama.cpp → calibrate → start download. Idempotent; `MINILLM_MODEL` picks 397B (default) or 2.4T. |
| **`run.sh`** | Day-to-day runner: finds the model, picks the right binary and flags, runs it. `--draft`, `--chat`. |
| `bench/roofline.py` | Measures DRAM, disk, and dequant bandwidth on any machine. Windows + Linux. |
| `minillm/capacity.py` | Predicts tok/s for any (machine × model × quantization). `report`, `solve`, `frontier`. |
| `minillm/machines.json` | Machine profiles — paste roofline output here to add a machine. |
| `minillm/models.json` | Parameter decomposition of each MoE (hot vs streamable). |
| `scripts/derive_spec.py` | Decomposes any MoE from `config.json` alone — no weight download. |
| `scripts/fetch_model.py` | Downloads to the big drive, skips redundant weight copies, resumes. |
| `sim/cache_sim.py` | Simulates cache policies (LRU / LFU / Belady) against routing traces. |
| `bench/llama_baseline.py` | Drives llama.cpp for the baseline tok/s measurement. |
| `ARCHITECTURE.md` | The full derivation from Qwen3.8-2.4T's `config.json`. |

The current state is **baseline measured, runtime not started**. gpt-oss-120b
runs correctly on the 8 GB laptop at 0.05–0.08 tok/s, and the profile of *why*
it is that slow (page-fault-bound mmap, three idle cores) is the specification
for the first thing `runtime/` should build. See "The baseline result" below.

## Measured roofline (the 8 GB dev laptop)

Run `python bench/roofline.py` to regenerate. Measured on i5-8350U / 8 GB
single-channel DDR4-2400 / SATA SSD:

| Constant | Measured |
| --- | --- |
| DRAM streaming read (1 thread) | 12.0 GB/s |
| DRAM streaming read (4 threads, saturated) | **15.3 GB/s** |
| SATA read, sequential QD1 | 0.48 GB/s |
| SATA read, random 256 KB QD4 | **0.51 GB/s** |
| Disk : DRAM ratio | **30x** |
| SIMD available | AVX2 + FMA (no AVX-512, no VNNI, no AMX) |
| RAM usable for weights | ~4 GB (8 GB − OS − KV/activations) |
| Free disk | 80 GB (D:) + 9 GB (C:) |

Two findings that change the design:

1. **Storage saturates at queue depth 4, and 256 KB blocks already reach full
   bandwidth.** The device is the wall, not the syscall path. Kernel-bypass I/O
   (`io_uring`/SQPOLL, HugePages, DMA-to-userspace) buys ~0% here — four threads
   issuing ordinary overlapped reads hit the SATA III ceiling. That machinery
   pays off at 7 GB/s NVMe, not at 0.5 GB/s SATA.
2. **Expert blocks can be small without penalty.** 256 KB at QD4 = 64 MB at QD8.
   This frees the layout to store experts individually rather than in fat
   coalesced chunks, which makes eviction much finer-grained.

## The token-time equation

With I/O overlapped against compute:

```
t_token  =  max( active_bytes / 15.3 GB/s ,  missed_bytes / 0.51 GB/s )
```

At 0.5 s/token the budget is **7.6 GB from RAM** or **260 MB from disk**. Because
disk is 30x slower, the entire design problem reduces to one number: *how few
bytes can we make miss the expert cache*.

## Model selection

The metric that decides everything is **active-routed parameters**, not total.
Total is bounded by disk; active-routed is bounded by RAM and disk bandwidth, and
it is the one that bites. Derived exactly by `scripts/derive_spec.py` using the
identity `hot = published_active - active_routed`:

| Model | Total | Active-routed | Hot | Pool |
| --- | --- | --- | --- | --- |
| gpt-oss-120b | 116.8 B | **3.58 B** | 1.52 B | 114.7 B |
| Qwen3.5-397B-A17B | 397 B | **7.55 B** | 9.45 B | 386.5 B |
| DeepSeek-V3 | 671 B | 20.43 B | 16.57 B | 653.9 B |
| Kimi-K2 | 1.04 T | 21.14 B | 10.86 B | 1014.7 B |
| Qwen3.8-2.4T-A95B | 2.42 T | 46.31 B | 48.69 B | 2370.8 B |

### Laptop (8 GB / 80 GB): `openai/gpt-oss-120b`

A model must satisfy **both** total ≤ ~162 B (disk at 4.25 bpw) **and**
active-routed ≤ ~5 B (the cliff where per-token demand outgrows the RAM cache).
gpt-oss-120b is the largest model found meeting both — 4-of-128 routing over
narrow 2880-wide experts. Predicted **0.96 tok/s at native MXFP4**, needing a
**71.3% expert-cache hit rate**, with the capacity ceiling at 100% — so it is
purely policy-bound. That is exactly the regime this project exists to test.

### Server (32 GB / 920 GB, i5-10210U, PCIe 3.0): `Qwen/Qwen3.5-397B-A17B`, not the 2.4T

The target is an **i5-10210U** — 4 cores AVX2, 15 W, and critically **PCIe 3.0
only**, so the NVMe caps at ~3 GB/s. `capacity.py` with those constants
(`kernel_gbps` taken from a CI runner with the same core count and SIMD,
which measured 21.9):

| Model / quant | Disk | Single | + MTP | Verdict |
| --- | --- | --- | --- | --- |
| **Qwen3.5-397B UD-Q4_K_XL** | **245 GB** | **1.6** | **2.8** | **clears the 2 tok/s aim** |
| Qwen3.8-2.4T UD-IQ2_XXS | 656 GB | 0.72 | 0.48 | misses the 1 tok/s floor |
| Qwen3.8-2.4T UD-IQ3_XXS | 956 GB | 0.36 | — | **does not fit** (36 GB over) |
| Qwen3.8-2.4T UD-IQ4_XS | 1,311 GB | 0.13 | — | **does not fit** (411 GB over) |

Two things decide it, both physical. **A 3-bit or 4-bit 2.4T does not fit on
920 GB** — the sizes above are read from the Hub, not estimated. And **on this
laptop, more bits on the 2.4T make it slower**: the bottleneck is the PCIe 3.0
disk feeding 46 B of routed-expert params per token, and every extra bit is
more bytes through that pipe. IQ2_XXS is both the largest 2.4T that fits and
the fastest — and it still misses the floor. Note the MTP column flips sign
for the 2.4T here (0.72 → 0.48): drafting touches more distinct experts than
a 3 GB/s disk can serve.

The 397B at near-Q8 attention and Q4 experts is a far better model per token
than the 2.4T at 2 bits, and it is 3–6× faster. `install.sh` defaults to it;
`MINILLM_MODEL=qwen3.8-2.4t` gets the IQ2_XXS 2.4T for comparison. Both fit
side by side.

> **Caveat:** most rows are *kernel*-bound — the binding constant is
> `kernel_gbps: 24`, an estimate never measured on any machine. The laptop
> baseline is the first real measurement of that; the server figure gets
> calibrated from it.

### Download planning

The laptop pulled 59 GB in ~5 h at ~3.4 MB/s. At that rate:

| Payload | Size | Time at 3.4 MB/s | At 30 MB/s |
| --- | --- | --- | --- |
| gpt-oss-120b GGUF (done) | 59 GB | 5 h | 33 min |
| 2.4T IQ2_XXS + MTP Q4_K_M | 630 GB | **~53 h** | 5.8 h |
| 2.4T IQ2_XS + MTP | 699 GB | ~59 h | 6.5 h |

The server should fetch directly on a better link; do not route 630 GB through
this laptop. `scripts/fetch_model.py --files ...` handles the multi-file split
and resumes on interruption.

### The falsifiable question

Everything hinges on one number: **what expert-cache hit rate is achievable at a
given residency?** On the server at Q2_K attention, the cache is 12.6 GB against
11.1 GB of per-token expert demand (IQ2_XXS) — it holds barely one token's
working set, so hit rate is almost entirely consecutive-token expert reuse. On
the laptop the same ratio is 1.40 tokens of working set. Both machines are in
the regime where a routing-aware policy could matter; neither is in the regime
where it is guaranteed to.

That question does not need the 2.4T weights to answer — it needs routing traces.
Those can come from **Qwen3.6-35B-A3B** (`qwen3_5_moe`, same family, ~18 GB at
4-bit) which fits this laptop's 80 GB free, and be replayed offline against
candidate policies. The project de-risks on hardware already in hand.

## Format matters more than it looks

The `openai/gpt-oss-120b` **safetensors** repo cannot run on a CPU box, even
though it is only 60 GB of MXFP4 on disk. Transformers' MXFP4 path needs Triton
GPU kernels; with no GPU it dequantizes to bf16 — 117 B × 2 bytes = **234 GB of
RAM**. The 60 GB on-disk figure is a trap.

The **GGUF** build of the same weights (`ggml-org/gpt-oss-120b-GGUF`, 59.0 GiB)
runs natively: llama.cpp implements MXFP4 directly and mmaps the file, so the
resident set is whatever the OS page cache holds, not the model size.

Two more format traps found while surveying, both the same shape — repos
shipping redundant full copies of the weights:

| Repo | Trap | Cost if naive |
| --- | --- | --- |
| `openai/gpt-oss-120b` | `metal/` + `original/` subdirs | 182 GB instead of 61 GB |
| `mistralai/Mistral-Small-4-119B` | HF + Mistral consolidated copies | 242 GB instead of 121 GB |

`scripts/fetch_model.py` fetches root-level files only, which handles both.

## What actually moves per token — and what does not

The whole point of targeting MoE is that **only the experts the router picks
are touched**. Concretely, per generated token:

| | Qwen3.5-397B (Q8 attn / Q4 exp) | Qwen3.8-2.4T (Q3 attn / IQ2 exp) |
| --- | --- | --- |
| Full model on disk | 213 GB | 588 GB |
| Routed-expert pool | 204 GB (60 × 512 experts) | 569 GB (92 × 512) |
| **Touched per token** | **13.3 GB** = 9.4 hot + 4.0 experts | **30.6 GB** = 19.5 hot + 11.1 experts |
| Fraction of the model read | **6.3 %** | **5.2 %** |
| Fraction of experts read | 2.0 % (10 of 512 per layer) | 2.0 % |

So ~94–95 % of the weights are **never read** for any given token. That is not
something MiniLLM has to build — it is how MoE inference already works, and
llama.cpp already honours it: it mmaps the file and only faults in the pages
of the experts the router selects. Nothing else is streamed.

But two things are *not* selective, and this is where the design work is:

1. **The "hot" half is touched on every token, by every model.** Attention,
   the shared expert, routers, norms, `lm_head` — 9.4 GB for the 397B, 19.5 GB
   for the 2.4T. There is no router in front of it; it *cannot* be skipped.
   That is why it must live in RAM permanently, and why the 2.4T's 19.5 GB hot
   set (at 3-bit!) is what starves its expert cache on a 32 GB box.
2. **The granularity of what gets moved is the OS's, not the router's.** mmap
   faults in 4 KB pages, and the page cache evicts 4 KB pages by LRU. When a
   selected expert is 13 MB, that is ~3,300 individual faults per expert; on
   the 8 GB laptop this saturated the fault handler at 300 MB/s and left three
   cores idle (see below). The router knows "expert 217 of layer 40, all
   13 MB, now" — the kernel only knows "page missing". A loader that reads
   *whole selected experts* with a few large `O_DIRECT` reads, and a cache
   that evicts *whole experts* by routing history rather than pages by
   recency, is what `runtime/` exists to build. It moves the same 5 % of the
   model per token — it just moves it in the right units.

One clarification about a llama.cpp flag that *looks* like an expert cache but
is not: `--n-cpu-moe N` / `--cpu-moe` decide CPU-vs-GPU **placement** of expert
tensors. On a CPU-only machine everything is already on the CPU, so they do
nothing — and neither pins experts in RAM against page-cache eviction. There
is no llama.cpp flag for "keep these experts resident"; that is precisely the
gap. And speculative decoding drafts k tokens per weight sweep, which
amortises the *hot* half but touches the union of k tokens' experts (≈26
distinct for 3 tokens, not 10) — which is why MTP flips negative on a slow
disk.

## Running the baseline

`bench/llama_baseline.py` drives llama.cpp (prebuilt `b10437`, AVX2 `haswell`
backend — this CPU has no AVX-512). llama.cpp mmaps the GGUF and lets the OS page
cache decide residency, which *is* an expert cache — just a routing-blind one
evicting 4 KB pages by LRU. **That number is the bar.** If a routing-aware cache
cannot beat page-cache LRU, the thesis is wrong and worth knowing early.

```bash
python bench/llama_baseline.py --generate --load-mode mmap
```

**For a model larger than RAM, `--generate` (llama-cli) is the only mode that
works on build b10437.** Three things about this build, each verified the hard
way against the binary:

1. **`--no-repack` is mandatory, and it is not the default.** llama.cpp enables
   weight repacking by default — rewriting quantized weights into a
   SIMD-friendly layout at load time. That requires a real allocation of the
   whole model:

   ```
   done_getting_tensors: tensor 'token_embd.weight' (q8_0) (and 578 others) cannot be used with preferred buffer type CPU_REPACK, using CPU instead
   alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 60914073600
   llama_model_load: error loading model: unable to allocate CPU_REPACK buffer
   ```

   60.9 GB on an 8 GB machine. Repacking **defeats mmap entirely**. `llama-cli`
   honors `--no-repack`; the harness passes it for you.

2. **`llama-bench` cannot run a model larger than RAM at all.** It has no
   `--no-repack` flag and ignores `LLAMA_ARG_REPACK` — it repacks
   unconditionally and dies on the same allocation. The harness refuses to
   invoke it when the model exceeds ~85% of RAM and says why. `llama-bench` is
   still the right tool for models that *fit*.

3. **This `llama-cli` is a chat frontend.** It prints
   `[ Prompt: X t/s | Generation: Y t/s ]` after each turn (one decimal —
   `0.0` means under 0.05), only prints per-token timing with `--perf` (default
   off), and drops into an interactive `>` prompt afterward *even with*
   `-no-cnv`. The harness passes `--perf` and closes stdin so it exits instead
   of hanging. If you run it by hand and it appears frozen after the answer, it
   is waiting for you to type.

The measured `kernel_gbps` from this path is the *un-repacked* AVX2 rate — the
honest number for a deployment where repacking is not an option.

**`--load-mode`:** only `mmap` streams. `dio` and `mlock` load the whole model
into RAM and cannot run anything larger than RAM — an earlier draft of this
README called `dio` a "no page cache" bracket for a custom runtime; that was
wrong, it is a load-time mode, not a streaming one.

**Run it detached.** A 60 GB model streaming through 8 GB takes many minutes
per run, and it will be killed if the launching terminal is interrupted — a
Ctrl-C to the shell propagates to the child. On Windows, `Start-Process` is
*not* enough (it shares the console group); use Task Scheduler via
`scripts/run_baseline_detached.cmd`. On Linux, use `tmux` or `nohup`.

Also: a scheduled task / cron job does **not** inherit your shell environment.
`HF_HOME`, `HF_HUB_CACHE`, `MINILLM_LLAMA_BIN` must be set inside the wrapper
script, or the harness will look in the wrong place. The Windows wrapper does
this; the Linux guide puts them in `~/.bashrc`, which `tmux` picks up but
`cron` would not.

The repo also ships `eagle3-gpt-oss-120b-Q8_0.gguf` (810 MB) — a draft model for
speculative decoding. gpt-oss has no MTP head, so this is how it gets the
amortization multiplier; `--draft` wires it in via `-md` with
`--spec-draft-n-max 4` (the older `--draft-max` flag no longer exists on this
build). The draft cannot load standalone — it needs the target model's
embeddings — so it only runs alongside the main model.

## The baseline result, and what it actually measures

gpt-oss-120b (59 GB MXFP4) on the 8 GB laptop, `llama-completion --perf`,
4 threads, mmap. Two consecutive runs; full data in
`results/baseline_gen_hackintosh.json`:

| Run | Cache | Prompt eval | **Generation** | Output |
| --- | --- | --- | --- | --- |
| 1 | cold | 12.7 s/tok | **19.7 s/tok = 0.05 tok/s** | "The capital of France is Paris." |
| 2 | warm | 7.2 s/tok | **12.9 s/tok = 0.08 tok/s** | |

The model runs and is correct. It is also **~12–19x slower than `capacity.py`
predicted** at a 70% hit rate — and, more tellingly, **3–5x slower than the
model's zero-hit-rate disk floor** (3.49 s/token). Neither cache hit rate nor
disk bandwidth can explain that. So something the model did not account for is
binding. Sampling the OS during generation found it:

| Counter during generation | Value |
| --- | --- |
| Disk read | 264–326 MB/s (SSD max: 510) |
| Pages input | **67k–83k / s** — × 4 KB = 262–326 MB/s |
| Hard page faults | 95k–181k / s |
| Available RAM | 68–136 MB |
| llama-completion CPU | **83–118% of 400%** — 3 of 4 cores idle |

Every byte of I/O arrives as an individual **4 KB hard page fault**. The page
cache (~2.5 GB after the process's own working set) is smaller than one
token's 3.27 GB of demand, so each token evicts most of the previous token's
experts, and mmap re-faults them one page at a time. The fault handler
saturates at ~300 MB/s — 59% of what the SSD can deliver — while three cores
wait. `3.27 GB ÷ 0.30 GB/s = 10.9 s/token` predicted from that mechanism alone;
measured 12.9–19.7. **The run is page-fault-bound**: not compute, not device.

Two consequences, one deflating and one encouraging:

- **The 8 GB laptop cannot test the *cache* thesis** (routing-aware vs LRU).
  With capacity below one token's working set there is nothing to cache — every
  policy degenerates to "reload everything". `sim/cache_sim.py`'s laptop
  scenario assumed 1.4 tokens of cache; the real figure after process overhead
  is ~0.75. The cache-policy question moves to the 32 GB server.
- **It is a clean test of the *loader* thesis**, and the loader thesis just got
  much stronger. `capacity.py` assumed the disk delivers at device bandwidth in
  large blocks. It does not, through mmap: it delivers 4 KB at a time through a
  fault handler, and that costs 40% of bandwidth and 3 cores. An expert loader
  that reads whole 13 MB experts with a handful of large overlapped / `O_DIRECT`
  reads — instead of faulting 3,300 pages each — should approach 510 MB/s
  *and* free the cores to compute. On this exact laptop that is worth **~1.7x
  from bandwidth alone**, before any caching. That is the first thing
  `runtime/` should build, and this measurement is its before-picture.

The 32 GB / NVMe server has 6.7x the disk bandwidth and ~10x the cache; whether
its mmap path is fault-bound too is the first thing to measure there
(SETUP_LINUX.md step 7 does it). If it is, the loader wins on both machines.

## Where the thesis actually lives

`sim/cache_sim.py` simulates cache policies against routing traces. The decisive
policy is **Belady** — optimal offline eviction. No online policy can beat it, so
if Belady misses the target, no engineering does.

For gpt-oss-120b on the laptop, one expert is 13.2 MB at MXFP4, the cache holds
~202 experts, and one token touches 144 (36 layers × 4). **The cache holds 1.40
tokens of working set**, so hit rate is almost entirely consecutive-token expert
reuse. Sweeping that reuse rate ("stickiness") against a synthetic trace:

| Stickiness | LRU hit | Belady hit | LRU tok/s | Belady tok/s |
| --- | --- | --- | --- | --- |
| 0.30 | 41.9% | 64.2% | 0.49 | 0.80 |
| 0.50 | 57.9% | 73.0% | 0.68 | 1.06 |
| 0.60 | 67.3% | 77.7% | 0.88 | 1.28 |
| 0.70 | 74.4% | 82.5% | 1.12 | 1.64 |
| 0.80 | 82.7% | 87.6% | 1.65 | 1.83 |
| 0.90 | 91.1% | 93.5% | 1.83 | 1.83 |

This carves the outcome into three regimes, and **which one we are in is an
empirical fact about the model, not a design choice**:

- **Stickiness ≥ 0.8** — page-cache LRU already saturates the hardware at
  1.83 tok/s (the DRAM/kernel ceiling; disk stops mattering). A routing-aware
  cache buys **nothing**. The thesis is moot on this machine.
- **Stickiness 0.5–0.7** — LRU lands at 0.68–1.12 tok/s, Belady at 1.06–1.64.
  A smarter policy is worth **30–50%**, and is the difference between missing
  and hitting the 1 tok/s aim. **This is the only band where the project pays.**
- **Stickiness < 0.4** — even optimal caching cannot reach 1 tok/s. Wrong
  hardware or wrong model; no policy helps.

Two secondary findings: **LFU is actively bad** (41.6% vs LRU's 74.5%) — when the
cache holds ~1 token of working set, recency is the only signal that matters and
frequency is noise. And **layer-partitioned LRU slightly underperforms global
LRU** (73.8% vs 74.5%), so my "no layer should starve another" hypothesis was
wrong; global recency already handles it.

Caveat: these are synthetic traces. Stickiness is a crude single parameter, and
real routing has structure it does not capture — layer-dependent locality,
prefill-vs-decode differences, semantic clustering. **Measuring real stickiness
is the critical experiment**, and it is cheap once weights land.

## Status

- [x] `bench/roofline.py` — hardware calibration (measured)
- [x] `minillm/capacity.py` — machine × model × quantization planner
- [x] `scripts/derive_spec.py` — decompose any MoE without downloading weights
- [x] `scripts/fetch_model.py` — download with the redundant-copy traps handled
- [x] `bench/llama_baseline.py` — baseline harness (awaiting weights)
- [ ] measure real `kernel_gbps` — currently an estimate, and it binds every
      server projection
- [ ] `sim/` — offline replay of cache policies against routing traces
- [ ] `runtime/` — routing-aware expert cache + prefetch

## Prior art worth not reinventing

`llama.cpp` already has GGUF, AVX2 Q4_K/Q2_K kernels, mmap weight streaming and
MoE support, and would likely reach 1–3 tok/s on 30B-A3B here using the OS page
cache as a crude expert cache. **That is the baseline to beat, not AirLLM** —
AirLLM is a Python dense-model layer streamer with no routing awareness, and
beating it 10x is not a meaningful bar. KTransformers, PowerInfer and
MoE-Infinity attack the same problem; read them before building.
