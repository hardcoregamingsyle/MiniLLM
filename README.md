# MiniLLM — running trillion-parameter MoE models on machines with 8–32 GB of RAM

CPU inference for Mixture-of-Experts models far larger than RAM, streaming
experts from disk. The bet: **a router-aware expert cache beats the OS page
cache**, because the OS evicts 4 KB pages by LRU and has no idea what a router is.

**Setting up the Linux target machine? Start with [SETUP_LINUX.md](SETUP_LINUX.md).**
It is a copy-paste sequence: system packages → clone → start the 630 GB download
→ build llama.cpp for your CPU → calibrate → first run.

## What is in here

| Path | What it does |
| --- | --- |
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

### Server (32 GB / 900 GB): Qwen3.8-2.4T-A95B — the confirmed target

The 2.4T is the model for the server. Runnable GGUFs exist as of 10 Aug 2026:
[`unsloth/Qwen3.8-2.4T-A95B-GGUF`](https://hf.co/unsloth/Qwen3.8-2.4T-A95B-GGUF).
Exact sizes, which decide what 900 GB can hold:

| Quant | Size | Fits 900 GB? | Note |
| --- | --- | --- | --- |
| UD-Q1_0 | 370.0 GB | yes | 1-bit; quality unknown |
| UD-IQ1_S | 473.5 GB | yes | |
| UD-IQ1_M | 525.2 GB | yes | |
| **UD-IQ2_XXS** | **611.5 GB** | **yes** | **best quality that fits** |
| UD-IQ2_XS | 680.5 GB | yes | 219 GB headroom, tighter |
| UD-IQ3_XXS | 889.9 GB | **no** — 10 GB short after OS/KV | |
| UD-IQ4_XS | 1220.8 GB | no | |
| Q8_0 / BF16 | 2.4 / 4.6 TB | no | |

**900 GB forces ≤ 2-bit experts.** IQ3_XXS misses by ~10 GB once the OS and KV
cache are accounted for; IQ2_XS is the largest that fits comfortably. UD (Unsloth
Dynamic) quants keep attention and shared experts at higher precision than the
routed experts, which is exactly the split `capacity.py` models — so the "hot"
half is not crushed to 2 bits even in an IQ2 file.

Native MTP draft for speculative decoding is also published:
[`a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF`](https://hf.co/a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF)
(Q4_K_M 18.5 GB, Q8_0 30.1 GB). Speculation is the biggest single lever on the
server; the plan is IQ2_XXS main + Q4_K_M MTP draft = **~630 GB total**.

Predicted by `capacity.py` (75% hit rate, MTP at k=3 / 60% acceptance):

| Attention | Experts | Cache | Single | + MTP | Binding |
| --- | --- | --- | --- | --- | --- |
| Q4_K_M | IQ2_XXS | 1.8 GB | 0.59 | 0.60 | disk |
| Q3_K_M | IQ2_XXS | 8.0 GB | 0.78 | 0.87 | kernel |
| **Q2_K** | **IQ2_XXS** | **12.6 GB** | **0.92** | **1.30** | kernel |
| Q2_K | IQ1_S | 12.6 GB | 1.03 | 1.68 | kernel |

Two things this settles. First, **attention precision decides more than expert
precision**: at Q4_K_M attention the cache is 1.8 GB and MTP has nothing to
amortize (0.59 → 0.60); at Q2_K attention it opens 12.6 GB and MTP is worth
40%. What precision the downloaded UD-IQ2_XXS file actually uses for attention
tensors is therefore the first thing to check on the server. Second, honest
verdict: **the 1 tok/s floor is reachable, the 2 tok/s aim is not** on 32 GB —
and sweeping RAM in `capacity.py` shows **more RAM does not fix it**: 32 GB →
1.30, 48 GB → 1.43, 64/96/128 GB → 1.43. It flatlines because above ~48 GB the
bottleneck is dequant compute, not memory. Reaching 2 tok/s on the 2.4T needs a
faster kernel, not more DDR4.

For reference, the strongest alternative found on 32 GB is Qwen3.5-397B-A17B
(1.80 tok/s at Q8_0 attn + Q4_K_M experts, 214 GB) — a much smaller model at
much better precision. It is not the target; it is what to fall back to if the
2.4T's IQ2 quality proves unusable.

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
