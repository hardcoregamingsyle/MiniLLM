# MiniLLM — Linux Mint setup (the target machine)

Target: Linux Mint (XFCE / X11), **Intel i5-10210U** (4 cores / 8 threads,
Comet Lake-U, AVX2, 15 W), 32.6 GB DDR4, NVMe, ~920 GB free, Radeon 520/610
Mobile (2 GB) + Intel UHD iGPU. Goal: run `Qwen/Qwen3.8-2.4T-A95B` at ≥ 1 tok/s
(floor), aiming for 2.

Three facts about this specific machine that shape everything below:

- **PCIe 3.0 only.** 10th-gen U-series has no Gen4 lanes, so the NVMe tops out
  around 3–3.5 GB/s, not the 5.5–7 a Gen4 drive is rated for. Expert
  streaming from disk is the bottleneck for this model, so this is *the*
  constraint. Step 5 measures the real figure.
- **Neither GPU helps.** The Radeon 520/610 is a 2 GB GCN 1.0 part — too small
  and too old for any useful offload; the Intel iGPU shares the same DRAM
  bandwidth the CPU needs. Everything runs `-ngl 0` on the CPU. Do not build
  llama.cpp with HIP or Vulkan; it only adds failure modes.
- **15 W thermal envelope.** All-core AVX2 throttles after ~30 s. Sustained
  tok/s will be lower than a short benchmark suggests; run 7a for at least 64
  tokens to see the settled rate. A cooling pad or `powersave`→`performance`
  governor is a real, measurable lever here.

Everything below is a copy-paste sequence. The model download (250 GB default,
680 GB for the 2.4T) is the long pole — start it early and do Steps 4–5 while it runs.

---

## 0. Before anything: know your numbers

```bash
# CPU: need AVX2 at minimum. AVX-512 / VNNI / AMX are bonuses.
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Flags" | sed 's/Flags:.*avx2/Flags: ...avx2/' | cut -c1-120
grep -o -E 'avx512[a-z_]*|avx2|avx_vnni|amx_[a-z]*' /proc/cpuinfo | sort -u | tr '\n' ' '; echo

# RAM and swap
free -h

# NVMe: confirm it is actually NVMe and see free space
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN | grep -v loop
df -h ~ | tail -1
```

Write down: total RAM, free disk, and whether `avx512` appears. The capacity
planner needs them.

---

## 1. System packages

```bash
sudo apt update
sudo apt install -y git build-essential cmake ninja-build python3 python3-pip python3-venv \
                    libcurl4-openssl-dev pkg-config htop
```

`build-essential` + `cmake` are needed because on Linux you **build llama.cpp
from source** — that lets it target your exact CPU (`-march=native`) instead of
a generic AVX2 baseline, which matters at 2-bit where dequant is the bottleneck.

---

## 2. Clone MiniLLM and set up Python

```bash
cd ~
git clone https://github.com/hardcoregamingsyle/MiniLLM.git
cd MiniLLM

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Every later `python` command assumes `.venv` is activated. If a new shell
loses it: `source ~/MiniLLM/.venv/bin/activate`.

---

## 3. Start the model download (long — begin now, keep going)

Model files go on the NVMe. Pick a directory on it and point HF there. This
also sets your token so downloads run authenticated (much faster).

```bash
# Put these in ~/.bashrc so they persist across shells.
# The quoted 'EOF' keeps $HOME literal in the file, so it expands at source
# time -- which is what you want.
cat >> ~/.bashrc <<'EOF'
export HF_HOME="$HOME/minillm/hf"
export HF_HUB_CACHE="$HF_HOME/hub"
EOF
source ~/.bashrc
mkdir -p "$HF_HUB_CACHE"
```

Now the token — **do not paste a placeholder.** A fake `hf_xxx` value is worse
than no token: `huggingface_hub` sends it as a Bearer credential and the Hub
returns HTTP 401 *even for public repos*, so Step 3 dead-ends with a confusing
auth error. Set the real one, and set it separately from the block above so it
never lands in a file you might commit:

```bash
# Get a token at https://huggingface.co/settings/tokens (read scope is enough).
read -rsp "HF token: " HF_TOKEN; echo
echo "export HF_TOKEN=\"$HF_TOKEN\"" >> ~/.bashrc
source ~/.bashrc
python -c "from huggingface_hub import whoami; print('auth ok:', whoami()['name'])"
```

If that last line prints your username, downloads will run authenticated
(higher rate limits, faster). If it errors, fix it before Step 3 — nothing
after this works unauthenticated.

**Choosing the model and quant.** The installer's default is
`Qwen3.5-397B-A17B` at `UD-Q4_K_XL`; `MINILLM_MODEL=qwen3.8-27b` selects the
27B dense model (17.6 GB, fits in RAM on a 32 GB box);
`MINILLM_MODEL=qwen3.8-2.4t` selects the
2.4T at `UD-IQ2_XXS`. Here is why, with exact sizes read from the Hub:

| Model / quant | Size | Fits 920 GB? | tok/s here (single / +MTP) |
| --- | --- | --- | --- |
| **Qwen3.5-397B UD-Q4_K_XL** | **245 GB** | **yes** | **1.6 / 2.8** |
| Qwen3.8-2.4T UD-IQ2_XXS | 656 GB | yes | 0.72 / 0.48 |
| Qwen3.8-2.4T UD-IQ2_XS | 731 GB | yes | ~0.7 / ~0.5 |
| Qwen3.8-2.4T UD-IQ3_XXS ("3-bit") | 956 GB | **no** — 36 GB over, before the 20 GB draft | 0.36 |
| Qwen3.8-2.4T UD-IQ4_XS ("4-bit") | 1,311 GB | **no** — 411 GB over | 0.13 |

Two facts, both physical:

- **A 3-bit or 4-bit 2.4T does not fit on this disk.** IQ3_XXS alone is 956 GB
  against 920 free; IQ4_XS is 1.3 TB. This is not a margin question.
- **On this laptop, more bits on the 2.4T make it *slower*.** The bottleneck is
  the PCIe 3.0 NVMe (~3 GB/s) feeding 46 B of routed-expert parameters per
  token; every extra bit is more bytes through that pipe. IQ2_XXS is both the
  largest 2.4T that fits and the fastest, at 0.72 tok/s — and that still misses
  the 1 tok/s floor. The 397B at high precision clears the 2 tok/s aim.

The MTP draft comes with either model (6 GB for the 397B, 20 GB for the 2.4T).
Both fit alongside each other. **To fetch a specific model, name it:**

```bash
bash ~/minillm/download.sh qwen3.8-2.4t          # the 2.4T
bash ~/minillm/download.sh qwen3.8-27b          # the 27B dense (17.6 GB)
bash ~/minillm/download.sh qwen3.5-397b          # the 397B (the default)
bash ~/minillm/download.sh qwen3.5-397b UD-Q6_K  # a different quant of it
```

`download.sh` is generated by `install.sh` and is model-agnostic — with no
argument it fetches whatever `install.sh` was last run for, so if you ran the
installer plainly and want the 2.4T, pass its name. Run it inside `tmux` so it
survives a closed terminal (`tmux new -d -s minillm-dl-qwen3.8-2.4t "bash
~/minillm/download.sh qwen3.8-2.4t"`); the installer prints this exact line
for every model it did *not* start. Each model gets its own session, so the
two can download at once.

Dry-run first to see exactly what will be pulled:

```bash
python scripts/fetch_model.py --repo unsloth/Qwen3.5-397B-A17B-GGUF --dry-run --files "UD-Q4_K_XL/*"
```

Then start the real download in a `tmux` session so it survives you closing
the terminal or the laptop sleeping:

```bash
sudo apt install -y tmux
tmux new -s dl
# inside tmux:
source ~/MiniLLM/.venv/bin/activate && cd ~/MiniLLM
# 397B (default):
python scripts/fetch_model.py --repo unsloth/Qwen3.5-397B-A17B-GGUF --files "UD-Q4_K_XL/*" --workers 8
python scripts/fetch_model.py --repo a4lg/Qwen3.5-397B-A17B-MTP-ONLY-GGUF --files Qwen3.5-397B-A17B-MTP-ONLY-Q4_K_M.gguf
# or the 2.4T:
# python scripts/fetch_model.py --repo unsloth/Qwen3.8-2.4T-A95B-GGUF --files "UD-IQ2_XXS/*" --workers 8
# python scripts/fetch_model.py --repo a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF --files Qwen3.8-2.4T-A95B-MTP-ONLY-Q4_K_M.gguf
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t dl`. The download
resumes automatically if interrupted — just rerun the same command.

397B: at 30 MB/s ~2.3 h, at 100 MB/s ~40 min. 2.4T: ~6 h / ~1.9 h. Check progress with:

```bash
du -sh "$HF_HUB_CACHE"/models--unsloth--*
```

**Disable sleep/suspend while it runs** — Mint's power settings will otherwise
suspend the laptop mid-download.

---

## 4. Build llama.cpp for this CPU (while the download runs)

```bash
cd ~
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
cmake -B build -DGGML_NATIVE=ON -DGGML_LTO=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON
cmake --build build --config Release -j"$(nproc)"

# sanity check: it should list your CPU features
./build/bin/llama-cli --version
```

`-DGGML_NATIVE=ON` compiles for exactly this CPU. If `lscpu` showed
`avx512`, this is where it pays off — the AVX-512 dequant path is meaningfully
faster than AVX2 for IQ2/IQ1 quants.

Tell MiniLLM where it is:

```bash
echo 'export MINILLM_LLAMA_BIN="$HOME/llama.cpp/build/bin"' >> ~/.bashrc
source ~/.bashrc
```

---

## 5. Calibrate the machine (5 minutes, run once)

This measures the three constants every prediction depends on — DRAM bandwidth,
NVMe random-read bandwidth, and dequant kernel throughput — and writes them
into the machine profile.

```bash
cd ~/MiniLLM && source .venv/bin/activate
python bench/roofline.py --scratch "$HOME/minillm/bench_scratch"
```

Then paste the numbers into `minillm/machines.json` under `server-32gb-nvme`,
replacing the estimates, and set `"measured": true`. Then:

```bash
python -m minillm.capacity report --machine server-32gb-nvme --model qwen3.8-2.4t-a95b --hit 0.75
```

This prints the predicted tok/s for every quantization option, so you can see
whether the roofline agrees with the choice made in Step 3 *before* the download
finishes.

---

## 6. System tuning (do this before the first run)

Three things matter on Linux for a model much larger than RAM.

**6a. Swap off, or at least not on the model's disk.** Swap on the same NVMe
competes with expert reads. If you have swap enabled:

```bash
swapon --show
sudo swapoff -a          # temporary, until reboot
```

**6b. Let the page cache do its job.** llama.cpp mmaps the file, and the kernel's
page cache is the expert cache. Two settings help it.

Swappiness — prefer keeping file pages over swapping process memory:

```bash
sudo sysctl vm.swappiness=1
echo 'vm.swappiness=1' | sudo tee /etc/sysctl.d/99-minillm.conf   # persist
```

Readahead — larger reads per expert fetch. **Find the device by hand**, because
"the disk under `~`" is not one command on every install: on a plain-partition
Mint install it is `nvme0n1`, but with LVM it resolves to a logical volume and
with LUKS to a `dm-*` mapper, neither of which is what `blockdev` wants.

```bash
lsblk -o NAME,TYPE,SIZE,MOUNTPOINTS      # find the top-level "disk" row that ~ lives on
sudo blockdev --getra /dev/nvme0n1       # current value (usually 256)
sudo blockdev --setra 4096 /dev/nvme0n1  # substitute your disk name
```

That is **not** persistent across reboots. If it helps in Step 7, make it stick
with a udev rule:

```bash
echo 'ACTION=="add|change", KERNEL=="nvme0n1", ATTR{bdi/read_ahead_kb}="2048"' \
  | sudo tee /etc/udev/rules.d/60-minillm-readahead.rules
```

Whether readahead helps at all depends on the access pattern — treat it as an
experiment, measure with and without in Step 7a.

**6c. Huge pages are optional.** They reduce TLB pressure but llama.cpp gets
most of the benefit from mmap already. Skip unless profiling shows TLB misses.

---

## 7. First run

Once the download is done and llama.cpp is built, confirm the harness can see
the files:

```bash
cd ~/MiniLLM && source .venv/bin/activate
python bench/llama_baseline.py --list
```

You should see the 15 `Qwen3.8-2.4T-A95B-UD-IQ2_XXS-000NN-of-00015.gguf` shards
and the `MTP-ONLY-Q4_K_M.gguf` draft. On Linux the harness defaults to looking
for exactly these (`MINILLM_MODEL_PATTERN` / `MINILLM_DRAFT_PATTERN` override
it), and for a split model it correctly picks shard `00001` — which is *not* the
largest file, so do not point `--model` at the biggest shard by hand.

**7a. The tok/s number — `llama-completion --perf`.** This is the measurement:

```bash
python bench/llama_baseline.py --generate --threads "$(nproc)" --n-gen 32
```

> **Why `llama-completion` and not `llama-bench` or `llama-cli`?** On llama.cpp
> b10437, both of the obvious tools fail for a model larger than RAM, and both
> failures are silent-ish:
> - `llama-bench` repacks weights **unconditionally** — no `--no-repack` flag,
>   ignores `LLAMA_ARG_REPACK` — and dies allocating the whole model. The
>   harness refuses to invoke it for oversized models and says why.
> - `llama-cli` is now a chat frontend. It ignores `-no-cnv`, prints only a
>   rounded `[ Prompt: X t/s | Generation: Y t/s ]`, and on EOF from stdin it
>   prints `>` and **loops forever** instead of exiting (one log here reached
>   46,000 prompt lines). Unusable for scripting.
>
> `llama-completion` is the classic non-chat binary: honors `--no-repack` and
> `-no-cnv`, exits when done, and with `--perf` prints the real breakdown.
> The harness uses it. All three binaries are in `build/bin/` after Step 4.

With `--perf` you get `common_perf_print` lines: **load time**, **prompt eval**
(tok/s over the prompt), and **eval** (tok/s during generation — *the* number).
The first run is cold — the page cache is empty and every expert comes from
NVMe — so run it twice and read the second. Result lands in
`results/baseline_gen_<hostname>.json`.

**While it generates, sample the OS in a second terminal.** On the 8 GB laptop
this is what exposed the real bottleneck (page-fault-bound mmap, three of four
cores idle — see README "The baseline result"). The same three numbers decide
whether the server has the same problem:

```bash
# in a second terminal, while 7a is generating (not loading):
vmstat 3 10          # columns: bi = KB/s read in; us/sy/id = CPU user/sys/idle
```

Read it as: `bi × 1024` vs your NVMe's measured bandwidth from Step 5, and
`id` (idle %) with all cores counted. If `bi` is well below the NVMe figure
*and* idle is high, the run is fault-bound like the laptop — the kernel is
serving mmap in 4 KB pages faster than the CPU can fault them, and a bulk
expert loader would win. If `bi` sits near the NVMe figure, it is genuinely
disk-bound and only caching helps. If idle is near 0, it is compute-bound and
the dequant kernel is the wall. Note which one; it decides what to build next.

For per-process faults specifically: `pidstat -r -p "$(pgrep llama-completion)" 3`
(package `sysstat`) — the `majflt/s` column is hard faults per second.

**7b. Speculative decoding — add the MTP draft:**

```bash
python bench/llama_baseline.py --generate --threads "$(nproc)" --n-gen 32 --draft
```

The harness passes **`--no-repack`** for you — mandatory for anything larger
than RAM, and *not* llama.cpp's default. Without it, llama.cpp tries to allocate
the whole model in RAM to rewrite the weights into a SIMD-friendly layout:

```
alloc_tensor_range: failed to allocate CPU_REPACK buffer of size 6xxxxxxxxxx
```

`--draft` finds the `MTP-ONLY` file and passes `-md ... --spec-draft-n-max 4`.
If it cannot find the draft it **exits with an error** rather than silently
running without speculation, so the two runs are never accidentally identical.
Compare `Generation: X t/s` between them. MTP is the single largest lever on this
machine — expect a meaningful improvement if the draft accepts well.

> **If you run `llama-cli` by hand** (for an interactive chat with the model,
> which is what it is for now), it prints `[ Prompt: X t/s | Generation: Y t/s ]`
> rounded to one decimal — `0.0` means "under 0.05" — and stays at a `>` prompt
> waiting for you. Type `/exit` to leave. Do not script it.

> **Whether an MTP-ONLY GGUF loads via `-md` is unverified.** `-md` expects a
> standalone draft model with its own embeddings and trunk. The `a4lg` MTP-ONLY
> file is a newer format; if llama.cpp rejects it, that is a real finding, not a
> setup mistake — note it and proceed with 7a alone.

`--load-mode dio` and `mlock` are **not** useful here: both load the entire
model into RAM and cannot run anything larger than RAM. Only `mmap` streams.

---

## 8. What to expect, honestly

These are `capacity.py` outputs for **this** machine — i5-10210U (4c AVX2),
34 GB/s DRAM, **PCIe 3.0 NVMe at ~3.0 GB/s**, `kernel_gbps` 21.9 taken from a
CI runner with the same core count and SIMD — at a 75% expert-cache hit rate,
MTP at k=3 / 60% acceptance. Reproduce with
`python -m minillm.capacity report --machine server-32gb-nvme --model <model>`.
They are estimates until Step 5 replaces DRAM and disk with measured values.

**Qwen3.8-2.4T-A95B on this machine:**

| Attention | Experts | Cache | Single-stream | With MTP |
| --- | --- | --- | --- | --- |
| Q4_K_M | IQ2_XXS | 1.9 GB | 0.33 | 0.33 |
| Q3_K_M | IQ2_XXS | 8.1 GB | 0.72 | 0.48 |
| Q2_K | IQ2_XXS | 12.7 GB | 0.84 | 0.72 |
| Q2_K | IQ1_S | 12.7 GB | 0.94 | 1.53 |

**The 2.4T does not reach the 1 tok/s floor on this machine at any usable
precision.** The one row that clears it (Q2_K attention + IQ1_S experts + MTP)
is ~1.6-bit experts — a badly degraded model. And note the sign flip in the
MTP column: at Q3_K_M and Q2_K attention, **speculation makes it slower**
(0.72 → 0.48, 0.84 → 0.72). Drafting k tokens touches more distinct experts,
and on a PCIe 3.0 disk those extra bytes cost more than the amortization saves.
That is what the U-series PCIe 3.0 lanes do to this plan; a Gen4 slot would
have given 1.30 on the Q2_K row.

**Qwen3.5-397B-A17B on the same machine:**

| Attention | Experts | Disk | Single-stream | With MTP |
| --- | --- | --- | --- | --- |
| **Q8_0** | **Q4_K_M** | **214 GB** | **1.64** | **2.80** |
| Q6_K | Q4_K_M | 212 GB | 1.96 | 3.20 |
| Q4_K_M | Q4_K_M | 209 GB | 2.44 | 3.36 |
| Q4_K_M | Q2_K | 124 GB | 3.00 | 5.05 |

**This clears both the floor and the 2 tok/s aim, with full-precision
attention and 4-bit experts.** 397 B total, 17 B active, MTP head included,
uses a quarter of the disk. Same family and tokenizer as the 2.4T.

So the honest recommendation for *this* laptop flips from the earlier draft:
**start with `Qwen3.5-397B-A17B` at Q8_0/Q4_K_M.** It is a much better model at
that precision than the 2.4T at 1.6-bit, it is 3× faster, and it downloads in a
third of the time. Set `MINILLM_MODEL=qwen3.5-397b` before running `install.sh`
(the installer supports both). If you want the 2.4T anyway — for the name, or
to measure it — the 656 GB fits alongside; run `install.sh` again with
`MINILLM_MODEL=qwen3.8-2.4t` and it adds it without redoing anything else.

Three things worth knowing:

1. **Attention precision matters more than expert precision** for the 2.4T.
   Every bit on the 48.7 B always-hot half is a bit taken from the expert
   cache; at Q4_K_M attention the cache is 1.9 GB and it is disk-bound at 0.33.
   Check what precision the downloaded UD file uses for `attn_*` tensors
   (`llama-cli --verbose` prints per-tensor types); if Q4-ish, expect row 1.
2. **More RAM would not fix the 2.4T here; a faster disk would.** Sweeping RAM
   in `capacity.py` flatlines past ~48 GB. Sweeping disk: 3.0 → 5.5 GB/s moves
   the Q2_K+MTP row from 0.72 to 1.30. The bottleneck on this machine is PCIe
   3.0, and that is soldered.
3. **The 15 W part will throttle.** `kernel_gbps` 21.9 came from a runner that
   does not. Run 7a for 64+ tokens and read the settled rate, and try
   `sudo cpupower frequency-set -g performance` — on a U-series it is worth
   measuring.

---

## 9. Troubleshooting

**`failed to allocate CPU_REPACK buffer`** — you ran llama.cpp directly without
`--no-repack`. Use the harness, or add the flag.

**Extremely slow first token, then fine** — normal. Cold page cache.

**Every token slow, `iotop` shows constant NVMe reads** — page cache is
thrashing. Check `free -h`: if `available` is small, something else is using
RAM. Check swap is off (`swapon --show` should be empty).

**Killed / OOM** — reduce `--ctx` (KV cache) or `--threads`. Each thread has
scratch buffers.

**Download restarts from zero** — `HF_HUB_CACHE` changed between runs. Make sure
the `export`s are in `~/.bashrc` and sourced.

**`llama-completion: command not found` from the harness** — `MINILLM_LLAMA_BIN` not
set or not exported. `echo $MINILLM_LLAMA_BIN` should print the build dir.

---

## Where things live

```
~/MiniLLM/                    this repo (code, docs, results)
~/MiniLLM/.venv/              Python environment
~/llama.cpp/build/bin/        llama.cpp binaries built for this CPU
~/minillm/hf/hub/             model files (on NVMe, 250-680 GB)
~/minillm/bench_scratch/      roofline test file (deleted after run)
```
