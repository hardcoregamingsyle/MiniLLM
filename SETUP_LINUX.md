# MiniLLM — Linux Mint setup (the target machine)

Target: Linux Mint, 32 GB RAM, ~920 GB free on NVMe. Goal: run
`Qwen/Qwen3.8-2.4T-A95B` at ≥ 1 tok/s (floor), aiming for 2.

Everything below is a copy-paste sequence. Total download is ~630 GB, so
Step 3 is the long pole — start it early and do Steps 4–5 while it runs.

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

**Choosing the quant.** 920 GB free forces ≤ 2-bit routed experts. Exact
sizes of the unsloth GGUFs (measured, not estimated):

| Quant | Size | Verdict |
| --- | --- | --- |
| UD-IQ2_XXS | 611.5 GB | **recommended** — best quality that fits with room |
| UD-IQ2_XS | 680.5 GB | fits, ~240 GB left; slightly better quality |
| UD-IQ3_XXS | 889.9 GB | **does not fit** — see below |

Plus the MTP draft model for speculative decoding (18.5 GB). Total ≈ 630 GB.

Why IQ3_XXS does not fit in 920 GB even though 889.9 < 920: the model is not
the only thing on the disk. The MTP draft is 18.5 GB, `hf-xet` keeps a chunk
cache during download, the llama.cpp build tree is ~1 GB, and a filesystem
running above ~95% full degrades badly. 889.9 + 18.5 + cache + margin exceeds
920. (RAM-side costs like the OS and KV cache do **not** affect this — they
are a separate budget.)

Dry-run first to see exactly what will be pulled:

```bash
python scripts/fetch_model.py --repo unsloth/Qwen3.8-2.4T-A95B-GGUF --dry-run --files "UD-IQ2_XXS/*"
```

Then start the real download in a `tmux` session so it survives you closing
the terminal or the laptop sleeping:

```bash
sudo apt install -y tmux
tmux new -s dl
# inside tmux:
source ~/MiniLLM/.venv/bin/activate && cd ~/MiniLLM
python scripts/fetch_model.py --repo unsloth/Qwen3.8-2.4T-A95B-GGUF --files "UD-IQ2_XXS/*" --workers 8
python scripts/fetch_model.py --repo a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF --files Qwen3.8-2.4T-A95B-MTP-ONLY-Q4_K_M.gguf
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t dl`. The download
resumes automatically if interrupted — just rerun the same command.

At 30 MB/s that is ~6 hours; at 100 MB/s ~1.7 hours. Check progress with:

```bash
du -sh "$HF_HUB_CACHE"/models--unsloth--Qwen3.8-2.4T-A95B-GGUF
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

With `--perf` you get `llama_perf_context_print` lines: **load time**, **prompt
eval** (tok/s over the prompt), and **eval** (tok/s during generation — *the*
number). The first run is cold — the page cache is empty and every expert comes
from NVMe — so run it twice and read the second. Result lands in
`results/baseline_gen_<hostname>.json`.

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

These are `capacity.py` outputs (75% expert-cache hit rate, MTP at k=3 with
60% acceptance), reproducible with
`python -m minillm.capacity report --machine server-32gb-nvme --model qwen3.8-2.4t-a95b`.
They are **estimates until Step 5 replaces the machine constants with measured
ones** — in particular `kernel_gbps`, which binds most rows and is currently a
guess.

| Attention | Experts | Cache | Single-stream | With MTP |
| --- | --- | --- | --- | --- |
| Q4_K_M | IQ2_XXS | 1.8 GB | 0.59 | 0.60 |
| Q3_K_M | IQ2_XXS | 8.0 GB | 0.78 | 0.87 |
| **Q2_K** | **IQ2_XXS** | **12.6 GB** | **0.92** | **1.30** |
| Q2_K | IQ1_S | 12.6 GB | 1.03 | 1.68 |

Three things this table says that are not obvious:

1. **The UD-IQ2_XXS file's attention precision matters more than its expert
   precision.** Unsloth Dynamic quants keep attention higher than the routed
   experts — good for quality — but every bit spent on the 48.7 B always-hot
   half is a bit taken from the expert cache. At Q4_K_M attention the cache is
   1.8 GB, MTP has nothing to amortize against, and it is disk-bound at 0.6.
   Check what precision the downloaded file actually uses for `attn_*` tensors
   (`llama-cli --verbose` prints per-tensor types at load); if it is Q4-ish, the
   honest expectation is the first row.
2. **The 1 tok/s floor is reachable; the 2 tok/s aim is not on this machine.**
   And — correcting an earlier draft of this guide — **more RAM does not fix it.**
   Sweeping RAM in `capacity.py`: 32 GB → 1.30, 48 GB → 1.43, 64/96/128 GB →
   1.43. It flatlines because past ~48 GB the bottleneck is dequant *compute*,
   not memory. Reaching 2 tok/s on the 2.4T needs a faster kernel (AVX-512
   helps; so would a GPU for the hot half), not more DDR4.
3. **`kernel_gbps` decides everything above 0.8 tok/s.** It is the one constant
   in `machines.json` that has never been measured on any machine. Step 5 fixes
   that; do not trust rows 2–4 until it has run.

If IQ2 quality proves unusable in practice, the fallback is
`Qwen3.5-397B-A17B` — a much smaller model at much better precision (Q8 attn +
Q4 experts, 214 GB) that `capacity.py` puts at ~1.8 tok/s here.

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
~/minillm/hf/hub/             model files (on NVMe, ~630 GB)
~/minillm/bench_scratch/      roofline test file (deleted after run)
```
