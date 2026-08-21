#!/usr/bin/env bash
# MiniLLM runner. Finds the model, picks the right llama.cpp binary and flags,
# and runs it. This is the thing to use day-to-day; the harness in bench/ is
# for measurement.
#
#   bash run.sh                       benchmark: 32 tokens, prints tok/s
#   bash run.sh --draft               same, with MTP speculative decoding
#   bash run.sh --chat                interactive chat with the model
#   bash run.sh --warm                pre-load the hot set with parallel readers first
#   bash run.sh --lock                PIN the hot set in RAM (un-evictable; needs sudo)
#   bash run.sh -p "your prompt" -n 64
#
# Env: MINILLM_LLAMA_BIN, HF_HUB_CACHE, MINILLM_MODEL_PATTERN, MINILLM_THREADS

set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$here"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate

# Pull env from ~/.bashrc's MiniLLM block if this shell did not source it.
if [[ -z "${MINILLM_LLAMA_BIN:-}" && -f "$HOME/.bashrc" ]]; then
  eval "$(sed -n '/# >>> MiniLLM >>>/,/# <<< MiniLLM <<</p' "$HOME/.bashrc" | grep '^export' || true)"
fi
BIN="${MINILLM_LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/minillm/hf}/hub}"
PATTERN="${MINILLM_MODEL_PATTERN:-Qwen3.5-397B-A17B-UD-Q4_K_XL}"
# Generation is memory-bandwidth-bound: extra hyperthreads contend for the same
# DRAM and usually do not help, so default to PHYSICAL cores. Prompt processing
# is compute-bound and does benefit from SMT, so it gets nproc.
PHYS=$(lscpu -p=Core,Socket 2>/dev/null | grep -v '^#' | sort -u | wc -l)
[[ "${PHYS:-0}" -gt 0 ]] 2>/dev/null || PHYS=$(nproc)
THREADS="${MINILLM_THREADS:-$PHYS}"
THREADS_BATCH="${MINILLM_THREADS_BATCH:-$(nproc)}"

mode=bench; draft=0; warm=0; lock=0; prompt="Explain, in three sentences, why mixture-of-experts models can run on machines with far less RAM than the model size."; ntok=32; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chat)  mode=chat; shift ;;
    --draft) draft=1; shift ;;
    --warm)  warm=1; shift ;;
    --lock)  lock=1; shift ;;
    -p|--prompt) prompt="$2"; shift 2 ;;
    -n) ntok="$2"; shift 2 ;;
    -t|--threads) THREADS="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) extra+=("$1"); shift ;;
  esac
done

# --- locate model: shard 00001 of a split GGUF, or a single file ------------
# HF cache files under snapshots/ are SYMLINKS to blobs/, so -L is required
# (plain 'find -type f' silently finds nothing). Restrict to snapshots/ so a
# half-downloaded blobs/*.incomplete can never be picked.
findgguf() { find -L "$CACHE" -path '*/snapshots/*' -type f -name "$1" 2>/dev/null | sort | head -1 || true; }
model=$(findgguf "*${PATTERN}*-00001-of-*.gguf")
[[ -n "$model" ]] || model=$(find -L "$CACHE" -path '*/snapshots/*' -type f -name "*${PATTERN}*.gguf" ! -name "*-of-*" 2>/dev/null | sort | head -1 || true)
# Pattern missed? If exactly one split model (excluding MTP drafts) is present,
# use it -- the common case after install.sh with any MINILLM_MODEL.
if [[ -z "$model" ]]; then
  mapfile -t firsts < <(find -L "$CACHE" -path '*/snapshots/*' -type f -name "*-00001-of-*.gguf" ! -iname "*MTP-ONLY*" 2>/dev/null | sort)
  if [[ ${#firsts[@]} -eq 1 ]]; then
    model="${firsts[0]}"; echo "note   : pattern '$PATTERN' not found; using the only model present"
  elif [[ ${#firsts[@]} -gt 1 ]]; then
    echo "Several models present; pick one with MINILLM_MODEL_PATTERN=<substring>:" >&2
    printf '  %s
' "${firsts[@]##*/}" >&2; exit 1
  fi
fi
if [[ -z "$model" ]]; then
  echo "No GGUF matching '$PATTERN' under $CACHE" >&2
  echo "Still downloading?  tmux attach -t minillm-dl" >&2
  echo "Different model?    MINILLM_MODEL_PATTERN=<substring> bash run.sh" >&2
  exit 1
fi
# For a split model, refuse to start until EVERY shard exists. llama.cpp opens
# all of them at load, and with parallel downloads they finish in arbitrary
# order -- shard 15 present says nothing about shard 7.
if [[ "$model" =~ -00001-of-([0-9]{5})\.gguf$ ]]; then
  n=$((10#${BASH_REMATCH[1]})); missing=()
  for ((i=1; i<=n; i++)); do
    f="${model/-00001-of-/-$(printf '%05d' "$i")-of-}"
    [[ -f "$f" ]] || missing+=("$i")
  done
  if [[ ${#missing[@]} -gt 0 ]]; then
    echo "Download incomplete: ${#missing[@]} of $n shards missing (${missing[*]:0:8}${missing[8]:+ ...})." >&2
    echo "Resume:  tmux attach -t minillm-dl   or   bash ~/minillm/download.sh" >&2
    exit 1
  fi
fi

# --- binary + flags -----------------------------------------------------------
# Three binaries, three jobs (all verified against llama.cpp b10437):
#   llama-completion : plain generation. Honors --no-repack, exits when done,
#                      --perf prints load/prompt/eval timings. NO speculative
#                      decoding support at all (2 spec flags vs 41 on llama-cli).
#   llama-cli        : chat frontend. Has the full --spec-* set. Loops forever
#                      on stdin EOF UNLESS -st (single-turn) is given, which
#                      makes it answer once and exit -- verified.
#   llama-bench      : cannot run a model larger than RAM (repacks always).
# So: plain -> llama-completion; --draft -> llama-cli -st; --chat -> llama-cli.
if [[ $mode == chat || $draft == 1 ]]; then
  exe="$BIN/llama-cli"
else
  exe="$BIN/llama-completion"
fi
[[ -x "$exe" ]] || { echo "missing $exe -- run install.sh (step 4 builds llama.cpp)" >&2; exit 1; }

# --no-repack is mandatory above RAM and is not the default. -c 4096 keeps the
# KV cache small; raise it for long prompts once you know the RAM headroom.
args=(-m "$model" -t "$THREADS" -tb "$THREADS_BATCH" -ngl 0 --no-repack --load-mode mmap -c 4096)
if [[ $draft == 1 ]]; then
  d=$(findgguf "*MTP-ONLY*.gguf")
  [[ -n "$d" ]] || { echo "--draft: no *MTP-ONLY*.gguf under $CACHE" >&2; exit 1; }
  # MTP-ONLY is the model's own prediction head, not a standalone draft model:
  # llama.cpp must be told with --spec-type draft-mtp (per the a4lg README).
  # Each verify pass reads the always-hot weights ONCE for k+1 tokens, so a
  # deeper draft directly divides the un-cacheable cost. Speculative decoding
  # is mathematically lossless: output matches non-speculative decoding.
  args+=(-md "$d" --spec-type draft-mtp --spec-draft-n-max "${MINILLM_DRAFT_N:-5}")
fi

if [[ $lock == 1 ]]; then
  # Pin the always-hot tensors so the kernel CANNOT evict them. When
  # hot + per-token experts exceeds RAM, the page cache throws the hot half
  # out to make room for experts and re-reads all of it next token -- that is
  # the 30 s/token failure mode. Locking makes only experts stream.
  # Subsumes --warm: mlock faults every locked page in.
  if [[ -f /tmp/minillm-lock-hot.pid ]] && [[ -e "/proc/$(cat /tmp/minillm-lock-hot.pid 2>/dev/null)" ]]; then
    echo "hot set already pinned (pid $(cat /tmp/minillm-lock-hot.pid))"
  else
    echo "pinning hot set in RAM (needs sudo for RLIMIT_MEMLOCK)..."
    mkdir -p "$here/results"
    LOCKLOG="$here/results/lock.log"
    : > "$LOCKLOG" 2>/dev/null || LOCKLOG=$(mktemp)
    # The redirect must happen INSIDE the root shell: `sudo cmd > file` would
    # open the file as the invoking user (shellcheck SC2024).
    # MINILLM_LOCK_GB caps the pin. When the hot set exceeds RAM, pinning as
    # much as fits still removes those bytes from every future token's read.
    # Default: leave 6 GB for the expert cache, KV cache and compute buffers.
    _lockgb="${MINILLM_LOCK_GB:-}"
    if [[ -z "$_lockgb" ]]; then
      _memgb=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
      _lockgb=$(( _memgb > 10 ? _memgb - 6 : 4 ))
    fi
    echo "  pin budget: ${_lockgb} GB"
    # ulimit -l unlimited inside the ROOT shell. Root may raise its own hard
    # limit (CAP_SYS_RESOURCE), and mlock ignores the limit anyway once the
    # process has CAP_IPC_LOCK -- but setting it removes the last excuse and
    # makes the intent obvious in ps/strace.
    sudo -b sh -c "ulimit -l unlimited 2>/dev/null; exec python3 '$here/tools/lock_hot.py' '$model' --max-gb '$_lockgb' >>'$LOCKLOG' 2>&1" || true
    # Pinning N GB means faulting N GB in from the NVMe: minutes, not seconds.
    # Starting llama.cpp before that finishes just makes the two fight over the
    # same disk. Wait for the locker's summary line, streaming its progress.
    _printed=0
    for _ in $(seq 1 "${MINILLM_LOCK_WAIT:-1800}"); do
      _n=$(wc -l < "$LOCKLOG" 2>/dev/null || echo 0)
      if [[ "${_n:-0}" -gt "$_printed" ]]; then
        sed -n "$((_printed + 1)),${_n}p" "$LOCKLOG"
        _printed=$_n
      fi
      grep -qE "^locked |^Traceback|^No GGUF" "$LOCKLOG" 2>/dev/null && break
      sleep 1
    done
    if ! grep -q "^locked " "$LOCKLOG" 2>/dev/null; then
      echo "  WARNING: no lock confirmed after ${MINILLM_LOCK_WAIT:-1800}s."
      echo "  Continuing unpinned -- the numbers below are NOT a --lock result."
    fi
  fi
  echo
elif [[ $warm == 1 ]]; then
  # Read the always-hot tensors (attention/shared/router/embeddings/lm_head --
  # NOT the expert pool) with parallel readers. llama.cpp would otherwise fault
  # them in serially in 4 KB pages; measured 4.2x faster on a SATA disk.
  # Warming only helps first-token latency: these pages can still be evicted.
  # Use --lock instead when the model is much larger than RAM.
  echo "warming hot set with ${MINILLM_WARM_WORKERS:-8} parallel readers..."
  python3 "$here/tools/warm_gguf.py" "$model" --workers "${MINILLM_WARM_WORKERS:-8}" ||     echo "  (warm failed; continuing -- llama.cpp will fault pages in itself)"
  echo
fi

# A model larger than RAM streams constantly; the 128 KB default readahead
# turns each contiguous 12 MB expert into ~96 fault cycles.
_ra_dev=$(lsblk -no PKNAME "$(df --output=source "$CACHE" 2>/dev/null | tail -1)" 2>/dev/null | head -1)
if [[ -n "$_ra_dev" && -r "/sys/block/$_ra_dev/queue/read_ahead_kb" ]]; then
  _ra=$(cat "/sys/block/$_ra_dev/queue/read_ahead_kb")
  [[ "$_ra" -le 256 ]] && echo "note   : readahead on $_ra_dev is ${_ra} kB -- try 'bash tools/tune_io.sh' (free, no quality cost)"
fi

echo "model  : $model"
echo "binary : $exe"
echo "threads: $THREADS gen / $THREADS_BATCH batch   draft: $draft   mode: $mode   warm: $warm   lock: $lock"
echo "note   : first token is cold (page cache empty); the steady-state rate is what matters"
echo

if [[ $mode == chat ]]; then
  exec "$exe" "${args[@]}" "${extra[@]}"
fi

# Always time it on the wall clock. llama-cli's own meter rounds to ONE
# decimal, so it prints "Generation: 0.0 t/s" for anything slower than
# 20 s/token -- 45 s/token and 25 s/token look identical in it. That is
# useless on a machine where we are trying to move exactly that number.
# Snapshot kernel I/O counters around the run. /proc/vmstat pgpgin counts
# kilobytes paged in from block devices system-wide, so (after-before) is the
# EXACT number of bytes this run pulled off the disk -- no guessing from
# vmstat's rounded per-second column. pgmajfault is the number of faults that
# caused those reads, and bytes/fault says whether readahead is doing anything:
# 4 KB/fault means every fault went to disk alone, 2 MB/fault means it is
# batching properly. pswpin separates "reading the model" from "thrashing".
_iosnap() { awk '/^(pgpgin|pgmajfault|pswpin) /{printf "%s ", $2}' /proc/vmstat; }
# Sample the counters through the run as well, so the bytes can be attributed
# to load / prompt / generation separately. Only the generation number decides
# tokens per second; a whole-run total is dominated by the multi-minute load
# and says nothing useful.
mkdir -p "$here/results"
RUNLOG="$here/results/run.log"
TRACE="$here/results/io_trace.tsv"
python3 "$here/tools/iotrace.py" sample "$TRACE" --hz 4 &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT

io0=$(_iosnap)
t0=$(date +%s.%N)
# pipefail + set -e would abort on a non-zero exit before rc is read, and a
# trailing "|| true" resets PIPESTATUS. Turn -e off for exactly this pipeline.
set +e
if [[ $draft == 1 ]]; then
  # llama-cli path: -st = answer once then exit.
  "$exe" "${args[@]}" -n "$ntok" -st --perf --no-warmup -p "$prompt" "${extra[@]}" </dev/null 2>&1 | tee "$RUNLOG"
else
  "$exe" "${args[@]}" -n "$ntok" --perf -no-cnv --no-warmup -p "$prompt" "${extra[@]}" </dev/null 2>&1 | tee "$RUNLOG"
fi
rc=${PIPESTATUS[0]}
set -e
t1=$(date +%s.%N)
io1=$(_iosnap)
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true

# Timing math in python3, not awk: the report needs literal newlines and
# getting those through shell quoting into an awk program is a known way to
# produce "runaway string constant". python3 is already a hard dependency.
python3 - "$t0" "$t1" "$ntok" "$rc" $io0 $io1 <<'PYTIME'
import sys
a = sys.argv[1:]
t0, t1, n, rc = float(a[0]), float(a[1]), int(a[2]), a[3]
d = t1 - t0
GB = 1024 ** 3
print("")
print("---------------- wall clock ----------------")
print(f"  total      : {d:.1f} s for {n} tokens (exit {rc})")
if n > 0 and d > 0:
    print(f"  per token  : {d/n:.2f} s/token   =   {n/d:.4f} tok/s")
    print("  NOTE: includes model load. The per-phase table below")
    print("        separates generation from it -- read that one.")
# 4 counters before + 4 after when /proc/vmstat had all three keys.
if len(a) >= 10:
    pin0, mf0, sw0 = (int(x) for x in a[4:7])
    pin1, mf1, sw1 = (int(x) for x in a[7:10])
    read_b = (pin1 - pin0) * 1024
    faults = mf1 - mf0
    swapped = (sw1 - sw0) * 4096
    print("")
    print("---------------- disk I/O ------------------")
    print(f"  read from disk : {read_b / GB:8.2f} GB")
    if d > 0:
        print(f"  effective rate : {read_b / GB / d:8.3f} GB/s   <- the real ceiling")
    if n > 0:
        print(f"  per token      : {read_b / GB / n:8.2f} GB/token")
    if faults > 0:
        kb = read_b / faults / 1024
        print(f"  major faults   : {faults:,} = {kb:7.1f} kB per fault", end="")
        print("   (4 kB = readahead is doing NOTHING)" if kb < 16 else "")
    if swapped > 0:
        print(f"  SWAPPED IN     : {swapped / GB:8.2f} GB  <- anonymous memory is")
        print("                   thrashing; that is wasted bandwidth, not model reads")
PYTIME

# Per-phase attribution. This is the number to move.
if [[ -s "$TRACE" ]]; then
  python3 "$here/tools/iotrace.py" report "$TRACE" "$RUNLOG" || true
fi
exit $rc
