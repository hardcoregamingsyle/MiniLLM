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
    LOCKLOG=/tmp/minillm-lock.log
    : > "$LOCKLOG"
    # The redirect must happen INSIDE the root shell: `sudo cmd > file` would
    # open the file as the invoking user (shellcheck SC2024).
    sudo -b sh -c "exec python3 '$here/tools/lock_hot.py' '$model' >>'$LOCKLOG' 2>&1" || true
    for _ in $(seq 1 60); do
      grep -qE "^locked |RLIMIT_MEMLOCK|failed" "$LOCKLOG" 2>/dev/null && break
      sleep 1
    done
    sed -n '1,12p' "$LOCKLOG"
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
elif [[ $draft == 1 ]]; then
  # llama-cli path: -st = answer once then exit. It prints a rounded
  # "[ Prompt: X t/s | Generation: Y t/s ]" (one decimal; 0.0 = under 0.05).
  exec "$exe" "${args[@]}" -n "$ntok" -st --perf --no-warmup -p "$prompt" "${extra[@]}" </dev/null
else
  exec "$exe" "${args[@]}" -n "$ntok" --perf -no-cnv --no-warmup -p "$prompt" "${extra[@]}" </dev/null
fi
