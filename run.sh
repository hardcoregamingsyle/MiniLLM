#!/usr/bin/env bash
# MiniLLM runner. Finds the model, picks the right llama.cpp binary and flags,
# and runs it. This is the thing to use day-to-day; the harness in bench/ is
# for measurement.
#
#   bash run.sh                       benchmark: 32 tokens, prints tok/s
#   bash run.sh --draft               same, with MTP speculative decoding
#   bash run.sh --chat                interactive chat with the model
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
PATTERN="${MINILLM_MODEL_PATTERN:-Qwen3.8-2.4T-A95B-UD-IQ2_XXS}"
THREADS="${MINILLM_THREADS:-$(nproc)}"

mode=bench; draft=0; prompt="Explain, in three sentences, why mixture-of-experts models can run on machines with far less RAM than the model size."; ntok=32; extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --chat)  mode=chat; shift ;;
    --draft) draft=1; shift ;;
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
args=(-m "$model" -t "$THREADS" -ngl 0 --no-repack --load-mode mmap -c 4096)
if [[ $draft == 1 ]]; then
  d=$(findgguf "*MTP-ONLY*.gguf")
  [[ -n "$d" ]] || { echo "--draft: no *MTP-ONLY*.gguf under $CACHE" >&2; exit 1; }
  # MTP-ONLY is the model's own prediction head, not a standalone draft model:
  # llama.cpp must be told with --spec-type draft-mtp (per the a4lg README).
  args+=(-md "$d" --spec-type draft-mtp --spec-draft-n-max 4)
fi

echo "model  : $model"
echo "binary : $exe"
echo "threads: $THREADS   draft: $draft   mode: $mode"
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
