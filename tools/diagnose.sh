#!/usr/bin/env bash
# One command that captures everything needed to explain a slow run.
#
#   bash tools/diagnose.sh              # full check, includes a timed generation
#   bash tools/diagnose.sh --no-gen     # skip the generation (fast)
#
# Paste the whole output. It answers, in order: is the machine swapping, how
# big is the hot set, is anything stealing CPU, is the lock actually held, and
# what is the disk doing during generation.

set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$here"
[[ -f .venv/bin/activate ]] && source .venv/bin/activate
if [[ -z "${MINILLM_LLAMA_BIN:-}" && -f "$HOME/.bashrc" ]]; then
  eval "$(sed -n '/# >>> MiniLLM >>>/,/# <<< MiniLLM <<</p' "$HOME/.bashrc" | grep '^export' || true)"
fi
BIN="${MINILLM_LLAMA_BIN:-$HOME/llama.cpp/build/bin}"
CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/minillm/hf}/hub}"
PATTERN="${MINILLM_MODEL_PATTERN:-Qwen3.8-2.4T}"
gen=1; [[ "${1:-}" == "--no-gen" ]] && gen=0

hr() { printf '\n=== %s %s\n' "$1" "$(printf '=%.0s' $(seq 1 $((60 - ${#1}))))"; }

hr "1. MACHINE"
echo "kernel : $(uname -r)"
echo "cpu    : $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | xargs)"
echo "cores  : $(nproc) logical, $(lscpu -p=Core,Socket 2>/dev/null | grep -vc '^#' || echo '?') listed"
grep -o -E 'avx512[a-z_]*|avx2' /proc/cpuinfo | sort -u | tr '\n' ' ' | sed 's/^/simd   : /'; echo
free -h | sed 's/^/       /'

hr "2. SWAP  (any swap IN/OUT during generation is fatal for speed)"
if [[ -n "$(swapon --show --noheadings 2>/dev/null)" ]]; then
  swapon --show | sed 's/^/       /'
  echo "swappiness: $(cat /proc/sys/vm/swappiness)"
  awk '/^(pswpin|pswpout)/{print "       "$0}' /proc/vmstat
  echo "  ^ if pswpin/pswpout climb while generating, RAM is oversubscribed."
else
  echo "       none active  (good)"
fi

hr "3. DISK"
root_src=$(df --output=source "$CACHE" 2>/dev/null | tail -1)
echo "model cache : $CACHE"
echo "backing dev : ${root_src:-unknown}"
lsblk -d -o NAME,MODEL,SIZE,ROTA,TRAN 2>/dev/null | grep -v loop | sed 's/^/       /'
for d in /sys/block/nvme*n1 /sys/block/sd?; do
  [[ -e "$d/queue/read_ahead_kb" ]] || continue
  echo "readahead ${d##*/}: $(cat "$d/queue/read_ahead_kb") kB   (128 is the default; larger helps mmap streaming)"
done
df -h "$CACHE" 2>/dev/null | tail -1 | sed 's/^/       /'

hr "4. WHAT IS RUNNING  (anything here competes for 4 cores)"
ps -eo pid,pcpu,pmem,rss,etime,comm,args --sort=-pcpu 2>/dev/null \
  | grep -E 'llama-|lock_hot|expert_cache|warm_gguf' | grep -v grep \
  | awk '{printf "       %-7s cpu%%=%-6s rss=%-9s %s\n", $1, $2, $4, $6}' || echo "       nothing"
for f in /tmp/minillm-lock-hot.pid /tmp/minillm-expert-cache.pid; do
  [[ -f "$f" ]] || continue
  pid=$(cat "$f" 2>/dev/null)
  if [[ -e "/proc/$pid" ]]; then
    echo "       $(basename "$f" .pid): pid $pid ALIVE, VmLck=$(awk '/VmLck/{print $2" "$3}' "/proc/$pid/status" 2>/dev/null)"
  else
    echo "       $(basename "$f" .pid): stale pidfile (process gone)"
  fi
done
echo "kernel-wide Mlocked: $(awk '/^Mlocked/{print $2" "$3}' /proc/meminfo)"

hr "5. MODEL + HOT/EXPERT SPLIT"
python3 tools/lock_hot.py --pattern "$PATTERN" --dry-run 2>&1 | sed 's/^/       /'

if [[ $gen == 1 ]]; then
  hr "6. TIMED GENERATION  (3 tokens; may take minutes -- that is the point)"
  M=$(find -L "$CACHE" -path '*/snapshots/*' -name "*${PATTERN}*-00001-of-*.gguf" 2>/dev/null | sort | head -1)
  [[ -n "$M" ]] || M=$(find -L "$CACHE" -path '*/snapshots/*' -name "*${PATTERN}*.gguf" ! -name '*-of-*' 2>/dev/null | head -1)
  if [[ -z "$M" ]]; then
    echo "       no model matching '$PATTERN' under $CACHE"
  else
    echo "       model: $(basename "$M")"
    b0=$(awk '/^pswpin|^pswpout/{s+=$2}END{print s+0}' /proc/vmstat)
    ( vmstat 5 24 > /tmp/diag_vmstat.txt 2>&1 ) &
    vm=$!
    timeout 900 "$BIN/llama-completion" -m "$M" -t "$(lscpu -p=Core,Socket 2>/dev/null | grep -vc '^#' || nproc)" \
      -n 3 -c 512 -ngl 0 --no-repack --load-mode mmap -no-cnv --no-warmup --perf \
      -p "The capital of France is" </dev/null 2>&1 | grep -E 'perf|eval time|load time|error|failed' | sed 's/^/       /'
    kill $vm 2>/dev/null; wait $vm 2>/dev/null
    b1=$(awk '/^pswpin|^pswpout/{s+=$2}END{print s+0}' /proc/vmstat)
    echo "       swap pages moved during run: $((b1 - b0))   (non-zero = RAM oversubscribed)"
    echo "       vmstat (bi = KB/s read from disk, id = % idle):"
    head -3 /tmp/diag_vmstat.txt | sed 's/^/         /'
    tail -8 /tmp/diag_vmstat.txt | sed 's/^/         /'
  fi
fi

hr "DONE"
echo "Paste everything above."
