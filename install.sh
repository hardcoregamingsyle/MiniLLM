#!/usr/bin/env bash
# MiniLLM installer for Linux (Debian / Ubuntu / Linux Mint).
#
# Does everything in SETUP_LINUX.md, in order, idempotently:
#   1. system packages          2. clone MiniLLM + Python venv
#   3. HF token + cache dirs    4. build llama.cpp for this CPU
#   5. roofline calibration     6. kernel tuning (optional, asks)
#   7. start the model download (~630 GB) in a tmux session
#
# Safe to re-run: every step checks whether it is already done and skips.
# If a step fails, the script stops and says which one; fix it and re-run.
#
# One-liner:
#   curl -fsSL https://raw.githubusercontent.com/hardcoregamingsyle/MiniLLM/main/install.sh | bash
# Or, if you already cloned the repo:
#   bash install.sh
#
# Options (env vars):
#   MINILLM_DIR=~/MiniLLM         where to clone
#   MINILLM_DATA=~/minillm        where models/caches go (put this on the big NVMe)
#   MINILLM_QUANT=UD-IQ2_XXS      which unsloth quant to fetch (UD-IQ2_XXS | UD-IQ2_XS)
#   MINILLM_SKIP_DOWNLOAD=1       set up everything but do not start the 630 GB fetch
#   MINILLM_SKIP_TUNING=1         skip the sudo sysctl / swap questions
#   MINILLM_YES=1                 answer yes to every prompt (unattended)

set -euo pipefail

# The whole installer lives inside main() and is invoked on the LAST line.
# Under `curl ... | bash`, bash executes as it streams -- a connection dropped
# mid-transfer would otherwise run a truncated script, possibly ending on a
# partial command. Wrapping in a function forces bash to parse everything
# before executing anything.
main() {

# ------------------------------------------------------------------ helpers
C_RESET=$'\033[0m'; C_BOLD=$'\033[1m'; C_GRN=$'\033[32m'; C_YLW=$'\033[33m'; C_RED=$'\033[31m'; C_CYN=$'\033[36m'
step()  { printf '\n%s%s==> %s%s\n' "$C_BOLD" "$C_CYN" "$*" "$C_RESET"; }
ok()    { printf '%s    ok: %s%s\n' "$C_GRN" "$*" "$C_RESET"; }
skip()  { printf '%s    skip: %s%s\n' "$C_YLW" "$*" "$C_RESET"; }
warn()  { printf '%s    WARN: %s%s\n' "$C_YLW" "$*" "$C_RESET"; }
die()   { printf '\n%s%sFAILED: %s%s\n' "$C_BOLD" "$C_RED" "$*" "$C_RESET" >&2; exit 1; }
ask()   { # ask "question" -> 0 for yes
  if [[ "${MINILLM_YES:-0}" == 1 ]]; then return 0; fi
  local a; read -r -p "    $* [y/N] " a </dev/tty || a=n
  [[ "$a" =~ ^[Yy] ]]
}
have()  { command -v "$1" >/dev/null 2>&1; }
gib()   { awk -v b="$1" 'BEGIN{printf "%.1f", b/1073741824}'; }

MINILLM_DIR="${MINILLM_DIR:-$HOME/MiniLLM}"
MINILLM_DATA="${MINILLM_DATA:-$HOME/minillm}"
MINILLM_QUANT="${MINILLM_QUANT:-UD-IQ2_XXS}"
REPO_URL="https://github.com/hardcoregamingsyle/MiniLLM.git"
LLAMA_DIR="$HOME/llama.cpp"
HF_HOME_DIR="$MINILLM_DATA/hf"
HF_CACHE_DIR="$HF_HOME_DIR/hub"
BASHRC="$HOME/.bashrc"
MARK_BEGIN="# >>> MiniLLM >>>"
MARK_END="# <<< MiniLLM <<<"

MODEL_REPO="unsloth/Qwen3.8-2.4T-A95B-GGUF"
DRAFT_REPO="a4lg/Qwen3.8-2.4T-A95B-MTP-ONLY-GGUF"
DRAFT_FILE="Qwen3.8-2.4T-A95B-MTP-ONLY-Q4_K_M.gguf"
# Bytes, verified against the Hub listing on 2026-08-15.
declare -A QUANT_BYTES=( [UD-IQ2_XXS]=656500000000 [UD-IQ2_XS]=730700000000 )
DRAFT_BYTES=19897255520

# ------------------------------------------------------------------ 0. preflight
step "0/7  Preflight"
[[ "$(uname -s)" == Linux ]] || die "This installer is for Linux. On Windows see README.md."
[[ $EUID -ne 0 ]] || die "Run as your normal user, not root. It will sudo when it needs to."
have apt-get || die "Needs apt (Debian/Ubuntu/Mint). Other distros: follow SETUP_LINUX.md by hand."
have sudo || die "sudo not found."

CPU_FLAGS=$(grep -m1 -o -E 'avx512[a-z_]*|avx2|avx_vnni|amx_[a-z]*' /proc/cpuinfo | sort -u | tr '\n' ' ')
grep -q avx2 /proc/cpuinfo || die "CPU has no AVX2. llama.cpp will be unusably slow. Stopping."
RAM_GB=$(awk '/MemTotal/{printf "%.0f", $2/1048576}' /proc/meminfo)
CORES=$(nproc)
printf '    CPU  : %s cores, SIMD: %s\n' "$CORES" "${CPU_FLAGS:-none}"
printf '    RAM  : %s GB\n' "$RAM_GB"

mkdir -p "$MINILLM_DATA"
FREE_B=$(df --output=avail -B1 "$MINILLM_DATA" | tail -1)
printf '    Disk : %s GiB free at %s\n' "$(gib "$FREE_B")" "$MINILLM_DATA"
NEED_B=$(( ${QUANT_BYTES[$MINILLM_QUANT]:-0} + DRAFT_BYTES ))
[[ $NEED_B -gt 0 ]] || die "Unknown MINILLM_QUANT='$MINILLM_QUANT'. Use UD-IQ2_XXS or UD-IQ2_XS."
# Credit what is already on disk, or every re-run would die here once a third
# of the model had landed -- and "just re-run to resume" would be a lie. The
# repo dirs do not exist on a fresh install, so guard each one.
HAVE_B=0
for d in "$HF_CACHE_DIR/models--${MODEL_REPO//\//--}" "$HF_CACHE_DIR/models--${DRAFT_REPO//\//--}"; do
  [[ -d "$d" ]] || continue
  b=$(du -sB1 "$d" 2>/dev/null | cut -f1) || b=0
  HAVE_B=$(( HAVE_B + ${b:-0} ))
done
REM_B=$(( NEED_B - HAVE_B )); (( REM_B > 0 )) || REM_B=0
[[ $HAVE_B -eq 0 ]] || printf '    Model: %s GiB already downloaded, ~%s GiB to go\n' "$(gib "$HAVE_B")" "$(gib "$REM_B")"
if [[ "${MINILLM_SKIP_DOWNLOAD:-0}" != 1 && $FREE_B -lt $(( REM_B + 30000000000 )) ]]; then
  die "Need ~$(gib "$REM_B") GiB more (+30 GiB margin) for $MINILLM_QUANT + MTP draft, have $(gib "$FREE_B") GiB free. Set MINILLM_DATA to a bigger disk, or MINILLM_SKIP_DOWNLOAD=1."
fi
[[ $RAM_GB -ge 16 ]] || warn "$RAM_GB GB RAM is below the 32 GB this project targets; expect it to be much slower."
[[ "$CPU_FLAGS" == *avx512* ]] || warn "No AVX-512: dequant will run the AVX2 path. Works, just slower."
ok "preflight"

# ------------------------------------------------------------------ 1. packages
step "1/7  System packages"
PKGS=(git build-essential cmake ninja-build python3 python3-pip python3-venv libcurl4-openssl-dev pkg-config tmux sysstat)
MISSING=()
for p in "${PKGS[@]}"; do dpkg -s "$p" >/dev/null 2>&1 || MISSING+=("$p"); done
if [[ ${#MISSING[@]} -eq 0 ]]; then
  skip "all present"
else
  printf '    installing: %s\n' "${MISSING[*]}"
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "${MISSING[@]}" || die "apt install failed"
  ok "installed"
fi

# ------------------------------------------------------------------ 2. clone + venv
step "2/7  MiniLLM repo + Python env"
if [[ -d "$MINILLM_DIR/.git" ]]; then
  skip "repo present at $MINILLM_DIR (git pull to update)"
else
  git clone -q "$REPO_URL" "$MINILLM_DIR" || die "git clone failed"
  ok "cloned to $MINILLM_DIR"
fi
cd "$MINILLM_DIR"
if [[ -x .venv/bin/python ]]; then
  skip "venv present"
else
  python3 -m venv .venv || die "venv creation failed (is python3-venv installed?)"
  ok "venv created"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt || die "pip install failed"
ok "python deps"

# ------------------------------------------------------------------ 3. env + token
step "3/7  Environment + Hugging Face token"
mkdir -p "$HF_CACHE_DIR"
# Rewrite our block in ~/.bashrc idempotently (strip old block, append new).
if grep -q "$MARK_BEGIN" "$BASHRC" 2>/dev/null; then
  sed -i "/$MARK_BEGIN/,/$MARK_END/d" "$BASHRC"
fi
{
  echo "$MARK_BEGIN"
  echo "export HF_HOME=\"$HF_HOME_DIR\""
  echo "export HF_HUB_CACHE=\"$HF_CACHE_DIR\""
  echo "export MINILLM_LLAMA_BIN=\"$LLAMA_DIR/build/bin\""
  echo "export MINILLM_DATA=\"$MINILLM_DATA\""
  echo "$MARK_END"
} >> "$BASHRC"
export HF_HOME="$HF_HOME_DIR" HF_HUB_CACHE="$HF_CACHE_DIR" MINILLM_LLAMA_BIN="$LLAMA_DIR/build/bin" MINILLM_DATA
ok "HF_HOME=$HF_HOME_DIR (written to ~/.bashrc)"

# Token: kept OUT of the block above so re-running never overwrites it, and
# stored in ~/.cache/huggingface/token by huggingface_hub itself, not in bashrc.
if [[ -n "${HF_TOKEN:-}" ]] || python - <<'EOF' 2>/dev/null
from huggingface_hub import whoami; whoami()
EOF
then
  ok "already authenticated as $(python -c 'from huggingface_hub import whoami; print(whoami()["name"])' 2>/dev/null || echo '(env HF_TOKEN)')"
else
  printf '    A Hugging Face token makes the 630 GB download much faster (higher rate limits).\n'
  printf '    Get one (read scope) at: https://huggingface.co/settings/tokens\n'
  if [[ "${MINILLM_YES:-0}" == 1 ]]; then
    warn "MINILLM_YES=1 and no HF_TOKEN in env: continuing unauthenticated (slow). Set HF_TOKEN=... to fix."
  else
    read -r -s -p "    Paste token (or press Enter to skip): " TOK </dev/tty || TOK=""
    echo
    if [[ -n "$TOK" ]]; then
      python -c "from huggingface_hub import login; login(token='$TOK', add_to_git_credential=False)" \
        && ok "authenticated as $(python -c 'from huggingface_hub import whoami; print(whoami()["name"])')" \
        || die "token rejected by the Hub. Check it and re-run."
    else
      warn "no token: downloads will be rate-limited. Re-run any time to add one."
    fi
  fi
fi

# ------------------------------------------------------------------ 4. llama.cpp
step "4/7  Build llama.cpp for this CPU"
if [[ -x "$LLAMA_DIR/build/bin/llama-completion" ]]; then
  skip "built at $LLAMA_DIR/build/bin ($("$LLAMA_DIR/build/bin/llama-completion" --version 2>&1 | head -1))"
else
  if [[ ! -d "$LLAMA_DIR/.git" ]]; then
    git clone -q --depth 1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_DIR" || die "llama.cpp clone failed"
  fi
  cd "$LLAMA_DIR"
  # -DGGML_NATIVE=ON: compile for exactly this CPU (AVX-512 if present).
  cmake -B build -G Ninja -DGGML_NATIVE=ON -DGGML_LTO=ON -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=ON \
        -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON >/dev/null || die "cmake configure failed"
  printf '    compiling with %s jobs (a few minutes)...\n' "$CORES"
  cmake --build build --config Release -j"$CORES" 2>&1 | grep -E "error|Error" && die "llama.cpp build failed" || true
  [[ -x build/bin/llama-completion ]] || die "build finished but llama-completion is missing"
  ok "built: $(build/bin/llama-completion --version 2>&1 | head -1)"
  cd "$MINILLM_DIR"
fi
# The harness needs llama-completion specifically; llama-bench and llama-cli
# both fail for models larger than RAM on recent builds (see README).
"$LLAMA_DIR/build/bin/llama-completion" --help 2>&1 | grep -q -- '--no-repack' \
  || warn "this llama.cpp lacks --no-repack; the harness needs it for models > RAM"

# ------------------------------------------------------------------ 5. roofline
step "5/7  Calibrate this machine (roofline)"
cd "$MINILLM_DIR"
ROOF_OUT="results/roofline_$(hostname -s).txt"
if [[ -s "$ROOF_OUT" ]]; then
  skip "already measured -> $ROOF_OUT (delete it to re-run)"
else
  mkdir -p results
  printf '    ~3 minutes: DRAM bandwidth, then O_DIRECT reads on %s\n' "$MINILLM_DATA"
  if python bench/roofline.py --scratch "$MINILLM_DATA/bench_scratch" | tee "$ROOF_OUT"; then
    ok "saved -> $ROOF_OUT"
    printf '    Paste dram_gbps / disk_gbps into minillm/machines.json (server-32gb-nvme)\n'
    printf '    and set "measured": true. capacity.py predictions then use real numbers.\n'
  else
    rm -f "$ROOF_OUT"
    warn "roofline failed (O_DIRECT unsupported on this filesystem?). Not fatal; continuing."
  fi
fi

# ------------------------------------------------------------------ 6. tuning
step "6/7  Kernel tuning (optional)"
if [[ "${MINILLM_SKIP_TUNING:-0}" == 1 ]]; then
  skip "MINILLM_SKIP_TUNING=1"
else
  if [[ "$(sysctl -n vm.swappiness 2>/dev/null || echo 60)" -le 5 ]]; then
    skip "vm.swappiness already low"
  elif ask "Set vm.swappiness=1 (keeps model pages in RAM over swap)? Needs sudo."; then
    sudo sysctl -q vm.swappiness=1
    echo 'vm.swappiness=1' | sudo tee /etc/sysctl.d/99-minillm.conf >/dev/null
    ok "vm.swappiness=1 (persisted)"
  fi
  if [[ -n "$(swapon --show --noheadings 2>/dev/null)" ]]; then
    printf '    Active swap:\n'; swapon --show | sed 's/^/      /'
    if ask "Turn swap OFF for this session? (it competes with model reads on the same disk)"; then
      sudo swapoff -a && ok "swap off until reboot" || warn "swapoff failed"
    fi
  else
    skip "no swap active"
  fi
fi

# ------------------------------------------------------------------ 7. download
step "7/7  Model download ($MINILLM_QUANT + MTP draft, ~$(gib "$NEED_B") GiB)"
if [[ "${MINILLM_SKIP_DOWNLOAD:-0}" == 1 ]]; then
  skip "MINILLM_SKIP_DOWNLOAD=1"
else
  # "Done" means EVERY shard is present, not just one. With --workers 8 the
  # shards land in arbitrary order, and llama.cpp opens all of them at load.
  # huggingface_hub keeps in-progress data in blobs/*.incomplete and only
  # creates the snapshots/ SYMLINK once a file is complete -- so counting
  # symlinks under snapshots/ (find -L ... -type f) is the correct signal.
  MODEL_SNAP="$HF_CACHE_DIR/models--${MODEL_REPO//\//--}/snapshots"
  DRAFT_SNAP="$HF_CACHE_DIR/models--${DRAFT_REPO//\//--}/snapshots"
  first=$(find -L "$MODEL_SNAP" -type f -name "*${MINILLM_QUANT}*-00001-of-*.gguf" 2>/dev/null | head -1 || true)
  model_done=0
  if [[ -n "$first" && "$first" =~ -00001-of-([0-9]{5})\.gguf$ ]]; then
    n_shards=$((10#${BASH_REMATCH[1]}))
    n_have=$(find -L "$MODEL_SNAP" -type f -name "*${MINILLM_QUANT}*-of-*.gguf" 2>/dev/null | wc -l)
    [[ $n_have -ge $n_shards ]] && model_done=1
    printf '    model shards present: %s / %s\n' "$n_have" "$n_shards"
  fi
  draft_done=0
  [[ -n "$(find -L "$DRAFT_SNAP" -type f -name "$DRAFT_FILE" 2>/dev/null | head -1)" ]] && draft_done=1

  # A finished (or crashed) download leaves its tmux session parked at a shell,
  # which must not block a resume forever. Alive-and-working means fetch_model
  # is actually running inside it.
  dl_running=0
  if tmux has-session -t minillm-dl 2>/dev/null; then
    pgrep -f "scripts/fetch_model.py" >/dev/null 2>&1 && dl_running=1 || tmux kill-session -t minillm-dl 2>/dev/null || true
  fi

  if [[ $model_done == 1 && $draft_done == 1 ]]; then
    skip "model (all shards) and draft already downloaded"
  elif [[ $dl_running == 1 ]]; then
    skip "download in progress in tmux session 'minillm-dl'  (attach: tmux attach -t minillm-dl)"
  else
    DL_SCRIPT="$MINILLM_DATA/download.sh"
    # Unquoted heredoc: paths are baked in NOW so the script is self-contained.
    # Runtime-only variables below are escaped (\$).
    cat > "$DL_SCRIPT" <<EOF
#!/usr/bin/env bash
# generated by install.sh -- resumable; re-run this file (or install.sh) if interrupted
set -uo pipefail
export HF_HOME="$HF_HOME_DIR" HF_HUB_CACHE="$HF_CACHE_DIR"
cd "$MINILLM_DIR" && source .venv/bin/activate
rc=0
echo "== main model: $MODEL_REPO / $MINILLM_QUANT"
python scripts/fetch_model.py --repo "$MODEL_REPO" --files "$MINILLM_QUANT/*" --workers 8 || rc=\$?
if [[ \$rc -eq 0 ]]; then
  echo "== MTP draft: $DRAFT_REPO"
  python scripts/fetch_model.py --repo "$DRAFT_REPO" --files "$DRAFT_FILE" || rc=\$?
fi
echo
if [[ \$rc -eq 0 ]]; then
  echo "== DONE. Model + draft complete. Next:  cd $MINILLM_DIR && bash run.sh"
else
  echo "== FAILED (exit \$rc). Usually a network drop. Resume with:  bash $DL_SCRIPT"
fi
echo "   (window stays open so you can read this; Ctrl-b d detaches, 'exit' closes)"
exec bash
EOF
    chmod +x "$DL_SCRIPT"
    tmux new-session -d -s minillm-dl "bash '$DL_SCRIPT'"
    ok "started in tmux session 'minillm-dl'"
    printf '    watch  : tmux attach -t minillm-dl     (Ctrl-b d to detach)\n'
    printf '    size   : du -sh %s\n' "$HF_CACHE_DIR/models--${MODEL_REPO//\//--}"
    printf '    resume : re-run this installer, or: bash %s\n' "$DL_SCRIPT"
    warn "Disable sleep/suspend in Power settings while this runs."
  fi
fi

# ------------------------------------------------------------------ done
step "Done"
cat <<EOF
    Everything is set up. Open a NEW terminal (or: source ~/.bashrc) and then:

      cd $MINILLM_DIR
      bash run.sh                # runs the model once the download finishes
      bash run.sh --draft        # same, with MTP speculative decoding
      bash run.sh --chat         # interactive chat

    Predictions for this machine:
      source .venv/bin/activate
      python -m minillm.capacity report --machine server-32gb-nvme --model qwen3.8-2.4t-a95b

    Re-run this installer any time; it only does what is missing.
EOF
}

main "$@"
