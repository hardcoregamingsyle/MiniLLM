#!/usr/bin/env bash
# Raise block-device readahead so mmap faults pull large chunks instead of 128 KB.
#
# Why this matters for a model larger than RAM: an expert is a contiguous
# ~12 MB region, so reading one IS sequential access. At the 128 KB default
# that is ~96 separate readahead cycles per expert; at 4 MB it is 3. Fewer,
# larger reads is exactly what an NVMe wants, and it costs nothing in quality.
#
# BUT this does nothing on its own, and measuring it alone will show nothing.
# do_sync_mmap_readahead() counts misses per file in mmap_miss and, once that
# passes 100, returns before it ever consults ra_pages -- so on a model far
# larger than RAM the latch closes almost immediately and readahead is over.
# Only the VM_SEQ_READ branch returns before that gate, and only MADV_SEQUENTIAL
# sets VM_SEQ_READ. That is what runtime/mmap_shim.c does, and run.sh loads it
# by default. Raise readahead BEFORE launching the model: llama.cpp's own
# posix_fadvise(POSIX_FADV_SEQUENTIAL) reads bdi->ra_pages at call time and
# doubles it, so a later change does not reach the mapping already in use.
#
#   bash tools/tune_io.sh            # show current, set 4 MB (asks for sudo)
#   bash tools/tune_io.sh 8192       # set 8 MB instead
#   bash tools/tune_io.sh --revert   # back to 128 KB
#
# Not persistent across reboots by design -- measure first, then decide. The
# script prints the udev rule to make it stick.

set -uo pipefail
CACHE="${HF_HUB_CACHE:-${HF_HOME:-$HOME/minillm/hf}/hub}"

want=4096
case "${1:-}" in
  --revert) want=128 ;;
  ''      ) ;;
  *       ) want="$1" ;;
esac

src=$(df --output=source "$CACHE" 2>/dev/null | tail -1)
[[ -n "$src" ]] || { echo "cannot resolve the device behind $CACHE" >&2; exit 1; }
# /dev/nvme0n1p2 -> nvme0n1 ; /dev/sda3 -> sda. Readahead is a whole-disk knob.
part=$(basename "$src")
disk=$(lsblk -no PKNAME "$src" 2>/dev/null | head -1)
[[ -n "$disk" ]] || disk="${part%p[0-9]*}"
[[ -n "$disk" ]] || disk="${part%[0-9]*}"
sysfs="/sys/block/$disk/queue/read_ahead_kb"

if [[ ! -w "$sysfs" && ! -r "$sysfs" ]]; then
  echo "no readahead knob at $sysfs" >&2
  echo "On LVM/LUKS the mapper device is what df reports; set it on the underlying disk." >&2
  exit 1
fi

echo "model cache : $CACHE"
echo "device      : $src  ->  disk $disk"
echo "readahead   : $(cat "$sysfs") kB  (current)"

if [[ "$(cat "$sysfs")" == "$want" ]]; then
  echo "already $want kB; nothing to do."
else
  echo "$want" | sudo tee "$sysfs" >/dev/null || { echo "failed to set" >&2; exit 1; }
  echo "readahead   : $(cat "$sysfs") kB  (new)"
fi

cat <<EOF

Measure it -- do not assume. Run the SAME generation before and after:
    bash run.sh -n 4

Watch 'bi' in vmstat while it runs (KB/s read from disk):
    vmstat 5

If bi rises and s/token falls, keep it. To make it survive reboot:
    echo 'ACTION=="add|change", KERNEL=="$disk", ATTR{queue/read_ahead_kb}="$want"' \\
      | sudo tee /etc/udev/rules.d/60-minillm-readahead.rules

Readahead is not free in every workload: it can waste bandwidth on truly
random access. For contiguous 12 MB expert tensors it should help; for a
workload with a good cache hit rate it matters less.
EOF
