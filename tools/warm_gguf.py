"""
Warm a GGUF's always-hot tensors into the OS page cache, with parallel readers.

Why this exists: llama.cpp mmaps the model and faults pages in on demand,
serially, as single 4 KB misses on whatever thread touches them first. On a
cold start that is the slowest possible way to load the ~10 GB hot set
(attention, shared expert, routers, norms, embeddings, lm_head). This tool
reads exactly those byte ranges -- and NOT the routed-expert pool -- with N
parallel readers in large chunks, which saturates the disk instead of the
fault handler. After it runs, the hot set is resident and stays resident,
because it is re-touched every token (LRU keeps truly-hot pages).

Expert tensors (ffn_*_exps.*) are deliberately skipped: they are 95% of the
file and the router decides per token which 2% of them matter. Warming them
all would evict the hot set and waste an hour.

  python tools/warm_gguf.py --pattern Qwen3.8-2.4T-A95B-UD-IQ2_XXS
  python tools/warm_gguf.py /path/to/model-00001-of-00015.gguf   # all shards
  python tools/warm_gguf.py --pattern ... --dry-run              # just report

Works on Linux and Windows; stdlib only.
"""

import argparse
import os
import re
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor

GB = 1024 ** 3
MB = 1024 ** 2

# GGUF metadata value type ids -> struct format (scalars)
SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
          7: "?", 10: "Q", 11: "q", 12: "d"}
T_STR, T_ARR = 8, 9

# Routed-expert tensors: blk.N.ffn_gate_exps.*, ffn_down_exps.*, ffn_up_exps.*
# (weights and, on gpt-oss, biases). Everything else is the always-hot set.
EXPERT_RE = re.compile(r"_exps\.")

SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$", re.I)

HF_CACHE = os.environ.get("HF_HUB_CACHE") or (
    os.path.join(os.environ["HF_HOME"], "hub") if os.environ.get("HF_HOME") else
    (r"D:\minillm\hf\hub" if sys.platform == "win32"
     else os.path.expanduser("~/minillm/hf/hub")))


class Gguf:
    """Just enough GGUF parsing to get (tensor name, offset, size) per file."""

    def __init__(self, path):
        self.path = path
        self.tensors = []      # [(name, abs_offset, size)]
        self.alignment = 32
        self._parse()

    def _parse(self):
        size = os.path.getsize(self.path)
        with open(self.path, "rb", buffering=1024 * 1024) as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"not a GGUF file: {self.path}")
            (version,) = struct.unpack("<I", f.read(4))
            if version < 2:
                raise ValueError(f"GGUF v{version} not supported (v2+ only)")
            n_tensors, n_kv = struct.unpack("<QQ", f.read(16))

            for _ in range(n_kv):
                key = self._str(f)
                (vtype,) = struct.unpack("<I", f.read(4))
                val = self._value(f, vtype, want=key == "general.alignment")
                if key == "general.alignment" and isinstance(val, int) and val > 0:
                    self.alignment = val

            infos = []
            for _ in range(n_tensors):
                name = self._str(f)
                (n_dims,) = struct.unpack("<I", f.read(4))
                f.seek(8 * n_dims, 1)          # dims
                f.seek(4, 1)                   # ggml type
                (off,) = struct.unpack("<Q", f.read(8))
                infos.append((name, off))

            a = self.alignment
            data_start = (f.tell() + a - 1) // a * a

        infos.sort(key=lambda t: t[1])
        for i, (name, off) in enumerate(infos):
            end = infos[i + 1][1] if i + 1 < len(infos) else size - data_start
            self.tensors.append((name, data_start + off, end - off))

    @staticmethod
    def _str(f):
        (n,) = struct.unpack("<Q", f.read(8))
        return f.read(n).decode("utf-8", "replace")

    def _value(self, f, vtype, want=False):
        if vtype in SCALAR:
            fmt = SCALAR[vtype]
            raw = f.read(struct.calcsize(fmt))
            return struct.unpack("<" + fmt, raw)[0] if want else None
        if vtype == T_STR:
            self._str(f)
            return None
        if vtype == T_ARR:
            (etype,) = struct.unpack("<I", f.read(4))
            (count,) = struct.unpack("<Q", f.read(8))
            if etype in SCALAR:
                f.seek(struct.calcsize(SCALAR[etype]) * count, 1)
            elif etype == T_STR:
                for _ in range(count):
                    self._str(f)
            else:
                raise ValueError(f"nested array of type {etype} unsupported")
            return None
        raise ValueError(f"unknown GGUF value type {vtype}")


def find_files(pattern):
    """All GGUF files matching pattern under the HF cache (shards + singles).
    Follows symlinks: HF stores snapshots/<rev>/<file> as links into blobs/."""
    hits = set()
    for dp, _, fs in os.walk(HF_CACHE, followlinks=True):
        if f"{os.sep}snapshots{os.sep}" not in dp + os.sep:
            continue
        for fn in fs:
            if fn.endswith(".gguf") and pattern.lower() in fn.lower():
                hits.add(os.path.join(dp, fn))
    return sorted(hits)


def sibling_shards(path):
    """Given any one shard, return all shards of that model; else [path]."""
    m = SHARD_RE.search(os.path.basename(path))
    if not m:
        return [path]
    n = int(m.group(2))
    out = []
    for i in range(1, n + 1):
        p = path.replace(f"-{m.group(1)}-of-", f"-{i:05d}-of-")
        if os.path.exists(p):
            out.append(p)
    return out


def warm_ranges(jobs, workers, chunk=16 * MB):
    """jobs: [(path, offset, size)]. Read every byte once, in parallel."""
    pieces = []
    for path, off, size in jobs:
        end = off + size
        while off < end:
            pieces.append((path, off, min(chunk, end - off)))
            off += chunk
    # Interleave pieces from different files/regions so parallel readers do not
    # queue behind each other on one region.
    pieces.sort(key=lambda p: p[1] % (workers * chunk))

    # Each piece opens its own handle: no shared-fd seek races, and it works on
    # Windows, which has no os.pread. Opens are trivially cheap next to 16 MB
    # of I/O per piece.
    def read_piece(p):
        path, off, n = p
        got = 0
        with open(path, "rb", buffering=0) as f:
            if hasattr(os, "posix_fadvise"):
                os.posix_fadvise(f.fileno(), off, n, os.POSIX_FADV_WILLNEED)
            f.seek(off)
            while got < n:
                b = f.read(min(4 * MB, n - got))
                if not b:
                    break
                got += len(b)
        return got

    total = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for got in pool.map(read_piece, pieces):
            total += got
    return total, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="GGUF file(s); one shard implies all")
    ap.add_argument("--pattern", help="find files matching this under the HF cache")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--include-embd", action="store_true", default=True,
                    help="warm token_embd too (default on; it is small and read every token)")
    args = ap.parse_args()

    files = []
    for f in args.files:
        files.extend(sibling_shards(f))
    if args.pattern:
        for f in find_files(args.pattern):
            files.extend(sibling_shards(f))
    files = sorted(set(files))
    if not files:
        print("No GGUF files given or matched. Use a path or --pattern.", file=sys.stderr)
        return 1

    jobs, hot_b, exp_b, n_hot, n_exp = [], 0, 0, 0, 0
    for path in files:
        try:
            g = Gguf(path)
        except ValueError as e:
            print(f"  skip {os.path.basename(path)}: {e}", file=sys.stderr)
            continue
        for name, off, size in g.tensors:
            if EXPERT_RE.search(name):
                exp_b += size; n_exp += 1
            else:
                hot_b += size; n_hot += 1
                jobs.append((path, off, size))

    print(f"files          : {len(files)}")
    print(f"hot tensors    : {n_hot:6d}  {hot_b / GB:8.2f} GB   <- will warm")
    print(f"expert tensors : {n_exp:6d}  {exp_b / GB:8.2f} GB   <- skipped (router decides)")
    if args.dry_run:
        return 0

    total, dt = warm_ranges(jobs, args.workers)
    print(f"warmed {total / GB:.2f} GB in {dt:.1f}s  ({total / GB / dt:.2f} GB/s, "
          f"{args.workers} readers)")
    print("hot set is now resident; it stays resident because every token touches it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
