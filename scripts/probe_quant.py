"""
Measure a remote GGUF's hot/expert byte split WITHOUT downloading the model.

Why: on a machine smaller than the model, the number that decides everything
is the ALWAYS-HOT set (attention + shared expert + routers + lm_head). It is
touched every token, has no router in front of it, and therefore must fit in
RAM or it is re-read from disk on every single token. The routed-expert pool
can be far larger than RAM -- only ~2% of it is read per token.

Unsloth "Dynamic" quants deliberately keep attention at high precision and
push the experts low. That is right for a GPU where everything is resident,
and wrong for a RAM-constrained CPU box: on Qwen3.8-2.4T UD-IQ2_XXS the
attention is 6.06 bits/weight while the experts are 2.09, giving a 34.3 GB hot
set that does not fit in 31 GB of RAM. The total file size tells you nothing
about this -- you have to look inside.

A GGUF's header lists every tensor with its byte offset, and consecutive
offsets give sizes. So fetching the first few MB of each shard over HTTP range
requests is enough to compute the split exactly, for ~30 MB instead of ~400 GB.

  python scripts/probe_quant.py unsloth/Qwen3.8-2.4T-A95B-GGUF UD-Q1_0
  python scripts/probe_quant.py unsloth/Qwen3.8-2.4T-A95B-GGUF UD-IQ2_XXS --ram 31
"""

import argparse
import io
import os
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tools"))
from warm_gguf import SCALAR, T_STR, T_ARR, EXPERT_RE  # noqa: E402

GB = 1024 ** 3
HDR_BYTES = 12 * 1024 * 1024      # plenty for KV + tensor table


class _Reader:
    """Minimal file-like over a bytes buffer, so the GGUF parse can seek."""

    def __init__(self, blob):
        self.f = io.BytesIO(blob)

    def read(self, n):
        b = self.f.read(n)
        if len(b) < n:
            raise EOFError("header larger than the fetched prefix")
        return b

    def seek(self, n, whence=0):
        self.f.seek(n, whence)

    def tell(self):
        return self.f.tell()


def _str(f):
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", "replace")


def _skip_value(f, vtype):
    if vtype in SCALAR:
        f.read(struct.calcsize(SCALAR[vtype]))
    elif vtype == T_STR:
        _str(f)
    elif vtype == T_ARR:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in SCALAR:
            f.seek(struct.calcsize(SCALAR[etype]) * count, 1)
        elif etype == T_STR:
            for _ in range(count):
                _str(f)
        else:
            raise ValueError(f"nested array type {etype}")
    else:
        raise ValueError(f"unknown value type {vtype}")


def parse_header(blob, file_size):
    """-> [(name, offset, size)] using offset deltas; last size from file_size."""
    f = _Reader(blob)
    if f.read(4) != b"GGUF":
        raise ValueError("not a GGUF")
    struct.unpack("<I", f.read(4))
    n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
    alignment = 32
    for _ in range(n_kv):
        key = _str(f)
        (vtype,) = struct.unpack("<I", f.read(4))
        if key == "general.alignment" and vtype in SCALAR:
            raw = f.read(struct.calcsize(SCALAR[vtype]))
            alignment = struct.unpack("<" + SCALAR[vtype], raw)[0] or 32
        else:
            _skip_value(f, vtype)
    infos = []
    for _ in range(n_tensors):
        name = _str(f)
        (nd,) = struct.unpack("<I", f.read(4))
        f.seek(8 * nd, 1)
        f.seek(4, 1)
        (off,) = struct.unpack("<Q", f.read(8))
        infos.append((name, off))
    data_start = (f.tell() + alignment - 1) // alignment * alignment
    infos.sort(key=lambda t: t[1])
    out = []
    for i, (name, off) in enumerate(infos):
        end = infos[i + 1][1] if i + 1 < len(infos) else file_size - data_start
        out.append((name, data_start + off, end - off))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("subdir", help="quant folder, e.g. UD-Q1_0")
    ap.add_argument("--only", help="substring filter, e.g. Q4_K_M")
    ap.add_argument("--ram", type=float, default=31.0, help="machine RAM in GB")
    ap.add_argument("--os-reserve", type=float, default=4.0)
    args = ap.parse_args()

    from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
    import requests

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)
    # subdir "." (or "") means the .gguf files live at the repo root.
    pref = "" if args.subdir in (".", "") else args.subdir + "/"
    shards = sorted((s for s in info.siblings
                     if s.rfilename.startswith(pref)
                     and s.rfilename.endswith(".gguf")
                     and (pref or "/" not in s.rfilename)
                     and (not args.only or args.only in s.rfilename)),
                    key=lambda s: s.rfilename)
    if not shards:
        print(f"no .gguf under {args.subdir}/ in {args.repo}", file=sys.stderr)
        return 1

    total = sum(s.size or 0 for s in shards)
    print(f"{args.repo} / {args.subdir}")
    print(f"shards: {len(shards)}   total: {total / 1e9:.0f} GB")
    print(f"fetching {HDR_BYTES // (1024*1024)} MB of header per shard "
          f"(~{len(shards) * HDR_BYTES / 1e6:.0f} MB, not {total / 1e9:.0f} GB)\n")

    hot = exp = 0
    for s in shards:
        url = hf_hub_url(args.repo, s.rfilename)
        size = s.size or get_hf_file_metadata(url).size
        if size < 1024:
            continue
        r = requests.get(url, headers={"Range": f"bytes=0-{HDR_BYTES - 1}"},
                         timeout=60)
        r.raise_for_status()
        try:
            tensors = parse_header(r.content, size)
        except (ValueError, EOFError) as e:
            print(f"  {os.path.basename(s.rfilename)}: {e}", file=sys.stderr)
            continue
        h = sum(sz for n, _, sz in tensors if not EXPERT_RE.search(n))
        e_ = sum(sz for n, _, sz in tensors if EXPERT_RE.search(n))
        hot += h
        exp += e_
        print(f"  {os.path.basename(s.rfilename):<52} hot {h / GB:7.2f} GB  "
              f"exp {e_ / GB:8.2f} GB")

    usable = args.ram - args.os_reserve
    print(f"\nHOT (every token, cannot be evicted) : {hot / GB:8.2f} GB")
    print(f"EXPERT POOL (router picks ~2%/token) : {exp / GB:8.2f} GB")
    print(f"usable RAM ({args.ram:.0f} GB - {args.os_reserve:.0f} OS)         : "
          f"{usable:8.2f} GB")
    if hot / GB >= usable:
        print(f"\n  VERDICT: hot set is {hot / GB - usable:.1f} GB LARGER than usable RAM.")
        print("  It can never stay resident, so it is re-read from disk EVERY token.")
        print("  No cache, lock, or prefetch fixes this. This quant will not run well here.")
    else:
        print(f"\n  VERDICT: hot set FITS, leaving {usable - hot / GB:.1f} GB for an expert cache.")
        print("  Lock the hot set (tools/lock_hot.py) and only experts stream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
