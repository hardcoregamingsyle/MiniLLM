"""
Roofline calibration for a CPU MoE streaming runtime.

Measures the three constants every design decision keys off:
  A. DRAM streaming read bandwidth  (sets the floor for RAM-resident weights)
  B. Storage read bandwidth at expert-block granularity, page-cache bypassed
     (sets the cost of an expert cache miss)
  C. Achievable int4-dequant + GEMV throughput (loose lower bound)

Run:  python bench/roofline.py [--scratch D:\\minillm_bench]
"""

import argparse
import ctypes
import ctypes.wintypes as wt
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

MB = 1 << 20
GB = 1 << 30


# ----------------------------------------------------------------------------
# A. DRAM streaming read bandwidth
# ----------------------------------------------------------------------------

def bench_dram(buf_bytes=512 * MB, threads=(1, 2, 4, 8), reps=5):
    """Sum a large uint64 buffer. Pure sequential read, no writeback traffic.

    Buffer is far larger than the 6 MB L3, so every access misses to DRAM.
    numpy's reduction releases the GIL, so thread scaling is real.
    """
    n = buf_bytes // 8
    arr = np.ones(n, dtype=np.uint64)
    arr.sum()  # fault in every page before timing

    results = {}
    for nt in threads:
        chunks = np.array_split(np.arange(nt), nt)
        bounds = [(int(n * i / nt), int(n * (i + 1) / nt)) for i in range(nt)]
        best = 0.0
        with ThreadPoolExecutor(max_workers=nt) as pool:
            for _ in range(reps):
                t0 = time.perf_counter()
                list(pool.map(lambda b: arr[b[0]:b[1]].sum(), bounds))
                dt = time.perf_counter() - t0
                best = max(best, buf_bytes / dt / GB)
        results[nt] = best
        print(f"  DRAM read  {nt:>2} thread(s):  {best:8.2f} GB/s")
    del arr
    return results


# ----------------------------------------------------------------------------
# B. Storage read bandwidth, page cache bypassed
#
# Two backends behind one interface. Both bypass the OS page cache so the
# number is the DEVICE, not RAM:
#   Windows : CreateFileW + FILE_FLAG_NO_BUFFERING
#   Linux   : open(O_DIRECT) + preadv into an aligned buffer
# Both need reads that are sector-aligned in offset, length, and buffer address.
# ----------------------------------------------------------------------------

IS_WIN = sys.platform == "win32"
ALIGN = 4096  # covers 4Kn and 512e devices


class _UnbufferedReader:
    """open / read_at / close over a page-cache-bypassing handle."""

    def __init__(self, path, block, random_hint):
        self.block = block
        if IS_WIN:
            self._open_win(path, random_hint)
        else:
            self._open_posix(path)

    # -- Windows ------------------------------------------------------------
    def _open_win(self, path, random_hint):
        GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING = 0x80000000, 1, 3
        NO_BUF, SEQ, RAND = 0x20000000, 0x08000000, 0x10000000
        self._k32 = k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateFileW.restype = wt.HANDLE
        k32.CreateFileW.argtypes = [wt.LPCWSTR, wt.DWORD, wt.DWORD, ctypes.c_void_p,
                                    wt.DWORD, wt.DWORD, wt.HANDLE]
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD, wt.DWORD]
        k32.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, wt.DWORD]
        k32.ReadFile.argtypes = [wt.HANDLE, ctypes.c_void_p, wt.DWORD,
                                 ctypes.POINTER(wt.DWORD), ctypes.c_void_p]
        k32.SetFilePointerEx.argtypes = [wt.HANDLE, ctypes.c_longlong,
                                         ctypes.POINTER(ctypes.c_longlong), wt.DWORD]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        hint = RAND if random_hint else SEQ
        self.h = k32.CreateFileW(path, GENERIC_READ, FILE_SHARE_READ, None,
                                 OPEN_EXISTING, NO_BUF | hint, None)
        if self.h == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        self.buf = k32.VirtualAlloc(None, self.block, 0x3000, 0x04)  # commit+reserve, RW
        if not self.buf:
            raise ctypes.WinError(ctypes.get_last_error())

    # -- POSIX --------------------------------------------------------------
    def _open_posix(self, path):
        import mmap
        flags = os.O_RDONLY | getattr(os, "O_DIRECT", 0)
        if not hasattr(os, "O_DIRECT"):
            raise RuntimeError("os.O_DIRECT unavailable; this bench needs Linux")
        self.fd = os.open(path, flags)
        # mmap gives a page-aligned buffer, which O_DIRECT requires.
        self.buf = mmap.mmap(-1, self.block)

    # -- shared -------------------------------------------------------------
    def read_at(self, offset):
        if IS_WIN:
            k32 = self._k32
            got = wt.DWORD(0)
            if not k32.SetFilePointerEx(self.h, ctypes.c_longlong(offset), None, 0):
                raise ctypes.WinError(ctypes.get_last_error())
            if not k32.ReadFile(self.h, ctypes.c_void_p(self.buf), self.block,
                                ctypes.byref(got), None):
                raise ctypes.WinError(ctypes.get_last_error())
            return got.value
        return os.preadv(self.fd, [self.buf], offset)

    def close(self):
        if IS_WIN:
            self._k32.VirtualFree(ctypes.c_void_p(self.buf), 0, 0x8000)
            self._k32.CloseHandle(self.h)
        else:
            self.buf.close()
            os.close(self.fd)


def _make_testfile(path, size):
    if os.path.exists(path) and os.path.getsize(path) == size:
        return
    print(f"  creating {size // MB} MB test file at {path} ...")
    chunk = np.random.randint(0, 256, 8 * MB, dtype=np.uint8).tobytes()
    with open(path, "wb") as f:
        written = 0
        while written < size:
            f.write(chunk[:min(len(chunk), size - written)])
            written += len(chunk)
        f.flush()
        os.fsync(f.fileno())


def _run_reads(path, size, block, is_random, nthreads, budget_bytes):
    """Issue reads of `block` bytes until budget_bytes consumed. Returns GB/s."""
    nblocks = size // block
    per_thread = max(1, (budget_bytes // block) // nthreads)

    def worker(seed):
        r = _UnbufferedReader(path, block, is_random)
        rng = random.Random(seed)
        total = 0
        try:
            for i in range(per_thread):
                off = (rng.randrange(nblocks) if is_random else (i % nblocks)) * block
                total += r.read_at(off)
        finally:
            r.close()
        return total

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=nthreads) as pool:
        moved = sum(pool.map(worker, range(nthreads)))
    dt = time.perf_counter() - t0
    return moved / dt / GB


def bench_storage(scratch, file_size=3 * GB, budget=768 * MB):
    """Sweep block size and queue depth. Block sizes bracket real expert sizes."""
    os.makedirs(scratch, exist_ok=True)
    path = os.path.join(scratch, "roofline.bin")
    _make_testfile(path, file_size)

    blocks = [256 * 1024, 1 * MB, 4 * MB, 16 * MB, 64 * MB]
    results = {}
    print(f"\n  {'block':>8} {'seq QD1':>10} {'rand QD1':>10} "
          f"{'rand QD4':>10} {'rand QD8':>10}")
    for b in blocks:
        row = {
            ("seq", 1): _run_reads(path, file_size, b, False, 1, budget),
            ("rand", 1): _run_reads(path, file_size, b, True, 1, budget),
            ("rand", 4): _run_reads(path, file_size, b, True, 4, budget),
            ("rand", 8): _run_reads(path, file_size, b, True, 8, budget),
        }
        results[b] = row
        label = f"{b // 1024} KB" if b < MB else f"{b // MB} MB"
        print(f"  {label:>8} {row[('seq',1)]:9.3f}  {row[('rand',1)]:9.3f}  "
              f"{row[('rand',4)]:9.3f}  {row[('rand',8)]:9.3f}   GB/s")
    return path, results


# ----------------------------------------------------------------------------
# C. int4 dequant + GEMV, loose lower bound
# ----------------------------------------------------------------------------

def bench_gemv(rows=4096, cols=4096, reps=20):
    """Unpack nibble-packed int4 to fp32 and do a matrix-vector product.

    numpy has no fused int4 kernel, so this is a floor, not a ceiling: a real
    AVX2 kernel that dequantizes inside registers should beat it substantially.
    What it does establish is that compute is not the binding constraint.
    """
    packed = np.random.randint(0, 256, (rows, cols // 2), dtype=np.uint8)
    scales = np.random.rand(rows).astype(np.float32)
    x = np.random.rand(cols).astype(np.float32)
    lo_lut = (np.arange(256, dtype=np.uint8) & 0x0F).astype(np.float32) - 8.0
    hi_lut = (np.arange(256, dtype=np.uint8) >> 4).astype(np.float32) - 8.0

    best = 0.0
    for _ in range(reps):
        t0 = time.perf_counter()
        w = np.empty((rows, cols), dtype=np.float32)
        w[:, 0::2] = lo_lut[packed]
        w[:, 1::2] = hi_lut[packed]
        y = (w @ x) * scales
        dt = time.perf_counter() - t0
        best = max(best, (2.0 * rows * cols) / dt / 1e9)
    weight_bytes = rows * cols // 2
    print(f"  int4 GEMV {rows}x{cols}: {best:7.2f} GFLOP/s effective "
          f"({weight_bytes / MB:.1f} MB of packed weights per pass)")
    return best


# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scratch",
                    default="D:\\minillm_bench" if IS_WIN
                    else os.path.expanduser("~/minillm/bench_scratch"),
                    help="directory on the drive under test -- put it on the "
                         "same disk the model will live on")
    ap.add_argument("--keep", action="store_true", help="keep the test file")
    args = ap.parse_args()

    print("=" * 68)
    print("MiniLLM roofline calibration")
    print("=" * 68)

    print("\n[A] DRAM streaming read")
    dram = bench_dram()

    print("\n[B] Storage read, page cache bypassed (FILE_FLAG_NO_BUFFERING)")
    path, disk = bench_storage(args.scratch)

    print("\n[C] int4 dequant + GEMV (numpy floor)")
    gemv = bench_gemv()

    peak_dram = max(dram.values())
    best_disk = max(r[("rand", 8)] for r in disk.values())

    print("\n" + "=" * 68)
    print("Token-budget implications")
    print("=" * 68)
    print(f"  peak DRAM read        : {peak_dram:6.2f} GB/s")
    print(f"  best random disk read : {best_disk:6.3f} GB/s")
    print(f"  disk/DRAM ratio       : {peak_dram / best_disk:6.1f}x slower\n")
    for target in (0.5, 1.0, 2.0):
        print(f"  at {target:.1f} s/token -> {target * peak_dram:6.2f} GB from RAM "
              f"OR {target * best_disk * 1024:7.0f} MB from disk")
    print("\n  A token's active weights must fit the RAM figure; every byte")
    print("  that misses the expert cache is charged at the disk figure.")

    if not args.keep and os.path.exists(path):
        os.remove(path)
        print(f"\n  removed {path}")


if __name__ == "__main__":
    if not IS_WIN and not hasattr(os, "O_DIRECT"):
        sys.exit("Needs Windows (FILE_FLAG_NO_BUFFERING) or Linux (O_DIRECT).")
    main()
