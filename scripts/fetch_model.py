"""
Download a model to D: with the traps handled.

Two traps this avoids:
  1. HF_HOME defaults to C:\\Users\\...\\.cache, and C: has ~9 GB free. Everything
     is redirected to D: before huggingface_hub is imported.
  2. openai/gpt-oss-120b carries metal/ and original/ subdirectories holding
     redundant copies of the whole checkpoint. Pulling them would need ~180 GB.
     Only root-level files are fetched.

Resumable: re-run after an interruption and it continues where it stopped.

  python scripts/fetch_model.py                      # default: gpt-oss-120b
  python scripts/fetch_model.py --repo Qwen/Qwen3.6-35B-A3B
  python scripts/fetch_model.py --dry-run            # size check only
"""

import argparse
import os
import shutil
import sys

# If HF_HOME is already set (the Linux guide puts it in ~/.bashrc), respect it.
# Otherwise fall back to a per-OS default on the big drive.
DEFAULT_ROOT = os.environ.get("HF_HOME") or (
    r"D:\minillm\hf" if sys.platform == "win32" else os.path.expanduser("~/minillm/hf"))


def _redirect_cache(root):
    """Must run before huggingface_hub is imported -- it reads env at import."""
    os.makedirs(root, exist_ok=True)
    os.environ["HF_HOME"] = root
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(root, "hub"))
    os.environ.setdefault("HF_XET_CACHE", os.path.join(root, "xet"))
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")


# Root-level files only. Directory copies (metal/, original/) are excluded by
# requiring that no pattern contains a separator and by the explicit ignore list.
ALLOW = [
    "*.safetensors",
    "*.json",
    "*.jinja",
    "*.txt",
    "*.model",
]
IGNORE = [
    "metal/*", "metal/**",
    "original/*", "original/**",
    "*/*.safetensors",
]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="openai/gpt-oss-120b")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--files", nargs="*", default=None,
                    help="exact filenames to fetch instead of the default weight set")
    args = ap.parse_args()

    _redirect_cache(args.root)
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    info = api.model_info(args.repo, files_metadata=True)

    import fnmatch
    wanted, skipped = [], []
    for f in info.siblings:
        if args.files:
            # Exact names or shell globs. "UD-IQ2_XXS/*" selects one quant's
            # shard directory out of a multi-quant repo.
            hit = any(f.rfilename == p or fnmatch.fnmatch(f.rfilename, p) for p in args.files)
            (wanted if hit else skipped).append(f)
            continue
        top_level = "/" not in f.rfilename
        is_weight_or_cfg = f.rfilename.endswith(
            (".safetensors", ".json", ".jinja", ".txt", ".model")
        )
        (wanted if (top_level and is_weight_or_cfg) else skipped).append(f)

    if args.files and not wanted:
        print(f"ABORT: none of {args.files} exist in {args.repo}")
        return 1

    total = sum(f.size or 0 for f in wanted)
    skipped_size = sum(f.size or 0 for f in skipped)

    # Bytes already on disk for this repo count toward `total` but are no longer
    # needed from the network -- and they are ALSO already subtracted from free
    # space. Charging them twice made this guard abort on a resumable download.
    cache_dir = os.path.join(
        os.environ["HF_HUB_CACHE"], "models--" + args.repo.replace("/", "--"))
    have = 0
    if os.path.isdir(cache_dir):
        for dp, _, fs in os.walk(cache_dir):
            for f in fs:
                try:
                    have += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
    remaining = max(0, total - have)

    # Measure the filesystem that will actually hold the files. On Windows that
    # is the drive letter; on Linux splitdrive() returns '' and would silently
    # measure '/', which is wrong whenever /home or the model dir is its own
    # mount. disk_usage on the (existing) target directory itself is correct
    # on both.
    probe = os.path.abspath(args.root)
    while not os.path.isdir(probe):
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    mount = (os.path.splitdrive(probe)[0] + os.sep) if os.name == "nt" else probe

    print(f"repo        : {args.repo}")
    print(f"destination : {args.root}")
    print(f"fetching    : {len(wanted)} files, {human(total)}")
    print(f"already have: {human(have)}  ->  {human(remaining)} still to fetch")
    print(f"excluded    : {len(skipped)} files, {human(skipped_size)}")
    print(f"free at {mount}: {human(free)}")

    if remaining > free * 0.92:
        print(f"\nABORT: need {human(remaining)} more but only {human(free)} free.")
        return 1
    print(f"headroom after download: {human(free - remaining)}\n")

    if args.dry_run:
        for f in sorted(wanted, key=lambda x: -(x.size or 0))[:8]:
            print(f"  {human(f.size or 0):>10}  {f.rfilename}")
        return 0

    path = snapshot_download(
        repo_id=args.repo,
        allow_patterns=args.files if args.files else ALLOW,
        ignore_patterns=None if args.files else IGNORE,
        max_workers=args.workers,
    )
    print(f"\nsnapshot at: {path}")

    got = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fs in os.walk(path) for f in fs
    )
    print(f"on disk    : {human(got)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
