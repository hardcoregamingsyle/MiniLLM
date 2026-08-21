/*
 * mmap_shim -- fix two things llama.cpp does to its model mapping that are
 * wrong when the model is 20x larger than RAM. No source patch, no rebuild of
 * llama.cpp: LD_PRELOAD this and it intercepts mmap/mmap64.
 *
 * 1. MAP_POPULATE over the whole file.
 *
 *    llama-model.cpp calls ml.init_mappings(true, ...) with prefetch hardcoded
 *    to SIZE_MAX, so llama_mmap passes MAP_POPULATE for the entire mapping.
 *    On a model that fits in RAM that is a good idea -- it front-loads the
 *    faults. On a 612 GB file mapped into 31 GB it means the kernel walks the
 *    whole file, reading all 612 GB from disk and evicting nearly all of it
 *    again, before a single token is produced. Measured on the target laptop:
 *    451 s of "load time", which is 612 GB at 1.36 GB/s. Every run pays it,
 *    and the page cache is left holding the tail of the file, which is the
 *    part least likely to be wanted next.
 *
 *    Stripping the flag makes the mapping lazy. Nothing else changes: the
 *    mapping is identical, pages just arrive on demand.
 *
 * 2. No readahead advice at all.
 *
 *    do_sync_mmap_readahead() keeps a per-file mmap_miss counter. Every fault
 *    that misses increments it, and once it passes 100 the function returns
 *    early with fault-around only -- no I/O readahead, ever again, for that
 *    file. A 612 GB model streaming through a small page cache latches it
 *    almost immediately, and from then on every major fault reads exactly one
 *    4 KB page. That is why raising the block device's read_ahead_kb does
 *    nothing: the code path that would consult it is no longer reached.
 *
 *    The VM_SEQ_READ branch returns BEFORE that gate, calling
 *    page_cache_sync_ra() with the full ra_pages. MADV_SEQUENTIAL is the only
 *    way to set VM_SEQ_READ. llama.cpp never calls madvise during inference,
 *    and its one MADV_RANDOM is gated behind ggml_is_numa() -- unreachable on
 *    a single-socket machine.
 *
 *    llama.cpp does call posix_fadvise(POSIX_FADV_SEQUENTIAL) on the fd, which
 *    sets f_ra.ra_pages = bdi->ra_pages * 2. That is the same struct the fault
 *    path reads, so raising read_ahead_kb DOES matter -- but only once
 *    VM_SEQ_READ gets us past the latch. Run tools/tune_io.sh before the
 *    model, not after: ra_pages is captured when fadvise is called.
 *
 * Both changes are advisory. Neither alters a byte of the model, the
 * arithmetic, or the output.
 *
 * Build:  gcc -shared -fPIC -O2 -o runtime/mmap_shim.so runtime/mmap_shim.c -ldl
 * Use:    LD_PRELOAD=$PWD/runtime/mmap_shim.so llama-completion ...
 *
 * Env:
 *   MINILLM_SHIM_POPULATE=1   keep MAP_POPULATE (to measure what it costs)
 *   MINILLM_SHIM_ADVICE=      sequential (default) | normal | willneed | none
 *   MINILLM_SHIM_MIN_GB=0.25  only touch mappings at least this large
 *   MINILLM_SHIM_VERBOSE=1    log every mapping decision to stderr
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef void *(*mmap_fn)(void *, size_t, int, int, int, off_t);
typedef void *(*mmap64_fn)(void *, size_t, int, int, int, off64_t);

static mmap_fn real_mmap;
static mmap64_fn real_mmap64;

/*
 * Resolve at load time, not on first call. dlsym() may allocate, allocation
 * may call mmap, and that would re-enter this shim while the pointer it needs
 * is still NULL -- the classic LD_PRELOAD recursion. A constructor runs before
 * the program does any of its own mapping.
 */
__attribute__((constructor))
static void resolve(void)
{
    if (!real_mmap)
        real_mmap = (mmap_fn)dlsym(RTLD_NEXT, "mmap");
    if (!real_mmap64)
        real_mmap64 = (mmap64_fn)dlsym(RTLD_NEXT, "mmap64");
}

/*
 * Last resort if resolution has still not happened (an mmap issued from
 * inside the dynamic loader before our constructor ran). Going straight to
 * the kernel cannot recurse. x86-64's SYS_mmap takes the offset in bytes.
 */
static void *raw_mmap(void *a, size_t l, int p, int f, int fd, off64_t o)
{
    /* glibc's syscall() already turns the kernel's -errno into -1 + errno,
       so this is the ordinary libc convention, not the raw kernel one. */
    long r = syscall(SYS_mmap, a, l, p, f, fd, (long)o);

    return r == -1 ? MAP_FAILED : (void *)r;
}

static int cfg_done;
static int cfg_keep_populate;
static int cfg_advice = MADV_SEQUENTIAL;
static int cfg_have_advice = 1;
static size_t cfg_min_bytes = 256UL << 20;
static int cfg_verbose;

static void cfg_init(void)
{
    const char *s;

    if (cfg_done)
        return;
    cfg_done = 1;

    s = getenv("MINILLM_SHIM_POPULATE");
    cfg_keep_populate = (s && *s == '1');

    s = getenv("MINILLM_SHIM_VERBOSE");
    cfg_verbose = (s && *s == '1');

    s = getenv("MINILLM_SHIM_MIN_GB");
    if (s && *s) {
        double gb = atof(s);
        if (gb > 0)
            cfg_min_bytes = (size_t)(gb * (double)(1UL << 30));
    }

    s = getenv("MINILLM_SHIM_ADVICE");
    if (s && *s) {
        if (!strcmp(s, "sequential"))    cfg_advice = MADV_SEQUENTIAL;
        else if (!strcmp(s, "willneed")) cfg_advice = MADV_WILLNEED;
        else if (!strcmp(s, "normal"))   cfg_advice = MADV_NORMAL;
        else if (!strcmp(s, "none"))     cfg_have_advice = 0;
        else fprintf(stderr, "[mmap_shim] unknown MINILLM_SHIM_ADVICE=%s, "
                             "using sequential\n", s);
    }

    if (cfg_verbose)
        fprintf(stderr, "[mmap_shim] loaded: populate=%s advice=%d min=%.2f GB\n",
                cfg_keep_populate ? "keep" : "strip",
                cfg_have_advice ? cfg_advice : -1,
                (double)cfg_min_bytes / (double)(1UL << 30));
}

/*
 * Only the model mapping is interesting: large, file-backed, read-only.
 * Leaving everything else alone means the shim cannot perturb allocator
 * arenas, thread stacks, or the compute buffers -- those are anonymous and
 * writable, so they fail two of these tests each.
 *
 * The floor is 256 MB rather than 1 GB because a split GGUF is mapped one
 * shard at a time and shard size is a packaging choice, not a fixed number.
 * A 1 GB floor would silently skip a model that happens to be cut finer,
 * and silently doing nothing is the worst failure mode a shim can have.
 */
static int is_model_mapping(size_t len, int prot, int flags, int fd)
{
    return len >= cfg_min_bytes
        && fd >= 0
        && !(flags & MAP_ANONYMOUS)
        && (prot & PROT_READ)
        && !(prot & PROT_WRITE);
}

static void *shim_mmap(int is64, void *addr, size_t len, int prot,
                       int flags, int fd, off64_t off)
{
    int touched;
    void *ret;

    cfg_init();
    touched = is_model_mapping(len, prot, flags, fd);

    if (touched && !cfg_keep_populate)
        flags &= ~MAP_POPULATE;

    if (is64 && real_mmap64)
        ret = real_mmap64(addr, len, prot, flags, fd, off);
    else if (!is64 && real_mmap)
        ret = real_mmap(addr, len, prot, flags, fd, (off_t)off);
    else
        ret = raw_mmap(addr, len, prot, flags, fd, off);

    if (ret == MAP_FAILED || !touched)
        return ret;

    if (cfg_have_advice && madvise(ret, len, cfg_advice) != 0 && cfg_verbose)
        perror("[mmap_shim] madvise");

    if (cfg_verbose)
        fprintf(stderr, "[mmap_shim] %.2f GB mapping: populate=%s advice=%d\n",
                (double)len / (double)(1UL << 30),
                cfg_keep_populate ? "kept" : "stripped",
                cfg_have_advice ? cfg_advice : -1);
    return ret;
}

void *mmap(void *addr, size_t len, int prot, int flags, int fd, off_t off)
{
    if (!real_mmap)
        resolve();
    return shim_mmap(0, addr, len, prot, flags, fd, (off64_t)off);
}

/*
 * glibc exports mmap64 separately and a caller may bind to either. On x86-64
 * they are aliases, but intercepting only one leaves a hole on any build where
 * they are not -- and a hole here is silent: the shim would simply do nothing.
 */
void *mmap64(void *addr, size_t len, int prot, int flags, int fd, off64_t off)
{
    if (!real_mmap64)
        resolve();
    return shim_mmap(1, addr, len, prot, flags, fd, off);
}
