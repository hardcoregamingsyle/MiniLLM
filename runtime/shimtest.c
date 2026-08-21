/*
 * shimtest -- prove what mmap_shim.so actually did, from the kernel's records
 * rather than from the shim's own log line.
 *
 * Creates a sparse file, maps it read-only with MAP_POPULATE, and reports the
 * two facts that distinguish shimmed from unshimmed:
 *
 *   Rss       MAP_POPULATE faults the whole mapping in up front, so an
 *             unshimmed run shows Rss ~= the mapping size. With the flag
 *             stripped the mapping is lazy and Rss is ~0.
 *   VmFlags   "sr" is set by MADV_SEQUENTIAL (VM_SEQ_READ) -- the flag that
 *             gets the fault path past do_sync_mmap_readahead()'s mmap_miss
 *             latch. An unshimmed mapping has neither sr nor rr.
 *
 * Build: gcc -O2 -o runtime/shimtest runtime/shimtest.c
 * Use:   runtime/shimtest /tmp/probe.bin 512
 */

#define _GNU_SOURCE
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

/* smaps entries start with "start-end perms ..."; find the one covering addr. */
static int report(unsigned long addr)
{
    char line[512];
    FILE *f = fopen("/proc/self/smaps", "r");
    int in_range = 0, found = 0;

    if (!f) {
        perror("open smaps");
        return 1;
    }
    while (fgets(line, sizeof line, f)) {
        unsigned long lo, hi;

        if (sscanf(line, "%lx-%lx", &lo, &hi) == 2 && strchr(line, ' ')) {
            in_range = (addr >= lo && addr < hi);
            if (in_range) {
                found = 1;
                printf("MAPPING %lx-%lx  %.2f MB\n", lo, hi,
                       (double)(hi - lo) / (1024.0 * 1024.0));
            }
            continue;
        }
        if (!in_range)
            continue;
        if (!strncmp(line, "Rss:", 4) || !strncmp(line, "VmFlags:", 8) ||
            !strncmp(line, "Locked:", 7))
            fputs(line, stdout);
    }
    fclose(f);
    if (!found)
        fprintf(stderr, "no smaps entry covering %lx\n", addr);
    return found ? 0 : 1;
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "/tmp/shimtest.bin";
    size_t mb = argc > 2 ? (size_t)atoi(argv[2]) : 512;
    size_t len = mb * 1024UL * 1024UL;
    void *p;
    int fd, rc;

    fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        perror("open");
        return 1;
    }
    /*
     * Write real bytes rather than ftruncate to a sparse file. Whether a
     * read fault over a hole in a shared mapping allocates a page-cache folio
     * is filesystem-dependent, and the Rss assertion below depends on it, so
     * do not make the test rest on that.
     */
    {
        static char buf[1 << 20];
        size_t done = 0;

        memset(buf, 0xA5, sizeof buf);
        while (done < len) {
            size_t n = len - done < sizeof buf ? len - done : sizeof buf;
            ssize_t w = write(fd, buf, n);

            if (w <= 0) {
                perror("write");
                return 1;
            }
            done += (size_t)w;
        }
        fsync(fd);
    }

    p = mmap(NULL, len, PROT_READ, MAP_SHARED | MAP_POPULATE, fd, 0);
    if (p == MAP_FAILED) {
        perror("mmap");
        return 1;
    }
    printf("requested %zu MB with MAP_POPULATE\n", mb);
    rc = report((unsigned long)p);

    munmap(p, len);
    close(fd);
    unlink(path);
    return rc;
}
