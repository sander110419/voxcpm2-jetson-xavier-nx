// cuda_compact_shim.c — LD_PRELOAD interposer for Tegra fragmentation OOMs.
// On cudaMalloc failure: drop caches + compact memory, then retry (x2).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>

typedef cudaError_t (*malloc_fn)(void**, size_t);
static malloc_fn real_malloc = NULL;

static void compact_memory(void) {
    int rc = system("sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory");
    (void)rc;
}

cudaError_t cudaMalloc(void** ptr, size_t size) {
    if (!real_malloc)
        real_malloc = (malloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
    cudaError_t e = real_malloc(ptr, size);
    if (e != cudaSuccess) {
        fprintf(stderr, "[shim] cudaMalloc(%zu MB) failed (%s) — compacting + retry\n",
                size >> 20, cudaGetErrorString(e));
        compact_memory();
        e = real_malloc(ptr, size);
        if (e != cudaSuccess) {
            usleep(500000);
            compact_memory();
            e = real_malloc(ptr, size);
        }
        fprintf(stderr, "[shim] retry %s\n", e == cudaSuccess ? "SUCCEEDED" : "still failing");
        // a failed-then-retried cudaMalloc leaves cudaErrorOutOfMemory latched in
        // the thread's last-error slot; ggml reads cudaGetLastError after later ops
        // and would abort on the stale error. Always clear after handling.
        (void)cudaGetLastError();
    }
    return e;
}
