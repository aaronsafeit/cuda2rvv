#ifndef CUDA2RVV_UNIFIED_H
#define CUDA2RVV_UNIFIED_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

/******************************************************************************
 * CUDA Unified Memory Shim (RVV & Host Shared Heap Memory)
 * ---------------------------------------------------------
 * - Compatible with CUDA's unified memory interface
 * - Internally maps to malloc/free for now
 * - Prefetch, stream, and advice support stubs included
 ******************************************************************************/

/** Allocate unified memory (shared between host and device simulation) */
static inline void* cudaMallocManaged(size_t size) {
    void* ptr = aligned_alloc(64, size); // alignment helps vector ops
    if (!ptr) {
        fprintf(stderr, "[cudaMallocManaged] Allocation failed: %zu bytes (%s)\n", size, strerror(errno));
        return NULL;
    }
    memset(ptr, 0, size); // zero-initialize for safety
    return ptr;
}

/** Free unified memory */
static inline void cudaFree(void* ptr) {
    if (ptr) free(ptr);
}


/******************************************************************************
 * Memcpy Simulation (All memory is unified)
 ******************************************************************************/

typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
} cudaMemcpyKind;

static inline int cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    if (!dst || !src) {
        fprintf(stderr, "[cudaMemcpy] Error: NULL pointer detected (kind=%d)\n", kind);
        return -1;
    }
    memcpy(dst, src, count);
    return 0;
}


/******************************************************************************
 * cudaMemPrefetchAsync / cudaMemAdvise (Future extensibility)
 ******************************************************************************/

static inline int cudaMemPrefetchAsync(const void* devPtr, size_t count, int device, pthread_t stream) {
    (void)devPtr; (void)count; (void)device; (void)stream;
    // Simulated environment: no action needed
    return 0;
}

static inline int cudaMemAdvise(const void* devPtr, size_t count, int advice, int device) {
    (void)devPtr; (void)count; (void)advice; (void)device;
    return 0;
}

#define cudaMemAdviseSetReadMostly 1
#define cudaMemAdviseUnsetReadMostly 2


/******************************************************************************
 * Stream & Synchronization Compatibility (Pthreads-based Stub)
 ******************************************************************************/

typedef pthread_t cudaStream_t;

static inline int cudaStreamCreate(cudaStream_t* stream) {
    (void)stream;
    // Stub: no real threading model tied to streams yet
    return 0;
}

static inline int cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    return 0;
}

static inline int cudaStreamSynchronize(cudaStream_t stream) {
    (void)stream;
    return 0;
}

/** Full device-wide synchronization (stub for compatibility) */
static inline int cudaDeviceSynchronize() {
    // In RVV simulation, all ops are synchronous for now
    return 0;
}


/******************************************************************************
 * Debug / Utility Tools for Memory Inspection
 ******************************************************************************/

static inline void cudaPrintBytes(const void* ptr, size_t count) {
    const unsigned char* p = (const unsigned char*)ptr;
    printf("[cuda2rvv] Memory dump (%zu bytes):\n", count);
    for (size_t i = 0; i < count; ++i) {
        printf("%02X ", p[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (count % 16 != 0) printf("\n");
}

#endif // CUDA2RVV_UNIFIED_H
