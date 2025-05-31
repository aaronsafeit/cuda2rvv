#ifndef CUDA2RVV_UNIFIED_H
#define CUDA2RVV_UNIFIED_H

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <pthread.h>

// --- Unified Memory simulation for RVV-based CUDA shim ---

// CUDA Unified Memory allocations map to malloc for now
// TODO: Extend with real managed memory or shared virtual memory support
static inline void* cudaMallocManaged(size_t size) {
    void *ptr = malloc(size);
    if (!ptr) {
        fprintf(stderr, "cudaMallocManaged: Allocation failed for %zu bytes\n", size);
        return NULL;
    }
    // In CUDA, unified memory can be accessed by host and device
    // Here host and "device" (emulated) share the same pointer
    return ptr;
}

static inline void cudaFree(void* ptr) {
    if (ptr) {
        free(ptr);
    }
}

// cudaMemcpyKind enumeration for compatibility
typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice,
    cudaMemcpyDeviceToHost,
    cudaMemcpyDeviceToDevice,
    cudaMemcpyDefault
} cudaMemcpyKind;

// Simulated cudaMemcpy: since host and device memory are unified,
// this is just a memcpy. Future extension can add async and memory domain checks.
static inline int cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind) {
    (void)kind;  // Unused, all memory is unified in this shim
    if (!dst || !src) {
        fprintf(stderr, "cudaMemcpy: Null pointer encountered\n");
        return -1;
    }
    memcpy(dst, src, count);
    return 0; // success
}

// cudaMemPrefetchAsync stub: no-op in this shim, placeholder for future prefetch support
static inline int cudaMemPrefetchAsync(const void* devPtr, size_t count, int device, pthread_t stream) {
    (void)devPtr; (void)count; (void)device; (void)stream;
    // Prefetch not implemented yet, treat as no-op
    return 0;
}

// cudaMemAdvise stub: no-op in shim, can be extended for page migration hints
static inline int cudaMemAdvise(const void* devPtr, size_t count, int advice, int device) {
    (void)devPtr; (void)count; (void)advice; (void)device;
    return 0;
}

// Memory advice constants (subset)
#define cudaMemAdviseSetReadMostly 1
#define cudaMemAdviseUnsetReadMostly 2

// Stream type for compatibility
typedef pthread_t cudaStream_t;

// cudaStreamCreate and cudaStreamDestroy map to pthread create/join in your threading shim
static inline int cudaStreamCreate(cudaStream_t* stream) {
    (void)stream;
    // Stream creation stub, expand with your threading model
    return 0;
}

static inline int cudaStreamDestroy(cudaStream_t stream) {
    (void)stream;
    // Stream destroy stub
    return 0;
}

#endif // CUDA2RVV_UNIFIED_H
