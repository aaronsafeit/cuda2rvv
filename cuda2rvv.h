#ifndef CUDA2RVV_H
#define CUDA2RVV_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

// --- Thread context (to be managed by runtime) ---
extern size_t __tid;        // Thread ID within the grid
extern size_t __blockDim;   // Number of threads per block
extern size_t __gridDim;    // Number of blocks in the grid

// --- CUDA qualifiers (empty macros for compatibility) ---
#define __global__
#define __device__
#define __host__

// --- Thread/block indexing mappings ---
#define threadIdx_x (__tid % __blockDim)
#define blockIdx_x (__tid / __blockDim)
#define blockDim_x (__blockDim)
#define gridDim_x  (__gridDim)

// --- Shared memory mapping ---
// Maps __shared__ keyword to static storage, can be extended for true shared memory support
#define __shared__ static

// --- Synchronization primitives ---
// Pthread barrier simulates __syncthreads() across threads in a block
extern pthread_barrier_t __cuda_block_barrier;
#define __syncthreads() pthread_barrier_wait(&__cuda_block_barrier)

// --- Initialization / teardown of synchronization context ---
static inline void init_cuda_context(size_t threads_per_block) {
    pthread_barrier_init(&__cuda_block_barrier, NULL, threads_per_block);
}

static inline void destroy_cuda_context() {
    pthread_barrier_destroy(&__cuda_block_barrier);
}

// --- Utility macros for explicit vector length control (RVV) ---
// Vector length 'vl' for RVV intrinsics (set by runtime/kernel launcher)
extern size_t __rvv_vl;
#define RVV_VL (__rvv_vl)

// --- Memory fence and atomic helpers ---
// RISC-V fence for ordering memory operations, extendable for atomic operations
#define __threadfence() asm volatile("fence rw, rw" ::: "memory")
#define __threadfence_block() asm volatile("fence rw, rw" ::: "memory")

// --- Lane ID and warp size helpers ---
// CUDA lane id (thread's index within warp), assuming warp size = 32 by default
static inline size_t __laneid() {
    return __tid % 32;
}
static inline size_t __warpsize() {
    return 32;
}

#endif // CUDA2RVV_H
