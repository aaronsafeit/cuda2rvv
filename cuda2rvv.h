#ifndef CUDA2RVV_H
#define CUDA2RVV_H

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

// --- Thread context ---
extern size_t __tid;
extern size_t __blockDim;
extern size_t __gridDim;

// --- CUDA qualifiers (no-op) ---
#define __global__
#define __device__
#define __host__

// --- Thread/block indexing ---
#define threadIdx_x (__tid % __blockDim)
#define blockIdx_x (__tid / __blockDim)
#define blockDim_x (__blockDim)
#define gridDim_x  (__gridDim)

// --- Shared memory (static for now) ---
#define __shared__ static

// --- Synchronization ---
extern pthread_barrier_t __cuda_block_barrier;
#define __syncthreads() pthread_barrier_wait(&__cuda_block_barrier)

// --- Initialize synchronization context ---
static inline void init_cuda_context(size_t threads_per_block) {
    pthread_barrier_init(&__cuda_block_barrier, nullptr, threads_per_block);
}

#endif // CUDA2RVV_H
