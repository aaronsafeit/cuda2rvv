#ifndef CUDA2RV_UNIFIED_H
#define CUDA2RV_UNIFIED_H

#include <riscv_vector.h>
#include <stdatomic.h>
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

/*******************************************************************************
 * Threading Context (simulated)
 ******************************************************************************/
extern size_t __tid;        // Current thread ID (linear)
extern size_t __blockDim;   // Threads per block
extern size_t __gridDim;    // Number of blocks

#define threadIdx_x (__tid % __blockDim)
#define blockIdx_x  (__tid / __blockDim)
#define blockDim_x  (__blockDim)
#define gridDim_x   (__gridDim)

/*******************************************************************************
 * Unified Memory (UM) Abstractions
 ******************************************************************************/
#define __managed__    // Managed memory annotation removed (no-op)

// Simple unified memory allocation macros (placeholders, map to malloc/free)
#include <stdlib.h>
#define cudaMallocManaged(ptr, size)  (*(ptr) = malloc(size), (*(ptr) != NULL) ? 0 : 1)
#define cudaFree(ptr)                  free(ptr)

/*******************************************************************************
 * Synchronization Primitives
 ******************************************************************************/
extern pthread_barrier_t __cuda_block_barrier;
extern pthread_barrier_t __cuda_warp_barrier;

#define __syncthreads()        pthread_barrier_wait(&__cuda_block_barrier)
#define __syncwarp(mask)       pthread_barrier_wait(&__cuda_warp_barrier)

static inline void init_cuda_context(size_t threads_per_block, size_t warps_per_block) {
    pthread_barrier_init(&__cuda_block_barrier, NULL, threads_per_block);
    pthread_barrier_init(&__cuda_warp_barrier, NULL, warps_per_block);
}

/*******************************************************************************
 * Atomic Operations (map to RVA via C11 atomics)
 ******************************************************************************/
static inline int atomicAdd(int* addr, int val) {
    return atomic_fetch_add((_Atomic int*)addr, val);
}

static inline int atomicMin(int* addr, int val) {
    int old = atomic_load((_Atomic int*)addr);
    while (val < old && !atomic_compare_exchange_weak((_Atomic int*)addr, &old, val)) {}
    return old;
}

static inline int atomicMax(int* addr, int val) {
    int old = atomic_load((_Atomic int*)addr);
    while (val > old && !atomic_compare_exchange_weak((_Atomic int*)addr, &old, val)) {}
    return old;
}

/*******************************************************************************
 * Warp Primitives: __shfl_sync and __ballot_sync
 ******************************************************************************/
static inline size_t set_vl_warp() {
    return vsetvl_e32m1(32);
}

static inline int __shfl_sync(uint32_t mask, int val, int srcLane) {
    size_t vl = set_vl_warp();
    vint32m1_t vals = vmv_v_x_i32m1(val, vl);
    vint32m1_t res = vrgather_vx_i32m1(vals, srcLane, vl);
    int result;
    vse32_v_i32m1(&result, res, vl);
    return result;
}

static inline vint32m1_t slide1up(int new_val, vint32m1_t vec) {
    size_t vl = set_vl_warp();
    return vslide1up_vx_i32m1(vec, new_val, vl);
}

// Placeholder for mask bit extraction - platform-specific implementation required
static inline int vget_m_b32(vbool32_t mask, size_t idx) {
    uint32_t raw_mask = *((uint32_t*)&mask);
    return (raw_mask & (1U << idx)) != 0;
}

static inline uint32_t __ballot_sync(uint32_t mask, int predicate) {
    size_t vl = set_vl_warp();
    vbool32_t pred_mask = predicate ? (vbool32_t){~0U} : (vbool32_t){0};
    uint32_t bitmask = 0;
    for (size_t i = 0; i < vl; ++i) {
        if (vget_m_b32(pred_mask, i))
            bitmask |= (1U << i);
    }
    return bitmask;
}

/*******************************************************************************
 * CUDA qualifiers no-ops for native compilation
 ******************************************************************************/
#define __global__
#define __device__
#define __host__
#define __shared__ static

#endif // CUDA2RV_UNIFIED_H
