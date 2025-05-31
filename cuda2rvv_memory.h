#ifndef CUDA2RVV_MEMORY_H
#define CUDA2RVV_MEMORY_H

#include <riscv_vector.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

/*******************************************************************************
 * CUDA Memory Qualifiers to RVV and RISC-V mappings
 ******************************************************************************/

/* Global memory (device memory) */
#define __device__
#define __host__
#define __global__

/* Shared memory (per block shared scratchpad) */
/* Emulated as static thread-local or explicitly allocated buffer */
#define __shared__ static __attribute__((aligned(64)))

/* Constant memory (read-only, cached) */
/* Emulate as const qualified variables in fast memory region */
#define __constant__ const __attribute__((section(".rodata")))

/* Unified memory */
/* CUDA Unified Memory: accessible by CPU and GPU */
/* We emulate UM as normal heap memory with atomic consistency */
/* Vector load/store and gather/scatter allow efficient UM accesses */

/*******************************************************************************
 * Vectorized load from global/unified memory
 ******************************************************************************/
static inline void vector_load_float(const float *src, float *dst, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vse32_v_f32m1(dst, vec, vl);
}

/*******************************************************************************
 * Vectorized store to global/unified memory
 ******************************************************************************/
static inline void vector_store_float(float *dst, const float *src, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vse32_v_f32m1(dst, vec, vl);
}

/*******************************************************************************
 * Vectorized gather from non-contiguous memory addresses (e.g., scatter-gather)
 ******************************************************************************/
static inline void vector_gather_float(const float *base, const size_t *indices, float *dst, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vuint32m1_t vindex = vle32_v_u32m1((const uint32_t *)indices, vl);
    vfloat32m1_t vec = vlseg4e32_v_f32m1(base, vindex, vl); // or use vrgather if supported
    vse32_v_f32m1(dst, vec, vl);
}

/*******************************************************************************
 * Vectorized scatter to non-contiguous memory addresses
 ******************************************************************************/
static inline void vector_scatter_float(float *base, const size_t *indices, const float *src, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vuint32m1_t vindex = vle32_v_u32m1((const uint32_t *)indices, vl);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vsse32_v_f32m1(base, vindex, vec, vl);
}

/*******************************************************************************
 * Atomic operations (using RVA extension) on global/unified memory
 ******************************************************************************/
static inline int atomicAdd_int(volatile int *addr, int val) {
    // Use RISC-V atomic fetch-add intrinsic
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

static inline float atomicAdd_float(volatile float *addr, float val) {
    // No direct atomic float add in RVA, emulate via CAS loop
    float old, assumed;
    do {
        old = __atomic_load_n(addr, __ATOMIC_SEQ_CST);
        float new_val = old + val;
        assumed = __atomic_compare_exchange_n(addr, &old, new_val, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    } while (!assumed);
    return old;
}

/*******************************************************************************
 * Memory fences (barriers) to emulate CUDA memory consistency
 ******************************************************************************/
#define __threadfence() __sync_synchronize()
#define __threadfence_block() __sync_synchronize()
#define __threadfence_system() __sync_synchronize()

/*******************************************************************************
 * Prefetch hints (mapped to RISC-V Zifencei or standard compiler builtins)
 ******************************************************************************/
static inline void prefetch_global(const void* ptr) {
    // May map to RISCV Zifencei or compiler intrinsic (no-op fallback)
    __builtin_prefetch(ptr, 0, 3);
}

#endif // CUDA2RVV_MEMORY_H
