
#ifndef CUDA2RVV_MEMORY_H
#define CUDA2RVV_MEMORY_H

#include <riscv_vector.h>
#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>

/*******************************************************************************
 * CUDA Memory Qualifiers to RVV and RISC-V mappings
 ******************************************************************************/

/* CUDA qualifiers as no-ops or mapped attributes */
#define __device__
#define __host__
#define __global__

/* Shared memory (per block shared scratchpad) */
#define __shared__ static __attribute__((aligned(64)))

/* Constant memory (read-only, cached) */
#define __constant__ const __attribute__((section(".rodata")))

/* Unified memory emulated as normal heap with atomic consistency */

/*******************************************************************************
 * Vectorized load/store from/to global/unified memory
 ******************************************************************************/
static inline void vector_load_float(const float *src, float *dst, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vse32_v_f32m1(dst, vec, vl);
}

static inline void vector_store_float(float *dst, const float *src, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vse32_v_f32m1(dst, vec, vl);
}

/*******************************************************************************
 * Vectorized gather from scattered indices (non-contiguous)
 ******************************************************************************/
static inline void vector_gather_float(const float *base, const uint32_t *indices, float *dst, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vuint32m1_t vindex = vle32_v_u32m1(indices, vl);
    // Use vrgather to gather float elements by index from base array
    vfloat32m1_t result = vrgather_vx_f32m1(base, vindex, vl);
    vse32_v_f32m1(dst, result, vl);
}

/*******************************************************************************
 * Vectorized scatter to scattered indices (non-contiguous)
 ******************************************************************************/
static inline void vector_scatter_float(float *base, const uint32_t *indices, const float *src, size_t n) {
    size_t vl = vsetvl_e32m1(n);
    vuint32m1_t vindex = vle32_v_u32m1(indices, vl);
    vfloat32m1_t vec = vle32_v_f32m1(src, vl);
    vsse32_v_f32m1(base, vindex, vec, vl);
}

/*******************************************************************************
 * Atomic operations on global/unified memory
 ******************************************************************************/

/* Atomic add for int32 */
static inline int atomicAdd_int(volatile int *addr, int val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

/* Atomic add for float using compare-exchange loop */
static inline float atomicAdd_float(volatile float *addr, float val) {
    float old_val, new_val;
    do {
        old_val = __atomic_load_n(addr, __ATOMIC_SEQ_CST);
        new_val = old_val + val;
    } while (!__atomic_compare_exchange_n(addr, &old_val, new_val, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST));
    return old_val;
}

/* Atomic add for int64_t */
static inline int64_t atomicAdd_int64(volatile int64_t *addr, int64_t val) {
    return __atomic_fetch_add(addr, val, __ATOMIC_SEQ_CST);
}

/* Atomic compare-and-exchange (generic) */
static inline int atomicCAS_int(volatile int *addr, int expected, int desired) {
    __atomic_compare_exchange_n(addr, &expected, desired, 0, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);
    return expected;
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
    // May map to RISCV Zifencei or compiler intrinsic (fallback to builtin prefetch)
    __builtin_prefetch(ptr, 0, 3);
    // Optional: Insert fence if platform requires
    // asm volatile("fence iorw, iorw" ::: "memory");
}

#endif // CUDA2RVV_MEMORY_H
