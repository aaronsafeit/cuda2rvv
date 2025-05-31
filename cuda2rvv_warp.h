#ifndef CUDA2RVV_WARP_H
#define CUDA2RVV_WARP_H

#include <riscv_vector.h>
#include <stdint.h>
#include <limits.h>
#include <stdatomic.h>
#include <pthread.h>

/*******************************************************************************
 * Warp Shuffle and Reduction — Emulated via RVV (LLVM Lowerable)
 ******************************************************************************/

/**
 * Scalar fallback for __shfl_sync.
 * Use this if vector shuffle not available (i.e., one thread sim).
 */
static inline int scalar_shfl_sync(uint32_t /*mask*/, int val, int /*srcLane*/) {
    return val; // fallback: return same value
}

/**
 * Vector shuffle using vrgather.
 * This is the RVV analog to CUDA's __shfl_sync.
 */
static inline vint32m1_t rvv_shfl_sync(vint32m1_t vals, int srcLane) {
    size_t vl = vsetvl_e32m1(32);
    return vrgather_vx_i32m1(vals, srcLane, vl);
}

#define __shfl_sync(mask, val, srcLane) scalar_shfl_sync(mask, val, srcLane)


/*******************************************************************************
 * Warp Reduction Helpers (__reduce_sum, __reduce_max, __reduce_min)
 ******************************************************************************/

static inline int __reduce_sum(int val) {
    size_t vl = vsetvl_e32m1(1);
    vint32m1_t v = vmv_v_x_i32m1(val, vl);
    vint32m1_t sum = vfredusum_vs_i32m1_i32m1(v, vmv_v_x_i32m1(0, vl), vl);
    return vget_i32m1_i32(sum, 0);
}

static inline int __reduce_max(int val) {
    size_t vl = vsetvl_e32m1(1);
    vint32m1_t v = vmv_v_x_i32m1(val, vl);
    vint32m1_t maxv = vfredmax_vs_i32m1_i32m1(v, vmv_v_x_i32m1(INT32_MIN, vl), vl);
    return vget_i32m1_i32(maxv, 0);
}

static inline int __reduce_min(int val) {
    size_t vl = vsetvl_e32m1(1);
    vint32m1_t v = vmv_v_x_i32m1(val, vl);
    vint32m1_t minv = vfredmin_vs_i32m1_i32m1(v, vmv_v_x_i32m1(INT32_MAX, vl), vl);
    return vget_i32m1_i32(minv, 0);
}


/*******************************************************************************
 * Warp Ballot — Return predicate mask for warp lanes
 ******************************************************************************/

static inline uint32_t scalar_ballot_sync(uint32_t /*mask*/, int predicate) {
    return predicate ? 0xFFFFFFFFu : 0u;
}

// Placeholder: real RVV-based ballot not expressible in C, but can be handled in IR
static inline uint32_t rvv_ballot_sync(vbool32_t pred_mask) {
    // This would be lowered via LLVM custom IR pattern
    return 0xFFFFFFFFu; // mock full warp active
}

#define __ballot_sync(mask, predicate) scalar_ballot_sync(mask, predicate)


/*******************************************************************************
 * Warp-Level Atomics — Use C11 Atomics or RVA intrinsics
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
 * Warp Sync Emulation
 ******************************************************************************/

extern pthread_barrier_t __cuda_warp_barrier;

#define __syncwarp(mask) pthread_barrier_wait(&__cuda_warp_barrier)

static inline void init_warp_barrier(size_t threads_per_warp) {
    pthread_barrier_init(&__cuda_warp_barrier, NULL, threads_per_warp);
}


/*******************************************************************************
 * Texture Interpolation and Surface Write Support Hooks
 ******************************************************************************/

// These are not implemented here but are forward-compatible entry points
// for generating proper LLVM IR lowering for CUDA-style sampling.

static inline float warp_lerp(float a, float b, float t) {
    return a + t * (b - a);
}

static inline float texture_warp_interp(const float* tex, int x, int y, float fx, float fy, int width, int height) {
    int x0 = x;
    int x1 = x + 1;
    int y0 = y;
    int y1 = y + 1;

    x0 = (x0 < 0) ? 0 : (x0 >= width ? width - 1 : x0);
    x1 = (x1 < 0) ? 0 : (x1 >= width ? width - 1 : x1);
    y0 = (y0 < 0) ? 0 : (y0 >= height ? height - 1 : y0);
    y1 = (y1 < 0) ? 0 : (y1 >= height ? height - 1 : y1);

    float tl = tex[y0 * width + x0];
    float tr = tex[y0 * width + x1];
    float bl = tex[y1 * width + x0];
    float br = tex[y1 * width + x1];

    float top = warp_lerp(tl, tr, fx);
    float bottom = warp_lerp(bl, br, fx);
    return warp_lerp(top, bottom, fy);
}

#endif // CUDA2RVV_WARP_H
