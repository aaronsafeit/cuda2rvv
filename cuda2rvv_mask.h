// cuda2rvv.h
#ifndef CUDA2RVV_H
#define CUDA2RVV_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// --- Execution qualifiers (ignored on RVV backend) ---
#define __global__
#define __device__
#define __host__

// --- Thread/block index simulation for scalarized code ---
#define threadIdx_x (__tid % __blockDim)
#define blockIdx_x (__tid / __blockDim)
#define blockDim_x (__blockDim)
#define gridDim_x  (__gridDim)

extern size_t __tid;
extern size_t __blockDim;
extern size_t __gridDim;

// --- Conditional execution macros ---
#define CUDA_IF(mask, then_expr, else_expr) \
    vmerge_vvm_f32m1(mask, then_expr, else_expr, vl)

// --- Kernel launch macro (macro emulation only) ---
#define CUDA_KERNEL(kernel, blocks, threads, ...) \
    __tid = 0; \
    __blockDim = threads; \
    __gridDim = blocks; \
    for (__tid = 0; __tid < (blocks * threads); ++__tid) { \
        kernel<<<0,0>>>(__VA_ARGS__); \
    }

// --- Vector mask helpers ---
#define CUDA_MASK_LT(vx, val) vmflt_vf_f32m1_b32(vx, val, vl)
#define CUDA_MASK_GT(vx, val) vmfgt_vf_f32m1_b32(vx, val, vl)

#endif // CUDA2RVV_H
