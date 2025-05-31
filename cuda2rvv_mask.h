#ifndef CUDA2RVV_MASK_H
#define CUDA2RVV_MASK_H

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

/*******************************************************************************
 * RVV mask utilities and CUDA mask emulation
 ******************************************************************************/

/**
 * Set vector length for mask operations.
 * Typically the warp size or active vector length in RVV.
 * This should be set dynamically or passed from runtime.
 */
extern size_t vl;

/**
 * Create a mask where lanes with values less than `val` are active.
 * @param vx Vector of floats
 * @param val Scalar float to compare against
 * @return vbool32_t mask with bits set for lanes where vx < val
 */
static inline vbool32_t mask_lt_f32(vfloat32m1_t vx, float val) {
    return vmflt_vf_f32m1_b32(vx, val, vl);
}

/**
 * Create a mask where lanes with values greater than `val` are active.
 * @param vx Vector of floats
 * @param val Scalar float to compare against
 * @return vbool32_t mask with bits set for lanes where vx > val
 */
static inline vbool32_t mask_gt_f32(vfloat32m1_t vx, float val) {
    return vmfgt_vf_f32m1_b32(vx, val, vl);
}

/**
 * Create a mask where lanes with values equal to `val` are active.
 * @param vx Vector of floats
 * @param val Scalar float to compare against
 * @return vbool32_t mask with bits set for lanes where vx == val
 */
static inline vbool32_t mask_eq_f32(vfloat32m1_t vx, float val) {
    return vmeq_vf_f32m1_b32(vx, val, vl);
}

/**
 * Merge two vectors using a mask:
 * For lanes where mask is true, select then_vec;
 * otherwise select else_vec.
 */
static inline vfloat32m1_t mask_merge(vbool32_t mask, vfloat32m1_t then_vec, vfloat32m1_t else_vec) {
    return vmerge_vvm_f32m1(mask, then_vec, else_vec, vl);
}

/**
 * Check if any lane in mask is active
 */
static inline int mask_any(vbool32_t mask) {
    return vmfirst_m_b32(mask, vl) != -1;
}

/**
 * Check if all lanes in mask are active
 */
static inline int mask_all(vbool32_t mask) {
    // vmredand_mm returns a mask of all bits ANDed; check if result is set
    return vmredand_mm_b32(mask, vl);
}

/**
 * Invert mask bits
 */
static inline vbool32_t mask_not(vbool32_t mask) {
    return vmnot_m_b32(mask, vl);
}

/**
 * Logical AND of two masks
 */
static inline vbool32_t mask_and(vbool32_t m1, vbool32_t m2) {
    return vmand_mm_b32(m1, m2, vl);
}

/**
 * Logical OR of two masks
 */
static inline vbool32_t mask_or(vbool32_t m1, vbool32_t m2) {
    return vmor_mm_b32(m1, m2, vl);
}

/**
 * Example macro for conditional execution based on mask
 */
#define CUDA_IF(mask, then_expr, else_expr) \
    mask_merge(mask, then_expr, else_expr)

/*******************************************************************************
 * Thread/block index simulation for scalar code, defined externally:
 * extern size_t __tid, __blockDim, __gridDim;
 ******************************************************************************/

/* Convenience macros for thread and block index */
#define threadIdx_x (__tid % __blockDim)
#define blockIdx_x (__tid / __blockDim)
#define blockDim_x (__blockDim)
#define gridDim_x  (__gridDim)

#endif // CUDA2RVV_MASK_H
