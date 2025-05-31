#ifndef CUDA2RVV_BALLOT_H
#define CUDA2RVV_BALLOT_H

#include <riscv_vector.h>
#include <stdint.h>

/**
 * Set vector length for warp (assumed 32 lanes)
 * Adjust if your hardware vector length or warp size differs
 */
static inline size_t set_vl_warp() {
    // Set vector length for 32 int32 elements
    return vsetvl_e32m1(32);
}

/**
 * Emulate CUDA __shfl_sync intrinsic
 * Returns value from srcLane within a warp (vector lane)
 *
 * @param mask    - active lanes mask (unused here, but kept for compatibility)
 * @param val     - input value of current lane
 * @param srcLane - lane index to fetch from (0 <= srcLane < warpSize)
 * @return int    - value from srcLane
 */
static inline int __shfl_sync(uint32_t mask, int val, int srcLane) {
    size_t vl = set_vl_warp();

    // Broadcast current lane value across vector lanes
    vint32m1_t vals = vmv_v_x_i32m1(val, vl);

    // Gather value from srcLane index for all lanes
    vint32m1_t gathered = vrgather_vx_i32m1(vals, srcLane, vl);

    // Extract lane 0 value as scalar result
    return vmv_x_s_i32m1(gathered);
}

/**
 * Slide vector elements up by 1 lane, inserting new_val at lane 0
 * Equivalent to shifting elements towards higher lane indices by one
 *
 * @param new_val - value inserted at lane 0
 * @param vec     - input vector
 * @return vint32m1 - shifted vector
 */
static inline vint32m1_t slide1up(int new_val, vint32m1_t vec) {
    size_t vl = set_vl_warp();
    return vslide1up_vx_i32m1(vec, new_val, vl);
}

/**
 * Emulate CUDA __ballot_sync intrinsic
 * Returns a bitmask of lanes where predicate is true
 *
 * @param mask      - active lanes mask (unused here, compatibility)
 * @param predicate - predicate (0 or non-zero)
 * @return uint32_t - bitmask with bits set for lanes with predicate true
 */
static inline uint32_t __ballot_sync(uint32_t mask, int predicate) {
    size_t vl = set_vl_warp();

    // Create mask vector with predicate replicated across lanes
    vbool32_t pred_mask = predicate ? vmsne_vx_i32m1_b32(1, 0, vl) : vmsle_vx_i32m1_b32(0, 0, vl);

    // Convert mask vector to integer bitmask
    // vmv_x_s_i32m1 extracts the mask bits as a uint32_t (one bit per lane)
    uint32_t result_mask = vmv_x_s_i32m1_b32(pred_mask);

    return result_mask;
}

#endif // CUDA2RVV_BALLOT_H
