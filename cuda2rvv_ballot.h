#ifndef CUDA2RVV_SHFL_BALLOT_H
#define CUDA2RVV_SHFL_BALLOT_H

#include <riscv_vector.h>
#include <stdint.h>

/**
 * Set vector length for warp (assumed 32 lanes)
 * Adjust this if your target warp size or hardware vector length differs
 */
static inline size_t set_vl_warp() {
    return vsetvl_e32m1(32);
}

/**
 * Vectorized __shfl_sync
 * Emulates CUDA warp shuffle: returns value from lane `srcLane` across warp
 *
 * @param mask  - active mask of lanes (not used in this simplified version)
 * @param val   - value from current lane
 * @param srcLane - lane ID whose value to fetch (0 to warpSize-1)
 * @return int - value from srcLane
 */
static inline int __shfl_sync(uint32_t mask, int val, int srcLane) {
    size_t vl = set_vl_warp();

    // Broadcast val across vector lanes
    vint32m1_t vals = vmv_v_x_i32m1(val, vl);

    // Gather element from vals at srcLane index for all lanes
    vint32m1_t res = vrgather_vx_i32m1(vals, srcLane, vl);

    // Extract value from lane 0 as scalar result
    int result;
    vse32_v_i32m1(&result, res, vl);

    return result;
}

/**
 * Slide vector elements up by 1 lane, insert new_val at lane 0
 *
 * @param new_val - value to insert at lane 0
 * @param vec     - input vector
 * @return vint32m1 - resulting vector with elements shifted up
 */
static inline vint32m1_t slide1up(int new_val, vint32m1_t vec) {
    size_t vl = set_vl_warp();
    return vslide1up_vx_i32m1(vec, new_val, vl);
}

/**
 * Helper: Extract mask bit at index `idx` from a vbool32_t mask
 *
 * Note: RVV C intrinsics do not provide direct bit extraction from masks.
 * This is a placeholder to be replaced by platform-specific code or inline asm.
 *
 * @param mask - vector mask
 * @param idx  - lane index
 * @return int - 1 if bit set, 0 otherwise
 */
static inline int vget_m_b32(vbool32_t mask, size_t idx) {
    // WARNING: Pseudo-implementation; replace with platform-specific method.
    // Assuming mask is stored as uint32_t for demo:
    uint32_t raw_mask = *((uint32_t*)&mask);
    return (raw_mask & (1U << idx)) != 0;
}

/**
 * Vectorized __ballot_sync
 * Emulates CUDA ballot: returns a bitmask where bits correspond to lanes
 * with predicate true.
 *
 * @param mask      - active mask of lanes (not used in this simplified version)
 * @param predicate - boolean predicate for current lane (0 or 1)
 * @return uint32_t - bitmask of lanes where predicate is true
 */
static inline uint32_t __ballot_sync(uint32_t mask, int predicate) {
    size_t vl = set_vl_warp();

    // Broadcast predicate across all lanes (true if predicate != 0)
    vbool32_t pred_mask = predicate ? (vbool32_t){~0U} : (vbool32_t){0};

    // Accumulate bitmask from vector mask lanes
    uint32_t bitmask = 0;
    for (size_t i = 0; i < vl; ++i) {
        if (vget_m_b32(pred_mask, i))
            bitmask |= (1U << i);
    }
    return bitmask;
}

#endif // CUDA2RVV_SHFL_BALLOT_H
