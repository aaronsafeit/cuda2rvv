#include "cuda2rvv.h"
#include <stdio.h>

__global__ void simple_add_kernel(int* data, int N) {
    int idx = blockIdx_x * blockDim_x + threadIdx_x;

    if (idx < N) {
        data[idx] += 1;

        // Basic warp-wide reduction using atomicAdd
        __syncthreads(); // simulate barrier
        atomicAdd(&data[0], 1); // all threads increment data[0]
    }
}
