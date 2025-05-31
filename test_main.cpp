#include "cuda2rvv_runtime.cpp" // Include your runtime
#include <stdio.h>
#include <stdlib.h>

extern void simple_add_kernel(void* args); // declare for runtime use

// Argument structure for kernel
struct KernelArgs {
    int* data;
    int N;
};

int main() {
    const int N = 16;
    int* dev_data = nullptr;

    // Allocate unified memory
    if (cudaMalloc((void**)&dev_data, N * sizeof(int)) != 0) {
        fprintf(stderr, "cudaMalloc failed\n");
        return 1;
    }

    // Initialize on host
    for (int i = 0; i < N; ++i) {
        dev_data[i] = i;
    }

    // Copy to device (same pointer in this shim)
    cudaMemcpy(dev_data, dev_data, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel: 1 block of 16 threads
    KernelArgs args = { dev_data, N };
    cudaLaunchKernel((cuda_kernel_t)simple_add_kernel, &args, {1, 1, 1}, {16, 1, 1});

    // Copy back to host (no-op in shim)
    cudaMemcpy(dev_data, dev_data, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print result
    for (int i = 0; i < N; ++i) {
        printf("data[%d] = %d\n", i, dev_data[i]);
    }

    cudaFree(dev_data);
    return 0;
}
