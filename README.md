🌟 Overview
cuda2rvv is a lightweight CUDA runtime shim and LLVM IR lowering toolkit designed to translate CUDA-style kernels to RISC-V Vector (RVV) instructions, enabling next-generation embedded vision, machine learning, and spatial AI systems to run on modern RISC-V hardware.

This project targets RISC-V platforms with RVV (RISC-V Vector Extension) and aims to provide a flexible, testable, and expandable infrastructure for building CUDA-compatible compute stacks on open hardware.

🎯 Project Goals
✅ Translate CUDA kernels to RVV-compatible LLVM IR

✅ Provide a runtime simulation of CUDA APIs on RISC-V

✅ Support atomic ops, memory fences, warp intrinsics (__shfl_sync, __ballot_sync)

✅ Enable texture memory fetch and surface writes

✅ Support __shared__, __global__, __device__ qualifiers

✅ Vectorized memory load/store, gather/scatter

✅ Toolchain integration via custom LLVM IR passes


🚀 Getting Started
1. Clone the repo
bash
Copy
Edit
git clone https://github.com/aaronsafeit/cuda2rvv.git
cd cuda2rvv
2. Build a Test Program
Use one of the provided test examples:

bash
Copy
Edit
g++ test_main.cpp -lpthread -o test_cuda2rvv
./test_cuda2rvv
3. Develop a CUDA-style kernel
Create a .cu file using __global__ functions and common CUDA APIs. Example provided in test_kernel.cu.

🧠 Features
Feature	Status
__global__, __device__, __host__	✅
threadIdx.x, blockIdx.x	✅
__syncthreads()	✅
atomicAdd, atomicMin/Max	✅
Texture fetch/surface write	✅ (host stub)
__shfl_sync, __ballot_sync	✅
Vector load/store	✅
Vector gather/scatter	✅
Warp reductions (sum, min, max)	✅
cudaMalloc, cudaFree	✅
cudaMemcpy, cudaMemcpyKind	✅
cudaMallocManaged (UM)	✅
RVV-aware memory fences	✅

🧪 Testing & Debugging
Drop your kernel in test_kernel.cu

Include it in test_main.cpp

Use unified memory and threading APIs to simulate parallelism

Output intermediate IR with Clang + LLVM for debugging:

bash
Copy
Edit
clang++ -S -emit-llvm -O1 -o kernel.ll test_kernel.cu
🛠 LLVM Pass Integration
The project includes:

IR pass for CUDA intrinsic lowering to RVV intrinsics

Pattern matching using LLVM’s PatternMatch API

Hook-in before llc or codegen phase for maximum compatibility

Coming soon:

Custom vector scheduling

Memory coalescing optimization

RVV-targeted warp-wide fusion


⚠️ Disclaimer
I have no idea what this code does.
I’ve never programmed a day in my life. 
If this works, it's a miracle. If it doesn't, I warned you.
Use at your own risk. Or don't. I won’t understand either way.

Maybe this will inspire one of us...

## 📜 License

MIT – Free to use, modify, and distribute. Open hardware deserves open software.

