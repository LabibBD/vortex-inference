# VORTEX

**GPU-Accelerated Neural Network Inference Runtime**

VORTEX is a C++/CUDA inference runtime built to explore how neural-network execution works below high-level frameworks such as PyTorch.

The project focuses on:

- custom CUDA kernels
- tensor and GPU memory management
- optimized matrix multiplication
- neural-network operators
- FP16 inference
- memory pooling and reuse
- computational graph execution
- kernel fusion
- CUDA Streams and CUDA Graphs
- ONNX model import
- transformer inference
- GPU profiling and benchmarking

## Hardware

Development system:

- NVIDIA GeForce RTX 4060 Laptop GPU
- CUDA 13.2
- Ubuntu 24.04
- C++17 / CUDA C++
- CMake

## Current Status

### Phase 0 — Environment Verification

- [x] NVIDIA driver
- [x] CUDA Toolkit
- [x] nvcc
- [x] CMake
- [x] Nsight Systems
- [x] Nsight Compute
- [x] Custom CUDA kernel compilation
- [x] CUDA kernel execution
- [x] PyTorch CUDA reference environment

### Phase 1 — CUDA Fundamentals

In progress.

## License

MIT
