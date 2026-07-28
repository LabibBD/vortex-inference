# VORTEX Interview Guide

## Thirty-Second Explanation

VORTEX is a C++ and CUDA neural-network inference runtime with a native graph format, reusable GPU memory management, tensor lifetime analysis, custom neural operators, transformer execution, profiling, numerical validation, and a Python integration bridge.

## Precise Performance Statement

For one fixed FP32 transformer block with batch 4, sequence length 16, and model dimension 32 on an RTX 4060 Laptop GPU, VORTEX measured 0.034802 ms median latency:

- 3.065× faster than PyTorch eager
- 2.439× faster than compiled PyTorch

This result should not be generalized to large models.

## Important Engineering Decisions

- Memory pooling avoids repeated `cudaMalloc` calls.
- Static shapes simplify allocation and validation.
- Lifetime analysis enables intermediate-buffer reuse.
- Boundary synchronization removes per-kernel synchronization overhead.
- Fused scaled-Softmax removes one attention kernel launch.
- INT8 remains experimental because quantization overhead can exceed its gains.

## Future Improvements

- cuBLASLt and Tensor Core paths
- Packed INT8 execution
- Kernel autotuning
- Dynamic shapes
- Multi-head attention
- Stream-aware memory reuse
- Wider ONNX operator support
