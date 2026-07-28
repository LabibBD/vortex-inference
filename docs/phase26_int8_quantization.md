# VORTEX Phase 26 — Dynamic INT8 Quantization

## Implementation

VORTEX implements an experimental dynamic symmetric INT8 backend for
rank-3 linear layers.

The execution path is:

1. Quantize each FP32 activation row dynamically to signed INT8.
2. Use persistent per-output-channel INT8 weights.
3. Accumulate products into signed 32-bit integers.
4. Dequantize using activation-row and weight-channel scales.
5. Add the original FP32 bias.
6. Produce an FP32 output tensor.

Weights are quantized once. Activations are quantized for every inference.

## Correctness

The implementation was validated against a CPU simulation of the same
quantization algorithm.

It was also compared with the corresponding FP32 linear result.

- Maximum absolute error against FP32: 0.09085990
- Maximum mean absolute error: 0.01624180
- Repeated execution: deterministic
- NaN/Inf validation: passed

## Performance Results

| Shape | FP32 mean | INT8 mean | INT8 speedup | Max error | Weight compression |
|---|---:|---:|---:|---:|---:|
| 64 × 32 × 32 | 0.003392 ms | 0.005682 ms | 0.597× | 0.040347 | 3.556× |
| 64 × 32 × 64 | 0.003412 ms | 0.005778 ms | 0.591× | 0.041193 | 3.556× |
| 64 × 64 × 32 | 0.004103 ms | 0.006571 ms | 0.625× | 0.048493 | 3.765× |
| 256 × 128 × 128 | 0.011860 ms | 0.019463 ms | 0.609× | 0.090860 | 3.879× |

## Aggregate Results

- Tested shapes: 4
- Shapes where INT8 was faster: 0
- Shapes where INT8 was slower or equal: 4
- Arithmetic mean speedup: 0.605×
- Geometric mean speedup: 0.605×
- Average weight compression: 3.689×

## Backend Decision

**Status: `memory_oriented_experimental`**

INT8 reduced model-weight storage, but its dynamic quantization and scalar integer GEMM overhead did not consistently outperform FP32.

The INT8 path remains explicitly selectable and experimental. It does not
replace the optimized FP32 transformer backend by default.

## Why INT8 May Not Be Faster

This implementation prioritizes clarity and correctness over peak hardware
utilization.

Current limitations include:

- A separate activation-quantization kernel
- Scalar signed-integer multiplication
- No packed `dp4a` instructions
- No INT8 Tensor Core execution
- No fused quantization and matrix multiplication
- Small matrices where launch overhead is significant
- FP32 dequantization and bias output

A production INT8 backend would normally use packed layouts, architecture-
specific instructions, calibrated quantization and highly optimized vendor
libraries.

## Storage

INT8 weights use approximately one byte per parameter rather than four bytes
for FP32. Per-output-channel FP32 scales slightly reduce the theoretical 4×
compression ratio.

Activation workspace is additional temporary memory and is reported separately
in the benchmark results.

## Conclusion

Phase 26 demonstrates a complete and numerically validated quantized operator,
while preserving an honest distinction between memory reduction and actual
latency improvement.

Measured results—not datatype alone—determine whether a quantized backend is
worth enabling for a particular model and GPU.
