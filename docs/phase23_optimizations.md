# VORTEX Phase 23 — Performance Experiments

## Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: 8.9
- CUDA Toolkit: 13.2
- Main workload: FP32 transformer block
- Batch: 4
- Tokens: 16
- Model dimension: 32
- Feed-forward dimension: 64

## Experiment 1 — Synchronization Policy

### Before

Every CUDA kernel was immediately followed by a device-wide
`cudaDeviceSynchronize()` call.

- Kernel launches in profiled workload: 976
- Device synchronizations: 976
- Synchronizations per transformer execution: 16
- Median latency: 0.074690 ms

### After

Kernels are submitted asynchronously in default-stream order, followed by one
synchronization at the graph-execution boundary.

- Kernel launches in profiled workload: 976
- Device synchronizations: 61
- Synchronizations per transformer execution: 1
- Median latency: 0.032959 ms

### Result

- Synchronization-call reduction: 93.75%
- Measured speedup: 2.266x
- Median latency reduction: 55.87%

Numerical validation remained unchanged, with maximum absolute error of
approximately 1e-6.

## Experiment 2 — Fused Attention Scaling and Softmax

### Before

Attention used two separate kernels:

1. Multiply scores by `1 / sqrt(head_dim)`
2. Apply row-wise Softmax

### After

A fused CUDA kernel calculates:

`Softmax(scores * scale_factor)`

without materializing an additional scaled-score pass.

### Result

- Separate median latency: 0.035164 ms
- Fused median latency: 0.032734 ms
- Measured speedup: 1.074x
- Median latency reduction: 6.91%

The complete transformer changed from 16 to 15 CUDA kernels per execution.

Across 61 executions:

- Before fusion: 976 kernel launches
- After fusion: 915 kernel launches
- Device synchronizations after fusion: 61
- Fused scaled-Softmax launches: 61
- Separate attention scale launches: 0
- Separate attention Softmax launches: 0

Numerical correctness and finite-output validation remained unchanged.

## Profiler Findings After Optimization

The optimized fused transformer kernel distribution was approximately:

| Kernel | GPU Time Share | Instances |
|---|---:|---:|
| Linear3D | 42.6% | 366 |
| LayerNorm | 19.8% | 122 |
| Batched MatMul | 11.1% | 122 |
| Fused Scaled-Softmax | 9.6% | 61 |
| Elementwise Add | 7.8% | 122 |
| Transpose | 5.1% | 61 |
| GELU | 4.0% | 61 |

Linear projection is now the largest GPU-time component. Attention still
contains an explicit key transpose and custom batched matrix multiplications,
which are candidates for later production optimization.

## Conclusions

The profiler-guided changes produced two independently measured improvements:

1. Boundary synchronization removed unnecessary CPU/GPU barriers.
2. Fused scaled-Softmax reduced kernel count and eliminated an intermediate
   memory pass.

The optimizations were retained only after numerical tests, full transformer
validation and profiler verification confirmed that the computation remained
correct.
