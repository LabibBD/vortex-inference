# VORTEX Phase 22 — Profiling Report

## Hardware

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: 8.9
- CUDA Toolkit: 13.2
- Precision: FP32 unless stated otherwise

## Workloads

### Complete Transformer Block

- Batch: 4
- Tokens: 16
- Model dimension: 32
- Feed-forward dimension: 64
- Graph nodes: 12
- Effective CUDA kernels per execution: 16

### Scaled Dot-Product Attention

- Batch: 2
- Tokens: 9
- Head dimension: 13

### GEMM

- Matrix size: 1024 × 1024 × 1024

## End-to-End Transformer Result

- Mean latency: 0.131185 ms
- Median latency: 0.106470 ms
- P95 latency: 0.140984 ms
- Maximum numerical error: approximately 1e-6
- Mean numerical error: approximately zero
- Output finite: yes

## Nsight Systems — GPU Kernels

The transformer block was executed 61 times:

- 1 validation execution
- 10 warmup executions
- 50 measured executions

A total of 976 kernels were launched, corresponding to 16 kernels per
transformer execution.

| Kernel | GPU Time | Instances | Average Duration |
|---|---:|---:|---:|
| Linear3D | 41.0% | 366 | 1.741 us |
| LayerNorm | 19.1% | 122 | 2.437 us |
| Batched MatMul | 10.7% | 122 | 1.361 us |
| Transpose Last Two | 9.3% | 61 | 2.378 us |
| Elementwise Add | 7.5% | 122 | 0.952 us |
| Softmax | 4.9% | 61 | 1.257 us |
| GELU | 3.8% | 61 | 0.974 us |
| Scale | 3.7% | 61 | 0.932 us |

Total GPU kernel time was approximately 1.555 ms across 61 executions,
or approximately 25.5 us of GPU kernel work per transformer block.

The measured mean end-to-end latency was approximately 131.2 us. The
difference indicates substantial kernel-submission, synchronization and
runtime-management cost relative to the amount of GPU computation.

## Nsight Systems — CUDA API

### cudaMalloc

- Calls: 27
- Total API time: approximately 71.10 ms
- Share of captured CUDA API time: 91.2%
- Largest call: approximately 71.06 ms

The largest allocation represents cold-start CUDA initialization and initial
allocation overhead. Memory-pool reuse prevents this cost from occurring on
every steady-state inference.

### cudaDeviceSynchronize

- Calls: 976
- Total API time: approximately 4.65 ms
- Average call duration: approximately 4.76 us
- Calls per transformer execution: 16

Each kernel is followed by a device-wide synchronization. This serializes the
runtime and prevents asynchronous kernel submission.

Synchronization duration includes both waiting overhead and the execution of
the preceding GPU kernel, so it must not be interpreted as pure CPU overhead.

### cudaLaunchKernel

- Calls: 976
- Total API time: approximately 1.97 ms
- Average call duration: approximately 2.02 us
- Calls per transformer execution: 16

Because most kernels execute in approximately 1–2.5 us, host-side launch cost
is comparable to the kernel execution time itself.

## GPU Memory Transfers

- Host-to-device copies: 17
- Host-to-device time: approximately 7.14 us
- Device-to-host copies: 1
- Device-to-host time: approximately 1.34 us

Memory transfers are not a meaningful steady-state bottleneck for this test.

## Nsight Compute — Attention

Observed attention-kernel characteristics included:

- Achieved occupancy: approximately 16.0–16.6%
- Theoretical occupancy: 100%
- Memory throughput: approximately 1.2–4.6%
- Kernel duration: approximately 1.9 us
- Registers per thread: approximately 18–37
- No shared-memory spilling

The kernels are too small to utilize the GPU fully. Nsight Compute warned that
the sampling interval was large relative to kernel duration, so sampled warp
stall percentages are considered low-confidence.

The low achieved occupancy is primarily an effect of the small workload and
limited number of active blocks, rather than a hard register or shared-memory
occupancy limit.

## Previous GEMM Findings

The naive GEMM suffered from inefficient global-memory access despite high
occupancy. Shared-memory tiling reduced execution time and instruction count,
but shifted pressure toward the shared-memory and MIO pipelines.

## Optimization Priorities

1. Remove unconditional device-wide synchronization after every kernel.
2. Preserve synchronization only at explicit execution boundaries.
3. Introduce stream-aware graph execution.
4. Benchmark asynchronous submission against the existing synchronized path.
5. Reduce the number of tiny kernels through fusion.
6. Fuse attention score scaling with Softmax.
7. Replace custom batched GEMM with a cuBLAS strided-batched implementation.
8. Avoid explicitly materializing the transposed key tensor.
9. Measure cold-start and steady-state latency separately.
10. Reprofile after every optimization.

## Conclusion

The complete transformer block is not primarily compute-bound at the current
small workload size. Its kernels execute in approximately 1–2.5 us each, while
the runtime performs 16 kernel launches and 16 device-wide synchronizations per
inference.

The first evidence-based optimization is therefore removal of unconditional
per-kernel cudaDeviceSynchronize calls. Kernel arithmetic optimization should
follow only after asynchronous execution has been implemented and reprofiled.
