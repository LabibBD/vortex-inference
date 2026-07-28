# VORTEX Phase 24 — Runtime Comparison

## Configuration

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Precision: FP32
- TF32: disabled
- Batch: 4
- Tokens: 16
- Model dimension: 32
- Feed-forward dimension: 64
- Warmup iterations: 100
- Timed iterations per repeat: 1000
- Outer repeats: 7

All runtimes used the same input tensors and the same trained/generated
parameters from Phase 21. Reported latency is synchronized end-to-end
latency and excludes model construction, file loading and optional compiler
startup.

## Results

| Runtime | Median latency | P95 | Tokens/s | Max error | Relative to eager |
|---|---:|---:|---:|---:|---:|
| VORTEX CUDA | 0.034802 ms | 0.035364 ms | 1838974.8 | 0.00000100 | 3.065x |
| PyTorch CUDA eager | 0.106659 ms | 0.123902 ms | 600043.1 | 0.00000107 | 1.000x |
| PyTorch CUDA compiled | 0.084890 ms | 0.091173 ms | 753916.8 | 0.00000095 | 1.256x |

PyTorch compilation required 1.309 seconds. That one-time compilation cost was excluded from steady-state latency.

## Memory Reporting

VORTEX reports peak active bytes managed by its own GPU memory pool.

PyTorch reports both peak tensor memory allocated and peak memory reserved by
its caching allocator. These allocator metrics are useful individually but are
not perfectly equivalent to one another or to total process VRAM.

## Correctness

All available runtimes were compared against the saved Phase-21 PyTorch
reference output. Performance results are valid only when numerical validation
remains within the established transformer tolerance.

## Interpretation

The comparison is not intended to claim that one small custom runtime
universally outperforms an established framework. It measures one fixed,
small-batch transformer block on one GPU.

VORTEX minimizes framework overhead and uses a fixed graph, but its custom
matrix kernels are relatively simple. PyTorch benefits from mature CUDA
libraries and, when compilation is available, graph optimization and kernel
fusion.

Cold-start and compiler startup costs are reported separately from steady-state
inference.
