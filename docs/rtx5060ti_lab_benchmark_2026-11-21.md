# RTX 5060 Ti benchmark, 2026-11-21

> **Evidence status:** These measurements are transcribed from the owner's
> successful lab-workstation terminal output. The generated JSON/CSV artifacts
> are still on the lab workstation and have **not yet been committed** to this
> repository. Consult them before publishing additional precision or accuracy claims.

## Workstation and execution

| Property | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5060 Ti (8 GB) |
| GPU architecture | Blackwell, compute capability 12.0 (`sm_120`) |
| NVIDIA driver | 595.99.02 |
| CUDA compiler / toolkit | CUDA 12.8.93 |
| PyTorch | 2.10.0+cu128 |
| PyTorch CUDA | 12.8 |
| Benchmark command | `python scripts/benchmark_phase24_runtimes.py` |
| Git revision | `375ec683e267cc4a4f8777e4eea4a48d1d87d426` (from reported `git pull` fast-forward) |
| Scope | Fixed small FP32 transformer block, batch 4, 16 tokens, model dim 32, FF dim 64 |
| Timing | 100 warmup iterations, 7 outer repeats, 1,000 timed iterations per repeat |
| TF32 | Disabled, as required by this benchmark's fixed-precision configuration |

The benchmark was executed in a dedicated Conda environment with PyTorch
CUDA functioning on the NVIDIA GPU. The GPU was idle immediately before the run.

## Results

The following is the **console-reported** summary of the new run:

| Runtime | Median latency | Reported P95 | Reported throughput |
| --- | ---: | ---: | ---: |
| **VORTEX CUDA** | **0.027391 ms** | 0.028667 ms | **2,336,533.9 tokens/s** |
| PyTorch CUDA eager | 0.139943 ms | 0.146378 ms | 457,327.4 tokens/s |
| PyTorch CUDA compiled | 0.080168 ms | 0.082928 ms | 798,323.5 tokens/s |

**Ratios of median latency**: VORTEX is **5.109×** faster than PyTorch
CUDA eager and **2.927×** faster than PyTorch CUDA compiled for this fixed
workload under the script's synchronization rules.

VORTEX per-repeat median latencies (ms): 0.027484, 0.027330, 0.027496,
0.027391, 0.027351, 0.027263, 0.027426. All seven were from the same
session. The full CSV and JSON are needed for more granular claims.

The script checks per-runtime maximum absolute numerical error against
a configured tolerance before writing results. Its successful completion
indicates this check was not triggered; specific per-runtime error values
should be read from the generated JSON, not inferred from the console.

## Verification and limitations

The same RTX 5060 Ti workstation successfully built the C++/CUDA runtime,
passed **30 of 30 CTest targets**, and validated the bundled HELIOS verifier
(maximum absolute error **0.00000003**, tolerance 0.00010000).
This verifies the included tests, not all possible model shapes or kernels.

This benchmark synchronizes between timed PyTorch calls and uses fixed
inputs and a very small transformer block, where framework overhead matters.
The compiled PyTorch run emitted warnings that TF32 Tensor Cores were not
enabled and `max_autotune_gemm` was unavailable for the device. No
additional PyTorch tuning is claimed here. The benchmark is **not**
evidence that VORTEX is faster for arbitrary models or optimized serving.

VORTEX's reported P95 and PyTorch's reported P95 use different
aggregation strategies in the benchmark script. Compare **median latency**
for the headline ratios, rather than treating the reported P95 figures
as measurements from an identical pooled-sample procedure.

The earlier RTX 4060 run used different hardware and software conditions.
Direct generation-to-generation speedup claims are not supported by these
two datasets alone.

## Preserving raw artifacts

The local output directory printed by the benchmark was:

```text
results/gpu-benchmarks/20261121T033811Z-nvidia-geforce-rtx-5060-ti-btosxu0i/
```

It contains `phase24_runtime_comparison.json`,
`phase24_runtime_comparison.csv`, and
`phase24_runtime_comparison.md`.

To archive the evidence, inspect the JSON's actual metadata and numerical
error fields and commit copies to an appropriate `results/` subdirectory,
or attach them to a tagged GitHub release. The generated directory is
ignored by Git to avoid accidental bulk commits.
