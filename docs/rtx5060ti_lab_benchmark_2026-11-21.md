# RTX 5060 Ti benchmark, 2026-11-21

> **Evidence status:** These measurements are transcribed from the owner's
> successful lab-workstation terminal output. The originally generated JSON/CSV were inspected, cross-checked against the
> uploaded archive and committed under [`results/rtx5060ti_2026-11-21/`](../results/rtx5060ti_2026-11-21/).
> CUDA test and HELIOS reference logs are archived alongside them.

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
session. The full [CSV](../results/rtx5060ti_2026-11-21/phase24_runtime_comparison.csv) and
[JSON](../results/rtx5060ti_2026-11-21/phase24_runtime_comparison.json)
are now archived for more granular review.

The recorded numerical tolerance is **0.0001**, and the JSON reports
maximum absolute errors of **1.0e-6** (VORTEX),
**8.344650268554688e-7** (PyTorch eager), and
**9.5367431640625e-7** (PyTorch compiled), all within tolerance.

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

The archived CTest log confirms **30/30 passing tests in 6.47 seconds**.
The [HELIOS verification log](../results/rtx5060ti_2026-11-21/helios_verifier.log)
shows a maximum absolute error of **3e-8** against tolerance **1e-4**.
The CUDA verification used commit `d26749e`; the benchmark used a later
reporting-only commit, `375ec68`.

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

**Archival status:** The benchmark JSON and CSV, complete CTest output,
and HELIOS verification output have been archived as text files at
[`results/rtx5060ti_2026-11-21/`](../results/rtx5060ti_2026-11-21/).
The original workstation output directory is ignored by Git to avoid
accidental bulk commits. The uploaded tarball also contains full build,
configure, and environment logs, which were not separately added to GitHub.

Original local artifact SHA-256 digests for provenance:

- `phase24_runtime_comparison.json`: `47f431344916519af41dfaca7c2ccacbeed76f1ebb82a40128a8583ee0cfa83f`
- `phase24_runtime_comparison.csv`: `92a12b9d4d9966f8385a3f9d819c4fd38315514339df555c5b65280076d1cd6d` (original local file used CRLF line endings)
- `ctest.log`: `4a4c8877a60acf3117e7ea2a00ef04d600d1c6f96652dc2edac82d16b4f54b57`
- `helios_verifier.log`: `cb1faa8449ff3598e8660a5a66b189f4e93f3180d8a03f1e6bbfb74d1eb67761`
