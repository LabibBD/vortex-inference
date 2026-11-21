# VORTEX

VORTEX is a compact neural-network inference runtime built in C++17 and CUDA. It implements GPU tensors, reusable device-memory allocation, static graph execution, lifetime-based buffer reuse, custom neural operators, transformer inference, ONNX conversion, profiling, numerical validation, quantization experiments, and application integration.

## Highlights

- C++17 and CUDA runtime
- CPU/CUDA tensor abstraction and reusable GPU memory pool
- Static computation graph with lifetime analysis and buffer reuse
- FP32 and FP16 neural operators
- Experimental dynamic INT8 Linear3D backend
- Linear, ReLU, GELU, Softmax, LayerNorm, batched matrix multiplication
- Scaled dot-product self-attention and a complete small transformer block
- Fused Linear + GELU and fused attention scaled-Softmax
- CUDA stream and CUDA Graph experiments
- ONNX compatibility checker and converter
- Native `model.vtx` + `weights.bin` format
- C++ JSON inference CLI and Python bridge
- HELIOS outcome-verifier integration
- Nsight Systems / Nsight Compute profiling
- 30 automated tests

## Architecture

```text
ONNX model
    -> compatibility checker
    -> VORTEX converter
    -> model.vtx + weights.bin
    -> ModelLoader
    -> GraphExecutor
    -> CUDA operators
    -> GPU output
```

## Measured Transformer Performance

Validated configuration:

- NVIDIA RTX 4060 Laptop GPU
- FP32
- Batch size 4
- Sequence length 16
- Model dimension 32
- Feed-forward dimension 64
- Boundary synchronization
- Fused attention scaled-Softmax

| Runtime | Median latency | P95 latency | Throughput |
|---|---:|---:|---:|
| VORTEX CUDA | **0.034802 ms** | **0.035364 ms** | **1,838,974.8 tokens/s** |
| PyTorch CUDA eager | 0.106659 ms | 0.123902 ms | 600,043.1 tokens/s |
| PyTorch CUDA compiled | 0.084890 ms | 0.091173 ms | 753,916.8 tokens/s |

Measured relative performance:

- **3.065× faster** than PyTorch CUDA eager
- **2.439× faster** than compiled PyTorch
- Fused scaled-Softmax reduced the transformer from 16 to 15 CUDA kernels and reduced measured latency by **6.91%**

These measurements apply only to the fixed small-transformer workload above. They are not a claim that VORTEX universally outperforms PyTorch.

## Automated importer checks (CPU-only)

The [ONNX importer CI workflow](.github/workflows/python-importer-ci.yml)
runs on GitHub Actions with Python 3.11 and 3.12, **without a GPU**.
It verifies:

- A valid ONNX `Gemm` model passes compatibility checking and conversion.
- Unsupported operator variants and missing Gemm bias inputs are rejected.
- Incompatible LayerNorm epsilon is rejected instead of reporting a false positive.
- Conversion preserves unrelated files in an existing output directory.
- Bundled Phase-16 ONNX conversion still produces native model and weight files.
- Arithmetic and workload metadata in the **archived** transformer benchmark are consistent.

Run the same tests locally:

```bash
python -m pip install -r requirements-importer-ci.txt
python -m unittest discover -s tests/python -p "test_*.py" -v
python scripts/check_onnx_compat.py models/phase16_mlp.onnx
```

The compatibility checker now probes the actual converter in a temporary
directory. Passing the check means **conversion works**, not that the model
has been numerically validated on an NVIDIA GPU.

**Important:** the CPU importer workflow does not compile CUDA, run the
30 C++/CUDA tests, or reproduce measured GPU performance. Those require
running CMake/CTest and benchmarks on a supported NVIDIA CUDA machine.
The latency numbers below are archived measurements, not CI results.

## GPU release verification (on an NVIDIA machine)

When you have access to a CUDA-capable machine, run:

```bash
bash scripts/verify_gpu_release.sh
```

The script captures the GPU/driver/toolchain information, builds the full C++17/CUDA
runtime, runs the CTest suite, and checks the bundled HELIOS verifier against
its reference output. It writes logs to
`results/local-gpu-verification/<UTC timestamp>/` without overwriting
the archived benchmark results.

This release check does **not** measure GPU performance. The separate Phase 24
benchmark and its saved numbers should only be compared with a fresh run made
under the same workload, GPU model, timing, and synchronization settings.

**Benchmark note:** For the saved Phase 24 results, VORTEX's
`minimum_ms` and `maximum_ms` were derived from the range of per-repeat
median latencies, while its `mean_ms` and `p95_ms` use different
aggregations. These fields should not be interpreted as moments of one common
sample distribution. The reported headline speedups are ratios of median
latencies for this fixed workload; do not generalize them to other models.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 2
```

## Test

```bash
ctest --test-dir build --output-on-failure
```

Expected final result:

```text
100% tests passed, 0 tests failed out of 30
```

## Run the HELIOS Verifier

```bash
VORTEX_SYNC_MODE=boundary \
./build/vortex_infer \
    --model models/phase27_helios_verifier.vortex \
    --input models/phase27_helios/input.bin \
    --labels models/phase27_helios/labels.txt \
    --reference models/phase27_helios/reference.bin \
    --tolerance 0.0001 \
    --json
```

Python integration:

```bash
source .venv/bin/activate
python3 scripts/demo_phase27_helios.py
```

The included verifier validates the runtime interface. It is not presented as a model trained on real robot demonstrations.

## Repository Layout

```text
benchmarks/   Performance benchmarks
docs/         Architecture and evaluation reports
graph/        Graph analysis and optimization
include/      Public C++ headers
kernels/      CUDA kernels
models/       Native and reference models
python/       Python integration bridge
runtime/      Runtime execution
scripts/      Conversion, benchmarking and demo utilities
src/          Core implementation
tests/        Automated tests
tools/        Inference command-line tools
results/      Measurements and validation output
```

## Limitations

VORTEX is a portfolio inference runtime, not a replacement for TensorRT, PyTorch, or ONNX Runtime. Current limitations include static-shape-focused execution, limited ONNX operator coverage, no production autotuning, no distributed inference, no packed INT8 Tensor Core path, limited multi-head attention, and no verifier trained on real HELIOS demonstrations.
