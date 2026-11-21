#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional

from benchmark_artifacts import create_output_directory, get_provenance


BATCH = 4
TOKENS = 16
MODEL_DIM = 32
FF_DIM = 64

WARMUP = 100
ITERATIONS = 1000
OUTER_REPEATS = 7

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

MODEL_ROOT = (
    ROOT /
    "models" /
    "phase21_transformer"
)

VORTEX_EXECUTABLE = (
    BUILD /
    "vortex_transformer_block_test"
)


class TinyTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.q_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.k_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.v_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.out_proj = nn.Linear(
            MODEL_DIM,
            MODEL_DIM,
        )

        self.norm1 = nn.LayerNorm(
            MODEL_DIM,
            eps=1e-5,
        )

        self.ff1 = nn.Linear(
            MODEL_DIM,
            FF_DIM,
        )

        self.ff2 = nn.Linear(
            FF_DIM,
            MODEL_DIM,
        )

        self.norm2 = nn.LayerNorm(
            MODEL_DIM,
            eps=1e-5,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        query = self.q_proj(x)
        key = self.k_proj(x)
        value = self.v_proj(x)

        scores = (
            query @
            key.transpose(-2, -1)
        ) / math.sqrt(MODEL_DIM)

        probabilities = torch.softmax(
            scores,
            dim=-1,
        )

        context = (
            probabilities @ value
        )

        attention_output = self.out_proj(
            context
        )

        normalized = self.norm1(
            x +
            attention_output
        )

        hidden = self.ff1(
            normalized
        )

        hidden = functional.gelu(
            hidden,
            approximate="tanh",
        )

        feed_forward = self.ff2(
            hidden
        )

        return self.norm2(
            normalized +
            feed_forward
        )


def read_fp32(
    filename: str,
    shape: tuple[int, ...],
) -> torch.Tensor:
    path = (
        MODEL_ROOT /
        filename
    )

    array = np.fromfile(
        path,
        dtype=np.float32,
    )

    expected = int(
        np.prod(shape)
    )

    if array.size != expected:
        raise RuntimeError(
            f"{path} contains {array.size} values; "
            f"expected {expected}"
        )

    return torch.from_numpy(
        array.reshape(shape).copy()
    )


def load_linear(
    layer: nn.Linear,
    prefix: str,
    input_features: int,
    output_features: int,
) -> None:
    vortex_weight = read_fp32(
        f"{prefix}_weight.bin",
        (
            input_features,
            output_features,
        ),
    )

    bias = read_fp32(
        f"{prefix}_bias.bin",
        (output_features,),
    )

    with torch.no_grad():
        # VORTEX:  [input, output]
        # PyTorch: [output, input]
        layer.weight.copy_(
            vortex_weight.transpose(
                0,
                1,
            )
        )

        layer.bias.copy_(
            bias
        )


def load_model() -> TinyTransformerBlock:
    model = TinyTransformerBlock()

    load_linear(
        model.q_proj,
        "q",
        MODEL_DIM,
        MODEL_DIM,
    )

    load_linear(
        model.k_proj,
        "k",
        MODEL_DIM,
        MODEL_DIM,
    )

    load_linear(
        model.v_proj,
        "v",
        MODEL_DIM,
        MODEL_DIM,
    )

    load_linear(
        model.out_proj,
        "out",
        MODEL_DIM,
        MODEL_DIM,
    )

    load_linear(
        model.ff1,
        "ff1",
        MODEL_DIM,
        FF_DIM,
    )

    load_linear(
        model.ff2,
        "ff2",
        FF_DIM,
        MODEL_DIM,
    )

    with torch.no_grad():
        model.norm1.weight.copy_(
            read_fp32(
                "norm1_gamma.bin",
                (MODEL_DIM,),
            )
        )

        model.norm1.bias.copy_(
            read_fp32(
                "norm1_beta.bin",
                (MODEL_DIM,),
            )
        )

        model.norm2.weight.copy_(
            read_fp32(
                "norm2_gamma.bin",
                (MODEL_DIM,),
            )
        )

        model.norm2.bias.copy_(
            read_fp32(
                "norm2_beta.bin",
                (MODEL_DIM,),
            )
        )

    model.eval()

    return model


def percentile(
    values: list[float],
    percentage: float,
) -> float:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return float(
        np.percentile(
            array,
            percentage,
        )
    )


def summarize_samples(
    samples: list[float],
) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(
            samples
        ),
        "median_ms": statistics.median(
            samples
        ),
        "p95_ms": percentile(
            samples,
            95.0,
        ),
        "minimum_ms": min(samples),
        "maximum_ms": max(samples),
        "stdev_ms": statistics.stdev(
            samples
        ),
    }


FLOAT_PATTERN = (
    r"([-+]?"
    r"(?:[0-9]*\.)?"
    r"[0-9]+"
    r"(?:[eE][-+]?[0-9]+)?)"
)


def parse_required(
    pattern: str,
    output: str,
    label: str,
) -> float:
    match = re.search(
        pattern,
        output,
    )

    if match is None:
        print(output)

        raise RuntimeError(
            f"Could not parse VORTEX {label}"
        )

    return float(
        match.group(1)
    )


def benchmark_vortex_once() -> dict[str, float]:
    environment = os.environ.copy()

    environment[
        "VORTEX_SYNC_MODE"
    ] = "boundary"

    environment[
        "VORTEX_ATTENTION_SOFTMAX_MODE"
    ] = "fused"

    environment[
        "VORTEX_BENCH_WARMUP"
    ] = str(WARMUP)

    environment[
        "VORTEX_BENCH_ITERATIONS"
    ] = str(ITERATIONS)

    result = subprocess.run(
        [str(VORTEX_EXECUTABLE)],
        cwd=BUILD,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    output = (
        result.stdout +
        "\n" +
        result.stderr
    )

    if result.returncode != 0:
        print(output)

        raise RuntimeError(
            "VORTEX transformer benchmark failed"
        )

    return {
        "mean_ms": parse_required(
            rf"Mean latency:\s+{FLOAT_PATTERN}",
            output,
            "mean latency",
        ),
        "median_ms": parse_required(
            rf"Median latency:\s+{FLOAT_PATTERN}",
            output,
            "median latency",
        ),
        "p95_ms": parse_required(
            rf"P95 latency:\s+{FLOAT_PATTERN}",
            output,
            "P95 latency",
        ),
        "peak_managed_mib":
            parse_required(
                rf"Peak managed VRAM:\s+{FLOAT_PATTERN}",
                output,
                "managed VRAM",
            ),
        "max_error":
            parse_required(
                rf"Maximum abs error:\s+{FLOAT_PATTERN}",
                output,
                "maximum error",
            ),
        "mean_error":
            parse_required(
                rf"Mean abs error:\s+{FLOAT_PATTERN}",
                output,
                "mean error",
            ),
    }


def summarize_vortex_runs(
    runs: list[dict[str, float]],
) -> dict[str, float]:
    return {
        "mean_ms":
            statistics.median(
                [
                    run["mean_ms"]
                    for run in runs
                ]
            ),
        "median_ms":
            statistics.median(
                [
                    run["median_ms"]
                    for run in runs
                ]
            ),
        "p95_ms":
            statistics.median(
                [
                    run["p95_ms"]
                    for run in runs
                ]
            ),
        "minimum_ms":
            min(
                run["median_ms"]
                for run in runs
            ),
        "maximum_ms":
            max(
                run["median_ms"]
                for run in runs
            ),
        "stdev_ms":
            statistics.stdev(
                [
                    run["median_ms"]
                    for run in runs
                ]
            ),
        "peak_managed_mib":
            max(
                run["peak_managed_mib"]
                for run in runs
            ),
        "max_error":
            max(
                run["max_error"]
                for run in runs
            ),
        "mean_error":
            max(
                run["mean_error"]
                for run in runs
            ),
    }


def benchmark_pytorch(
    name: str,
    callable_model: Callable[
        [torch.Tensor],
        torch.Tensor,
    ],
    input_tensor: torch.Tensor,
    reference: torch.Tensor,
) -> dict[str, float]:
    print(
        f"\n[VORTEX] Benchmarking {name}"
    )

    torch.cuda.synchronize()

    with torch.inference_mode():
        for _ in range(WARMUP):
            callable_model(
                input_tensor
            )

    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()

    all_samples: list[float] = []

    output = None

    with torch.inference_mode():
        for repeat in range(
            OUTER_REPEATS
        ):
            repeat_samples = []

            for _ in range(
                ITERATIONS
            ):
                start = (
                    time.perf_counter_ns()
                )

                output = callable_model(
                    input_tensor
                )

                torch.cuda.synchronize()

                stop = (
                    time.perf_counter_ns()
                )

                latency_ms = (
                    stop - start
                ) / 1_000_000.0

                repeat_samples.append(
                    latency_ms
                )

                all_samples.append(
                    latency_ms
                )

            print(
                f"  Repeat {repeat + 1}: "
                f"median="
                f"{statistics.median(repeat_samples):.6f} ms"
            )

    if output is None:
        raise RuntimeError(
            "PyTorch produced no output"
        )

    output_cpu = (
        output.detach()
        .float()
        .cpu()
    )

    reference_cpu = (
        reference.detach()
        .float()
        .cpu()
    )

    difference = torch.abs(
        output_cpu -
        reference_cpu
    )

    statistics_result = (
        summarize_samples(
            all_samples
        )
    )

    statistics_result[
        "peak_allocated_mib"
    ] = (
        torch.cuda.max_memory_allocated()
        / (1024.0 * 1024.0)
    )

    statistics_result[
        "peak_reserved_mib"
    ] = (
        torch.cuda.max_memory_reserved()
        / (1024.0 * 1024.0)
    )

    statistics_result[
        "max_error"
    ] = float(
        difference.max()
    )

    statistics_result[
        "mean_error"
    ] = float(
        difference.mean()
    )

    return statistics_result


def add_throughput(
    result: dict[str, float],
) -> None:
    median_seconds = (
        result["median_ms"] /
        1000.0
    )

    result["sequences_per_second"] = (
        BATCH /
        median_seconds
    )

    result["tokens_per_second"] = (
        BATCH *
        TOKENS /
        median_seconds
    )


def format_optional(
    value: object,
    digits: int = 6,
) -> str:
    if isinstance(
        value,
        (float, int),
    ):
        return f"{float(value):.{digits}f}"

    return "N/A"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark VORTEX against PyTorch without overwriting archives."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="New, non-existing output directory (otherwise auto-create unique run).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "PyTorch CUDA is unavailable"
        )

    if not VORTEX_EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Missing VORTEX executable: "
            f"{VORTEX_EXECUTABLE}"
        )

    provenance = get_provenance(torch, ROOT)
    results_directory = create_output_directory(
        args.output_dir, ROOT / "results", provenance["gpu_name"]
    )
    print(f"[VORTEX] GPU: {provenance['gpu_name']}")
    print(f"[VORTEX] Results directory: {results_directory}")

    torch.manual_seed(2401)

    torch.set_float32_matmul_precision(
        "highest"
    )

    torch.backends.cuda.matmul.allow_tf32 = (
        False
    )

    device = torch.device(
        "cuda"
    )

    model = load_model().to(
        device
    )

    input_tensor = read_fp32(
        "input.bin",
        (
            BATCH,
            TOKENS,
            MODEL_DIM,
        ),
    ).to(device)

    reference = read_fp32(
        "reference.bin",
        (
            BATCH,
            TOKENS,
            MODEL_DIM,
        ),
    )

    # ==================================================
    # VORTEX
    # ==================================================

    vortex_runs = []

    print(
        "\n[VORTEX] Benchmarking optimized VORTEX"
    )

    for repeat in range(
        OUTER_REPEATS
    ):
        run = benchmark_vortex_once()

        vortex_runs.append(
            run
        )

        print(
            f"  Repeat {repeat + 1}: "
            f"median="
            f"{run['median_ms']:.6f} ms"
        )

    vortex = summarize_vortex_runs(
        vortex_runs
    )

    add_throughput(
        vortex
    )

    # ==================================================
    # PyTorch eager
    # ==================================================

    eager = benchmark_pytorch(
        "PyTorch CUDA eager",
        model,
        input_tensor,
        reference,
    )

    add_throughput(
        eager
    )

    # ==================================================
    # PyTorch compiled
    # ==================================================

    compiled: dict[str, object]

    try:
        print(
            "\n[VORTEX] Compiling PyTorch model..."
        )

        compile_start = time.perf_counter()

        compiled_model = torch.compile(
            model,
            mode="reduce-overhead",
            fullgraph=True,
        )

        with torch.inference_mode():
            compiled_model(
                input_tensor
            )

        torch.cuda.synchronize()

        compile_seconds = (
            time.perf_counter() -
            compile_start
        )

        compiled_result = benchmark_pytorch(
            "PyTorch CUDA compiled",
            compiled_model,
            input_tensor,
            reference,
        )

        add_throughput(
            compiled_result
        )

        compiled_result[
            "compile_seconds"
        ] = compile_seconds

        compiled_result[
            "status"
        ] = "available"

        compiled = compiled_result

    except Exception as error:
        compiled = {
            "status": "skipped",
            "reason": (
                f"{type(error).__name__}: "
                f"{error}"
            ),
        }

        print(
            "\n[VORTEX] torch.compile baseline "
            "was skipped:"
        )

        print(
            f"  {compiled['reason']}"
        )

    # ==================================================
    # Comparisons
    # ==================================================

    eager_median = eager[
        "median_ms"
    ]

    vortex[
        "speedup_vs_pytorch_eager"
    ] = (
        eager_median /
        vortex["median_ms"]
    )

    eager[
        "speedup_vs_pytorch_eager"
    ] = 1.0

    if (
        compiled.get("status") ==
        "available"
    ):
        compiled[
            "speedup_vs_pytorch_eager"
        ] = (
            eager_median /
            float(
                compiled["median_ms"]
            )
        )

        compiled[
            "vortex_speedup_vs_compiled"
        ] = (
            float(
                compiled["median_ms"]
            ) /
            vortex["median_ms"]
        )

    # Do not publish a timing comparison for numerically invalid output.
    tolerance = 1e-4
    measured = {
        "vortex_cuda": vortex,
        "pytorch_cuda_eager": eager,
    }
    if compiled.get("status") == "available":
        measured["pytorch_cuda_compiled"] = compiled
    for name, run in measured.items():
        if float(run["max_error"]) > tolerance:
            raise RuntimeError(
                f"{name} exceeds numerical tolerance {tolerance:g}: "
                f"{run['max_error']}"
            )

    results = {
        "provenance": provenance,
        "numerical_tolerance": tolerance,
        "configuration": {
            "batch": BATCH,
            "tokens": TOKENS,
            "model_dimension":
                MODEL_DIM,
            "feed_forward_dimension":
                FF_DIM,
            "precision": "FP32",
            "warmup": WARMUP,
            "iterations_per_repeat":
                ITERATIONS,
            "outer_repeats":
                OUTER_REPEATS,
            "tf32_enabled": False,
        },
        "runtimes": {
            "vortex_cuda": vortex,
            "pytorch_cuda_eager": eager,
            "pytorch_cuda_compiled":
                compiled,
        },
    }

    json_path = (
        results_directory /
        "phase24_runtime_comparison.json"
    )

    csv_path = (
        results_directory /
        "phase24_runtime_comparison.csv"
    )

    markdown_path = (
        results_directory /
        "phase24_runtime_comparison.md"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    runtime_rows = [
        (
            "VORTEX CUDA",
            vortex,
        ),
        (
            "PyTorch CUDA eager",
            eager,
        ),
    ]

    if (
        compiled.get("status") ==
        "available"
    ):
        runtime_rows.append(
            (
                "PyTorch CUDA compiled",
                compiled,
            )
        )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "runtime",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "sequences_per_second",
                "tokens_per_second",
                "max_error",
                "mean_error",
                "peak_managed_mib",
                "peak_allocated_mib",
                "peak_reserved_mib",
                "speedup_vs_pytorch_eager",
            ]
        )

        for name, result in runtime_rows:
            writer.writerow(
                [
                    name,
                    result.get(
                        "mean_ms",
                        "",
                    ),
                    result.get(
                        "median_ms",
                        "",
                    ),
                    result.get(
                        "p95_ms",
                        "",
                    ),
                    result.get(
                        "sequences_per_second",
                        "",
                    ),
                    result.get(
                        "tokens_per_second",
                        "",
                    ),
                    result.get(
                        "max_error",
                        "",
                    ),
                    result.get(
                        "mean_error",
                        "",
                    ),
                    result.get(
                        "peak_managed_mib",
                        "",
                    ),
                    result.get(
                        "peak_allocated_mib",
                        "",
                    ),
                    result.get(
                        "peak_reserved_mib",
                        "",
                    ),
                    result.get(
                        "speedup_vs_pytorch_eager",
                        "",
                    ),
                ]
            )

    table_lines = [
        "| Runtime | Median latency | P95 | Tokens/s | Max error | Relative to eager |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for name, result in runtime_rows:
        table_lines.append(
            "| "
            + name
            + " | "
            + format_optional(
                result.get(
                    "median_ms"
                )
            )
            + " ms | "
            + format_optional(
                result.get(
                    "p95_ms"
                )
            )
            + " ms | "
            + format_optional(
                result.get(
                    "tokens_per_second"
                ),
                1,
            )
            + " | "
            + format_optional(
                result.get(
                    "max_error"
                ),
                8,
            )
            + " | "
            + format_optional(
                result.get(
                    "speedup_vs_pytorch_eager"
                ),
                3,
            )
            + "x |"
        )

    compiled_note = ""

    if (
        compiled.get("status") ==
        "available"
    ):
        compiled_note = (
            "\nPyTorch compilation required "
            f"{float(compiled['compile_seconds']):.3f} seconds. "
            "That one-time compilation cost was excluded from "
            "steady-state latency.\n"
        )
    else:
        compiled_note = (
            "\nThe optional PyTorch compiled baseline was skipped: "
            f"`{compiled.get('reason', 'unknown reason')}`.\n"
        )

    markdown = f"""# VORTEX Phase 24 — Runtime Comparison

## Configuration

- GPU: {provenance['gpu_name']}
- GPU compute capability: {provenance['gpu_compute_capability']}
- NVIDIA driver: {provenance['nvidia_driver']}
- PyTorch version: {provenance['pytorch_version']}
- PyTorch CUDA version: {provenance['pytorch_cuda_version']}
- VORTEX Git commit: {provenance['git_commit']}
- UTC date/time: {provenance['utc_timestamp']}
- Precision: FP32
- TF32: disabled
- Batch: {BATCH}
- Tokens: {TOKENS}
- Model dimension: {MODEL_DIM}
- Feed-forward dimension: {FF_DIM}
- Warmup iterations: {WARMUP}
- Timed iterations per repeat: {ITERATIONS}
- Outer repeats: {OUTER_REPEATS}

All runtimes used the same input tensors and the same trained/generated
parameters from Phase 21. Reported latency is synchronized end-to-end
latency and excludes model construction, file loading and optional compiler
startup.

## Results

{chr(10).join(table_lines)}
{compiled_note}
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
"""

    markdown_path.write_text(
        markdown
    )

    print(
        "\n================================="
    )

    print(
        " VORTEX Phase-24 Comparison"
    )

    print(
        "=================================\n"
    )

    for name, result in runtime_rows:
        print(
            f"{name:24s} "
            f"median="
            f"{float(result['median_ms']):.6f} ms  "
            f"p95="
            f"{float(result['p95_ms']):.6f} ms  "
            f"tokens/s="
            f"{float(result['tokens_per_second']):.1f}"
        )

    print(
        "\nVORTEX speedup vs eager: "
        f"{vortex['speedup_vs_pytorch_eager']:.3f}x"
    )

    if (
        compiled.get("status") ==
        "available"
    ):
        print(
            "Compiled speedup vs eager: "
            f"{float(compiled['speedup_vs_pytorch_eager']):.3f}x"
        )

        print(
            "VORTEX speedup vs compiled: "
            f"{float(compiled['vortex_speedup_vs_compiled']):.3f}x"
        )

    print(
        "\nSaved:"
    )

    print(
        f"  {json_path}"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {markdown_path}"
    )


if __name__ == "__main__":
    main()
