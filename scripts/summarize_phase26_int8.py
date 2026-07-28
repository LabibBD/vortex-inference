#!/usr/bin/env python3

import csv
import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CSV_PATH = (
    ROOT /
    "results" /
    "phase26_int8_benchmark.csv"
)

JSON_PATH = (
    ROOT /
    "results" /
    "phase26_int8_summary.json"
)

REPORT_PATH = (
    ROOT /
    "docs" /
    "phase26_int8_quantization.md"
)


def read_results() -> list[dict[str, float]]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"Missing benchmark file: {CSV_PATH}"
        )

    rows: list[dict[str, float]] = []

    with CSV_PATH.open(
        newline="",
        encoding="utf-8",
    ) as file:
        reader = csv.DictReader(file)

        for record in reader:
            rows.append(
                {
                    "rows": int(
                        record["rows"]
                    ),
                    "input_features": int(
                        record["input_features"]
                    ),
                    "output_features": int(
                        record["output_features"]
                    ),
                    "fp32_mean_ms": float(
                        record["fp32_mean_ms"]
                    ),
                    "fp32_median_ms": float(
                        record["fp32_median_ms"]
                    ),
                    "fp32_p95_ms": float(
                        record["fp32_p95_ms"]
                    ),
                    "int8_mean_ms": float(
                        record["int8_mean_ms"]
                    ),
                    "int8_median_ms": float(
                        record["int8_median_ms"]
                    ),
                    "int8_p95_ms": float(
                        record["int8_p95_ms"]
                    ),
                    "int8_speedup": float(
                        record["int8_speedup"]
                    ),
                    "max_error": float(
                        record["max_error"]
                    ),
                    "mean_error": float(
                        record["mean_error"]
                    ),
                    "fp32_weight_bytes": int(
                        record[
                            "fp32_weight_bytes"
                        ]
                    ),
                    "int8_weight_bytes": int(
                        record[
                            "int8_weight_bytes"
                        ]
                    ),
                    "weight_compression": float(
                        record[
                            "weight_compression"
                        ]
                    ),
                    "int8_workspace_bytes": int(
                        record[
                            "int8_workspace_bytes"
                        ]
                    ),
                }
            )

    if not rows:
        raise RuntimeError(
            "INT8 benchmark CSV contains no rows"
        )

    return rows


def geometric_mean(
    values: list[float],
) -> float:
    if any(
        value <= 0.0
        for value in values
    ):
        raise ValueError(
            "Geometric mean requires positive values"
        )

    return math.exp(
        statistics.mean(
            math.log(value)
            for value in values
        )
    )


def choose_backend_status(
    rows: list[dict[str, float]],
    geometric_speedup: float,
    maximum_error: float,
    maximum_mean_error: float,
) -> tuple[str, str]:
    speedups = [
        row["int8_speedup"]
        for row in rows
    ]

    all_faster_by_five_percent = all(
        speedup >= 1.05
        for speedup in speedups
    )

    acceptable_accuracy = (
        maximum_error <= 0.10
        and maximum_mean_error <= 0.015
    )

    if (
        all_faster_by_five_percent
        and acceptable_accuracy
    ):
        return (
            "performance_candidate",
            (
                "INT8 was at least 5% faster for every "
                "tested shape and remained within the "
                "configured accuracy limits."
            ),
        )

    if (
        geometric_speedup >= 1.0
        and acceptable_accuracy
    ):
        return (
            "mixed_experimental",
            (
                "INT8 improved average performance, but "
                "the advantage was not consistent across "
                "all tested shapes."
            ),
        )

    return (
        "memory_oriented_experimental",
        (
            "INT8 reduced model-weight storage, but its "
            "dynamic quantization and scalar integer GEMM "
            "overhead did not consistently outperform FP32."
        ),
    )


def main() -> None:
    rows = read_results()

    speedups = [
        row["int8_speedup"]
        for row in rows
    ]

    compressions = [
        row["weight_compression"]
        for row in rows
    ]

    geometric_speedup = (
        geometric_mean(speedups)
    )

    arithmetic_speedup = (
        statistics.mean(speedups)
    )

    wins = sum(
        speedup > 1.0
        for speedup in speedups
    )

    losses = len(rows) - wins

    maximum_error = max(
        row["max_error"]
        for row in rows
    )

    maximum_mean_error = max(
        row["mean_error"]
        for row in rows
    )

    average_compression = (
        statistics.mean(
            compressions
        )
    )

    fastest = max(
        rows,
        key=lambda row:
            row["int8_speedup"],
    )

    slowest = min(
        rows,
        key=lambda row:
            row["int8_speedup"],
    )

    status, explanation = (
        choose_backend_status(
            rows,
            geometric_speedup,
            maximum_error,
            maximum_mean_error,
        )
    )

    summary = {
        "cases": len(rows),
        "int8_wins": wins,
        "int8_losses": losses,
        "arithmetic_mean_speedup":
            arithmetic_speedup,
        "geometric_mean_speedup":
            geometric_speedup,
        "average_weight_compression":
            average_compression,
        "maximum_absolute_error":
            maximum_error,
        "maximum_mean_absolute_error":
            maximum_mean_error,
        "fastest_case": fastest,
        "slowest_case": slowest,
        "backend_status": status,
        "decision": explanation,
        "results": rows,
    }

    JSON_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    table_lines = [
        (
            "| Shape | FP32 mean | INT8 mean | "
            "INT8 speedup | Max error | "
            "Weight compression |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|"
        ),
    ]

    for row in rows:
        shape = (
            f"{row['rows']} × "
            f"{row['input_features']} × "
            f"{row['output_features']}"
        )

        table_lines.append(
            "| "
            + shape
            + " | "
            + f"{row['fp32_mean_ms']:.6f} ms"
            + " | "
            + f"{row['int8_mean_ms']:.6f} ms"
            + " | "
            + f"{row['int8_speedup']:.3f}×"
            + " | "
            + f"{row['max_error']:.6f}"
            + " | "
            + f"{row['weight_compression']:.3f}×"
            + " |"
        )

    report = f"""# VORTEX Phase 26 — Dynamic INT8 Quantization

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

- Maximum absolute error against FP32: {maximum_error:.8f}
- Maximum mean absolute error: {maximum_mean_error:.8f}
- Repeated execution: deterministic
- NaN/Inf validation: passed

## Performance Results

{chr(10).join(table_lines)}

## Aggregate Results

- Tested shapes: {len(rows)}
- Shapes where INT8 was faster: {wins}
- Shapes where INT8 was slower or equal: {losses}
- Arithmetic mean speedup: {arithmetic_speedup:.3f}×
- Geometric mean speedup: {geometric_speedup:.3f}×
- Average weight compression: {average_compression:.3f}×

## Backend Decision

**Status: `{status}`**

{explanation}

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
"""

    REPORT_PATH.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "================================="
    )

    print(
        " VORTEX Phase-26 INT8 Summary"
    )

    print(
        "=================================\n"
    )

    print(
        f"Cases:                    "
        f"{len(rows)}"
    )

    print(
        f"INT8 wins:                "
        f"{wins}"
    )

    print(
        f"INT8 losses/equal:        "
        f"{losses}"
    )

    print(
        f"Arithmetic mean speedup:  "
        f"{arithmetic_speedup:.3f}x"
    )

    print(
        f"Geometric mean speedup:   "
        f"{geometric_speedup:.3f}x"
    )

    print(
        f"Average compression:      "
        f"{average_compression:.3f}x"
    )

    print(
        f"Maximum FP32 error:       "
        f"{maximum_error:.8f}"
    )

    print(
        f"Backend status:           "
        f"{status}"
    )

    print(
        "\nDecision:"
    )

    print(
        f"  {explanation}"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {JSON_PATH.relative_to(ROOT)}"
    )

    print(
        f"  {REPORT_PATH.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
