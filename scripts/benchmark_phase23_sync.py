#!/usr/bin/env python3

import csv
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


REPEATS = 7

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build"

EXECUTABLE = (
    BUILD /
    "vortex_transformer_block_test"
)

MEAN_PATTERN = re.compile(
    r"Mean latency:\s+"
    r"([0-9]+(?:\.[0-9]+)?)"
    r"\s+ms"
)


def run_once(
    mode: str,
) -> float:
    environment = os.environ.copy()

    environment[
        "VORTEX_SYNC_MODE"
    ] = mode

    result = subprocess.run(
        [str(EXECUTABLE)],
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
            f"Transformer failed in "
            f"{mode} mode"
        )

    match = MEAN_PATTERN.search(
        output
    )

    if match is None:
        print(output)

        raise RuntimeError(
            "Could not parse transformer latency"
        )

    return float(
        match.group(1)
    )


def run_mode(
    mode: str,
) -> list[float]:
    values = []

    print(
        f"\n[VORTEX] Running mode: {mode}"
    )

    for index in range(REPEATS):
        latency = run_once(
            mode
        )

        values.append(
            latency
        )

        print(
            f"  Run {index + 1}: "
            f"{latency:.6f} ms"
        )

    return values


def summarize(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean_ms": statistics.mean(
            values
        ),
        "median_ms": statistics.median(
            values
        ),
        "minimum_ms": min(values),
        "maximum_ms": max(values),
        "stdev_ms": statistics.stdev(
            values
        ),
    }


def main() -> None:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Missing executable: "
            f"{EXECUTABLE}"
        )

    per_kernel_values = run_mode(
        "per-kernel"
    )

    boundary_values = run_mode(
        "boundary"
    )

    per_kernel = summarize(
        per_kernel_values
    )

    boundary = summarize(
        boundary_values
    )

    speedup = (
        per_kernel["median_ms"] /
        boundary["median_ms"]
    )

    latency_reduction = (
        100.0 *
        (
            per_kernel["median_ms"] -
            boundary["median_ms"]
        ) /
        per_kernel["median_ms"]
    )

    results = {
        "repeats": REPEATS,
        "per_kernel": per_kernel,
        "boundary": boundary,
        "speedup": speedup,
        "latency_reduction_percent":
            latency_reduction,
        "raw_per_kernel_ms":
            per_kernel_values,
        "raw_boundary_ms":
            boundary_values,
    }

    results_directory = (
        ROOT /
        "results"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = (
        results_directory /
        "phase23_sync_comparison.json"
    )

    csv_path = (
        results_directory /
        "phase23_sync_comparison.csv"
    )

    json_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
        + "\n"
    )

    with csv_path.open(
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "mode",
                "mean_ms",
                "median_ms",
                "minimum_ms",
                "maximum_ms",
                "stdev_ms",
            ]
        )

        writer.writerow(
            [
                "per_kernel",
                per_kernel["mean_ms"],
                per_kernel["median_ms"],
                per_kernel["minimum_ms"],
                per_kernel["maximum_ms"],
                per_kernel["stdev_ms"],
            ]
        )

        writer.writerow(
            [
                "boundary",
                boundary["mean_ms"],
                boundary["median_ms"],
                boundary["minimum_ms"],
                boundary["maximum_ms"],
                boundary["stdev_ms"],
            ]
        )

    print(
        "\n================================="
    )

    print(
        " VORTEX Phase-23 Synchronization"
    )

    print(
        "=================================\n"
    )

    print(
        "Per-kernel median: "
        f"{per_kernel['median_ms']:.6f} ms"
    )

    print(
        "Boundary median:   "
        f"{boundary['median_ms']:.6f} ms"
    )

    print(
        "Speedup:           "
        f"{speedup:.3f}x"
    )

    print(
        "Latency reduction: "
        f"{latency_reduction:.2f}%"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {json_path.relative_to(ROOT)}"
    )

    print(
        f"  {csv_path.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
