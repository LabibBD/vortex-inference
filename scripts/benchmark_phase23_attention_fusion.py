#!/usr/bin/env python3

import csv
import json
import os
import re
import statistics
import subprocess
from pathlib import Path


REPEATS = 9

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
    attention_mode: str,
) -> float:
    environment = os.environ.copy()

    environment[
        "VORTEX_SYNC_MODE"
    ] = "boundary"

    environment[
        "VORTEX_ATTENTION_SOFTMAX_MODE"
    ] = attention_mode

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
            "Transformer failed in "
            f"{attention_mode} mode"
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


def summarize(
    values: list[float],
) -> dict[str, float]:
    return {
        "mean_ms":
            statistics.mean(values),
        "median_ms":
            statistics.median(values),
        "minimum_ms":
            min(values),
        "maximum_ms":
            max(values),
        "stdev_ms":
            statistics.stdev(values),
    }


def main() -> None:
    if not EXECUTABLE.exists():
        raise FileNotFoundError(
            f"Missing executable: "
            f"{EXECUTABLE}"
        )

    separate_values = []
    fused_values = []

    for repeat in range(
        REPEATS
    ):
        order = (
            ["separate", "fused"]
            if repeat % 2 == 0
            else ["fused", "separate"]
        )

        print(
            f"\n[VORTEX] Pair "
            f"{repeat + 1}/{REPEATS}"
        )

        for mode in order:
            latency = run_once(
                mode
            )

            print(
                f"  {mode:8s}: "
                f"{latency:.6f} ms"
            )

            if mode == "separate":
                separate_values.append(
                    latency
                )
            else:
                fused_values.append(
                    latency
                )

    separate = summarize(
        separate_values
    )

    fused = summarize(
        fused_values
    )

    speedup = (
        separate["median_ms"] /
        fused["median_ms"]
    )

    reduction = (
        100.0 *
        (
            separate["median_ms"] -
            fused["median_ms"]
        ) /
        separate["median_ms"]
    )

    results = {
        "repeats": REPEATS,
        "separate": separate,
        "fused": fused,
        "speedup": speedup,
        "latency_reduction_percent":
            reduction,
        "raw_separate_ms":
            separate_values,
        "raw_fused_ms":
            fused_values,
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
        "phase23_attention_fusion.json"
    )

    csv_path = (
        results_directory /
        "phase23_attention_fusion.csv"
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

        for mode, data in (
            ("separate", separate),
            ("fused", fused),
        ):
            writer.writerow(
                [
                    mode,
                    data["mean_ms"],
                    data["median_ms"],
                    data["minimum_ms"],
                    data["maximum_ms"],
                    data["stdev_ms"],
                ]
            )

    print(
        "\n================================="
    )

    print(
        " VORTEX Attention Fusion"
    )

    print(
        "=================================\n"
    )

    print(
        "Separate median: "
        f"{separate['median_ms']:.6f} ms"
    )

    print(
        "Fused median:    "
        f"{fused['median_ms']:.6f} ms"
    )

    print(
        "Speedup:         "
        f"{speedup:.3f}x"
    )

    print(
        "Reduction:       "
        f"{reduction:.2f}%"
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
