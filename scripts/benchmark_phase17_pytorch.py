from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


BATCH = 128
INPUT = 784

CPU_WARMUP = 20
CPU_ITERATIONS = 200

GPU_WARMUP = 50
GPU_ITERATIONS = 500


class VortexMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(784, 512)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(256, 10)
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.relu1(
            self.fc1(x)
        )

        x = self.relu2(
            self.fc2(x)
        )

        return self.softmax(
            self.fc3(x)
        )


def statistics(
    values: list[float],
) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    return {
        "mean_ms": float(
            array.mean()
        ),
        "median_ms": float(
            np.median(array)
        ),
        "p95_ms": float(
            np.percentile(
                array,
                95,
            )
        ),
        "min_ms": float(
            array.min()
        ),
        "max_ms": float(
            array.max()
        ),
    }


def benchmark_cpu(
    model: nn.Module,
    x: torch.Tensor,
) -> dict[str, float]:
    model.eval()

    with torch.inference_mode():
        for _ in range(
            CPU_WARMUP
        ):
            model(x)

        samples = []

        for _ in range(
            CPU_ITERATIONS
        ):
            start = time.perf_counter()

            model(x)

            end = time.perf_counter()

            samples.append(
                (end - start)
                * 1000.0
            )

    return statistics(
        samples
    )


def benchmark_cuda(
    model: nn.Module,
    x: torch.Tensor,
) -> tuple[
    dict[str, float],
    float,
]:
    model.eval()

    torch.cuda.reset_peak_memory_stats()

    with torch.inference_mode():
        for _ in range(
            GPU_WARMUP
        ):
            model(x)

        torch.cuda.synchronize()

        samples = []

        for _ in range(
            GPU_ITERATIONS
        ):
            start = time.perf_counter()

            model(x)

            torch.cuda.synchronize()

            end = time.perf_counter()

            samples.append(
                (end - start)
                * 1000.0
            )

    peak_mib = (
        torch.cuda.max_memory_allocated()
        / (1024.0 * 1024.0)
    )

    return (
        statistics(samples),
        float(peak_mib),
    )


def main() -> None:
    root = Path("models")

    state = torch.load(
        root / "phase17_mlp.pt",
        map_location="cpu",
    )

    input_array = np.fromfile(
        root / "phase17_input.bin",
        dtype=np.float32,
    ).reshape(
        BATCH,
        INPUT,
    )

    cpu_model = VortexMLP()
    cpu_model.load_state_dict(
        state
    )
    cpu_model.eval()

    cpu_input = torch.from_numpy(
        input_array.copy()
    )

    print(
        "[VORTEX] Benchmarking PyTorch CPU..."
    )

    cpu_stats = benchmark_cpu(
        cpu_model,
        cpu_input,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available in PyTorch"
        )

    gpu_model = VortexMLP().cuda()

    gpu_model.load_state_dict(
        state
    )

    gpu_input = (
        torch.from_numpy(
            input_array.copy()
        )
        .cuda()
    )

    print(
        "[VORTEX] Benchmarking PyTorch CUDA..."
    )

    gpu_stats, gpu_peak_mib = (
        benchmark_cuda(
            gpu_model,
            gpu_input,
        )
    )

    with torch.inference_mode():
        gpu_output = (
            gpu_model(gpu_input)
            .cpu()
            .numpy()
        )

    predictions = (
        gpu_output.argmax(
            axis=1
        )
    )

    results = {
        "batch": BATCH,
        "architecture":
            "784-512-256-10",
        "cpu": cpu_stats,
        "cuda": gpu_stats,
        "cuda_peak_memory_mib":
            gpu_peak_mib,
        "first_10_predictions":
            predictions[:10].tolist(),
    }

    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    Path(
        "results/phase17_pytorch.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print(
        "\n================================="
    )
    print(
        " PyTorch Phase-17 Results"
    )
    print(
        "=================================\n"
    )

    print(
        "CPU mean:  "
        f"{cpu_stats['mean_ms']:.6f} ms"
    )

    print(
        "CUDA mean: "
        f"{gpu_stats['mean_ms']:.6f} ms"
    )

    print(
        "CUDA P95:  "
        f"{gpu_stats['p95_ms']:.6f} ms"
    )

    print(
        "CUDA peak: "
        f"{gpu_peak_mib:.3f} MiB"
    )

    print(
        "\nSaved results/phase17_pytorch.json"
    )


if __name__ == "__main__":
    main()
