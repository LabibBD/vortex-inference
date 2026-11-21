"""Validate arithmetic and provenance of archived benchmarks, not GPU performance."""

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


class TestSavedBenchmarks(unittest.TestCase):
    def test_archived_transformer_speedups_are_consistent(self) -> None:
        data = json.loads(
            (ROOT / "results/phase24_runtime_comparison.json").read_text()
        )
        self.assertEqual(data["configuration"]["batch"], 4)
        self.assertEqual(data["configuration"]["tokens"], 16)
        self.assertEqual(data["configuration"]["precision"], "FP32")
        runtimes = data["runtimes"]
        vortex = runtimes["vortex_cuda"]
        eager = runtimes["pytorch_cuda_eager"]
        compiled = runtimes["pytorch_cuda_compiled"]

        self.assertAlmostEqual(
            vortex["speedup_vs_pytorch_eager"],
            eager["median_ms"] / vortex["median_ms"],
            places=3,
        )
        self.assertAlmostEqual(
            compiled["vortex_speedup_vs_compiled"],
            compiled["median_ms"] / vortex["median_ms"],
            places=3,
        )
        for runtime in (vortex, eager, compiled):
            self.assertGreater(runtime["median_ms"], 0)
            self.assertGreaterEqual(runtime["p95_ms"], runtime["median_ms"])
            expected_throughput = (
                data["configuration"]["batch"]
                * data["configuration"]["tokens"]
                * 1000
                / runtime["median_ms"]
            )
            self.assertAlmostEqual(
                runtime["tokens_per_second"],
                expected_throughput,
                delta=expected_throughput * 0.001,
            )


if __name__ == "__main__":
    unittest.main()
