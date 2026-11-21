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


    def test_rtx5060ti_evidence_consistency(self) -> None:
        evidence = ROOT / "results/rtx5060ti_2026-11-21"
        data = json.loads(
            (evidence / "phase24_runtime_comparison.json").read_text()
        )
        self.assertEqual(
            data["provenance"]["gpu_name"], "NVIDIA GeForce RTX 5060 Ti"
        )
        self.assertEqual(
            data["provenance"]["gpu_compute_capability"], [12, 0]
        )
        self.assertEqual(data["configuration"]["precision"], "FP32")
        self.assertFalse(data["configuration"]["tf32_enabled"])
        self.assertEqual(data["configuration"]["outer_repeats"], 7)
        self.assertEqual(data["configuration"]["iterations_per_repeat"], 1000)

        runtimes = data["runtimes"]
        vortex = runtimes["vortex_cuda"]
        eager = runtimes["pytorch_cuda_eager"]
        compiled = runtimes["pytorch_cuda_compiled"]
        tolerance = data["numerical_tolerance"]

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
            self.assertLess(runtime["max_error"], tolerance)
            throughput = (
                data["configuration"]["batch"]
                * data["configuration"]["tokens"]
                * 1000
                / runtime["median_ms"]
            )
            self.assertAlmostEqual(
                runtime["tokens_per_second"],
                throughput,
                delta=throughput * 0.001,
            )

        import csv
        with (evidence / "phase24_runtime_comparison.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = {row["runtime"]: row for row in csv.DictReader(handle)}
        self.assertEqual(
            set(rows),
            {"VORTEX CUDA", "PyTorch CUDA eager", "PyTorch CUDA compiled"},
        )
        self.assertAlmostEqual(
            float(rows["VORTEX CUDA"]["median_ms"]), vortex["median_ms"]
        )
        self.assertAlmostEqual(
            float(rows["PyTorch CUDA eager"]["median_ms"]), eager["median_ms"]
        )
        self.assertAlmostEqual(
            float(rows["PyTorch CUDA compiled"]["median_ms"]),
            compiled["median_ms"],
        )

        ctest = (evidence / "ctest_summary.log").read_text()
        self.assertIn("100% tests passed, 0 tests failed out of 30", ctest)
        report = json.loads((evidence / "helios_verifier.log").read_text())
        self.assertTrue(report["validation"]["passed"])
        self.assertLessEqual(
            report["validation"]["max_abs_error"],
            report["validation"]["tolerance"],
        )


if __name__ == "__main__":
    unittest.main()
