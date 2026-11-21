"""CPU-only tests for benchmark output safety and metadata generation."""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_artifacts import create_output_directory, get_provenance


class TestBenchmarkArtifacts(unittest.TestCase):
    def test_default_creates_unique_gpu_directories(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = create_output_directory(None, root, "RTX 5060 Ti")
            second = create_output_directory(None, root, "RTX 5060 Ti")
            self.assertNotEqual(first, second)
            self.assertEqual(first.parent, root / "gpu-benchmarks")
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_explicit_directory_cannot_overwrite_existing_results(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "archive"
            target.mkdir()
            sentinel = target / "phase24_runtime_comparison.json"
            sentinel.write_text("untouched", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                create_output_directory(target, root, "RTX 5060 Ti")
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "untouched")

    def test_explicit_directory_is_created_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            target = root / "new" / "experiment"
            self.assertEqual(
                create_output_directory(target, root, "RTX 5060 Ti"),
                target.resolve(),
            )
            self.assertTrue(target.is_dir())

    def test_provenance_extracts_gpu_and_framework(self) -> None:
        fake_cuda = SimpleNamespace(
            get_device_name=lambda index: "NVIDIA RTX 5060 Ti",
            get_device_capability=lambda index: (12, 0),
            get_device_properties=lambda index: SimpleNamespace(
                total_memory=8 * 1024**3
            ),
        )
        fake_torch = SimpleNamespace(
            cuda=fake_cuda,
            version=SimpleNamespace(cuda="12.8"),
            __version__="2.10.0+cu128",
        )
        details = get_provenance(fake_torch, ROOT)
        self.assertEqual(details["gpu_compute_capability"], [12, 0])
        self.assertEqual(details["pytorch_cuda_version"], "12.8")
        self.assertEqual(details["gpu_vram_gib"], 8.0)
        self.assertEqual(details["gpu_name"], "NVIDIA RTX 5060 Ti")


if __name__ == "__main__":
    unittest.main()
