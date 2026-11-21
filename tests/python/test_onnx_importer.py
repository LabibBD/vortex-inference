"""CPU-only integration tests for the ONNX -> VORTEX model-import contract."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


ROOT = Path(__file__).resolve().parents[2]
CHECK = ROOT / "scripts" / "check_onnx_compat.py"
CONVERT = ROOT / "scripts" / "convert_onnx_to_vortex.py"


def run_cli(script: Path, *args: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *(str(x) for x in args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def save_model(path: Path, node, inputs, outputs, initializers=()) -> Path:
    graph = helper.make_graph(
        [node],
        "vortex-import-contract",
        inputs,
        outputs,
        initializer=list(initializers),
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 18)],
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path


def gemm_model(path: Path, *, include_bias: bool = True) -> Path:
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])]
    weight = numpy_helper.from_array(np.ones((4, 3), dtype=np.float32), "w")
    bias = numpy_helper.from_array(np.zeros(4, dtype=np.float32), "b")
    names = ["x", "w", "b"] if include_bias else ["x", "w"]
    node = helper.make_node("Gemm", names, ["y"], transB=1)
    return save_model(
        path, node, inputs, outputs, [weight, bias] if include_bias else [weight]
    )


class TestOnnxImporter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vortex-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_valid_gemm_check_and_conversion(self) -> None:
        source = gemm_model(self.root / "good.onnx")
        checked = run_cli(CHECK, source)
        self.assertEqual(checked.returncode, 0, checked.stderr)
        self.assertIn("CONVERTIBLE", checked.stdout)
        self.assertIn("GPU execution not verified", checked.stdout)

        output = self.root / "output"
        converted = run_cli(CONVERT, source, output)
        self.assertEqual(converted.returncode, 0, converted.stdout + converted.stderr)
        self.assertIn("VORTEX_MODEL\t1", (output / "model.vtx").read_text())
        self.assertGreater((output / "weights.bin").stat().st_size, 0)

    def test_checker_rejects_gemm_without_required_bias(self) -> None:
        source = gemm_model(self.root / "missing_bias.onnx", include_bias=False)
        checked = run_cli(CHECK, source)
        self.assertEqual(checked.returncode, 2, checked.stdout + checked.stderr)
        self.assertIn("converter rejected", checked.stderr.lower())
        converted = run_cli(CONVERT, source, self.root / "output")
        self.assertNotEqual(converted.returncode, 0)

    def test_checker_rejects_unsupported_layernorm_epsilon(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
        scale = numpy_helper.from_array(np.ones(3, dtype=np.float32), "scale")
        bias = numpy_helper.from_array(np.zeros(3, dtype=np.float32), "bias")
        node = helper.make_node(
            "LayerNormalization", ["x", "scale", "bias"], ["y"], epsilon=0.001
        )
        source = save_model(
            self.root / "bad_norm.onnx", node, [x], [y], [scale, bias]
        )
        checked = run_cli(CHECK, source)
        self.assertEqual(checked.returncode, 2, checked.stdout + checked.stderr)
        self.assertIn("converter rejected", checked.stderr.lower())

    def test_identity_rejected_until_executor_supports_it(self) -> None:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
        source = save_model(
            self.root / "identity.onnx",
            helper.make_node("Identity", ["x"], ["y"]),
            [x],
            [y],
        )
        checked = run_cli(CHECK, source)
        converted = run_cli(CONVERT, source, self.root / "output")
        self.assertEqual(checked.returncode, 2)
        self.assertNotEqual(converted.returncode, 0)
        self.assertIn("Identity", converted.stdout + converted.stderr)

    def test_conversion_preserves_unrelated_output_files(self) -> None:
        source = gemm_model(self.root / "good.onnx")
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "keep-me.txt"
        sentinel.write_text("DO NOT DELETE", encoding="utf-8")
        converted = run_cli(CONVERT, source, output)
        self.assertEqual(converted.returncode, 0, converted.stdout + converted.stderr)
        self.assertEqual(sentinel.read_text(), "DO NOT DELETE")
        self.assertTrue((output / "model.vtx").is_file())

    def test_conversion_refuses_file_output_without_deletion(self) -> None:
        source = gemm_model(self.root / "good.onnx")
        target = self.root / "important.txt"
        target.write_text("protected", encoding="utf-8")
        converted = run_cli(CONVERT, source, target)
        self.assertNotEqual(converted.returncode, 0)
        self.assertEqual(target.read_text(), "protected")

    def test_existing_phase16_model_is_convertible(self) -> None:
        source = ROOT / "models" / "phase16_mlp.onnx"
        self.assertTrue(source.is_file())
        checked = run_cli(CHECK, source)
        self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
        converted = run_cli(CONVERT, source, self.root / "phase16")
        self.assertEqual(converted.returncode, 0, converted.stdout + converted.stderr)
        self.assertTrue((self.root / "phase16" / "model.vtx").is_file())


if __name__ == "__main__":
    unittest.main()
