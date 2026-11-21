#!/usr/bin/env python3

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import onnx
from onnx import TensorProto, helper, shape_inference


SUPPORTED_OPS = {
    "Gemm",
    "MatMul",
    "Add",
    "Relu",
    "Gelu",
    "Softmax",
    "LayerNormalization",
}

SUPPORTED_DTYPES = {
    TensorProto.FLOAT: "FP32",
    TensorProto.FLOAT16: "FP16",
}


class CompatibilityError(RuntimeError):
    pass


def static_shape(value_info) -> list[int]:
    tensor_type = value_info.type.tensor_type

    if not tensor_type.HasField("shape"):
        raise CompatibilityError(
            f"{value_info.name}: missing tensor shape"
        )

    shape: list[int] = []

    for dimension in tensor_type.shape.dim:
        if (
            dimension.HasField("dim_value")
            and dimension.dim_value > 0
        ):
            shape.append(
                int(dimension.dim_value)
            )
        else:
            raise CompatibilityError(
                f"{value_info.name}: "
                "dynamic dimensions are not supported yet"
            )

    return shape


def check_dtype(value_info) -> str:
    element_type = (
        value_info.type.tensor_type.elem_type
    )

    if element_type not in SUPPORTED_DTYPES:
        raise CompatibilityError(
            f"{value_info.name}: unsupported dtype "
            f"{element_type}"
        )

    return SUPPORTED_DTYPES[element_type]


def attributes(node) -> dict:
    return {
        attribute.name:
            helper.get_attribute_value(
                attribute
            )
        for attribute in node.attribute
    }


def validate_node(node, index: int) -> None:
    if node.op_type not in SUPPORTED_OPS:
        raise CompatibilityError(
            f"node {index} "
            f"('{node.name or node.op_type}') uses "
            f"unsupported operator: {node.op_type}"
        )

    attrs = attributes(node)

    if node.op_type == "Gemm":
        alpha = float(
            attrs.get("alpha", 1.0)
        )

        beta = float(
            attrs.get("beta", 1.0)
        )

        trans_a = int(
            attrs.get("transA", 0)
        )

        trans_b = int(
            attrs.get("transB", 0)
        )

        if alpha != 1.0 or beta != 1.0:
            raise CompatibilityError(
                "Gemm currently requires "
                "alpha=1 and beta=1"
            )

        if trans_a != 0:
            raise CompatibilityError(
                "Gemm transA=1 is not supported"
            )

        if trans_b not in (0, 1):
            raise CompatibilityError(
                "Invalid Gemm transB value"
            )

    elif node.op_type == "Softmax":
        axis = int(
            attrs.get("axis", -1)
        )

        if axis not in (-1, 1):
            raise CompatibilityError(
                "VORTEX Softmax currently supports "
                "the final dimension of a 2D tensor"
            )

    elif node.op_type == "LayerNormalization":
        axis = int(
            attrs.get("axis", -1)
        )

        if axis != -1:
            raise CompatibilityError(
                "VORTEX LayerNormalization "
                "currently supports axis=-1 only"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether an ONNX model is "
            "compatible with the VORTEX importer."
        )
    )

    parser.add_argument(
        "model",
        type=Path,
    )

    args = parser.parse_args()

    try:
        model = onnx.load(
            args.model
        )

        onnx.checker.check_model(
            model
        )

        model = shape_inference.infer_shapes(
            model
        )

        initializer_names = {
            initializer.name
            for initializer
            in model.graph.initializer
        }

        print(
            "================================="
        )
        print(
            " VORTEX ONNX Compatibility Check"
        )
        print(
            "=================================\n"
        )

        print(
            f"Model: {args.model}"
        )

        print(
            f"IR version: {model.ir_version}"
        )

        opsets = [
            (
                opset.domain or "ai.onnx",
                opset.version,
            )
            for opset in model.opset_import
        ]

        print(
            f"Opsets: {opsets}\n"
        )

        print("Inputs:")

        for value in model.graph.input:
            if (
                value.name
                in initializer_names
            ):
                continue

            shape = static_shape(
                value
            )

            dtype = check_dtype(
                value
            )

            print(
                f"  {value.name}: "
                f"{shape} {dtype}"
            )

        print("\nOperators:")

        for index, node in enumerate(
            model.graph.node
        ):
            validate_node(
                node,
                index,
            )

            print(
                f"  {index:02d}: "
                f"{node.op_type} "
                f"({node.name or 'unnamed'})"
            )

        print("\nOutputs:")

        for value in model.graph.output:
            shape = static_shape(
                value
            )

            dtype = check_dtype(
                value
            )

            print(
                f"  {value.name}: "
                f"{shape} {dtype}"
            )

        print(
            "\nInitializers:",
            len(
                model.graph.initializer
            ),
        )

        print(
            "Graph nodes:",
            len(
                model.graph.node
            ),
        )

        # A supported operator name alone is not enough: the
        # converter imposes additional constraints on shape,
        # optional inputs, constant weights, and attributes.
        # Probe actual conversion in a disposable directory.
        with tempfile.TemporaryDirectory(prefix="vortex-onnx-") as temp_dir:
            converted = Path(temp_dir) / "model"
            command = [
                sys.executable,
                str(Path(__file__).with_name("convert_onnx_to_vortex.py")),
                str(args.model),
                str(converted),
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                detail = (result.stderr or result.stdout).strip()
                raise CompatibilityError(
                    "The converter rejected this model: "
                    + (detail or f"exit status {result.returncode}")
                )
            if not (converted / "model.vtx").is_file() or not (
                converted / "weights.bin"
            ).is_file():
                raise CompatibilityError(
                    "Converter did not produce the required model artifacts"
                )

        print(
            "\n[VORTEX] ONNX model is CONVERTIBLE "
            "(GPU execution not verified)"
        )

        return 0

    except (
        CompatibilityError,
        onnx.checker.ValidationError,
        OSError,
    ) as error:
        print(
            "\n[VORTEX] ONNX model is NOT compatible",
            file=sys.stderr,
        )

        print(
            f"[VORTEX][ERROR] {error}",
            file=sys.stderr,
        )

        return 2


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
