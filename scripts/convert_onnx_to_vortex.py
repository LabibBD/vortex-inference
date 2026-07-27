#!/usr/bin/env python3

import argparse
import shutil
import struct
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


DTYPE_NAMES = {
    TensorProto.FLOAT: "FP32",
    TensorProto.FLOAT16: "FP16",
}

NUMPY_DTYPES = {
    "FP32": np.dtype("<f4"),
    "FP16": np.dtype("<f2"),
}


class ConversionError(RuntimeError):
    pass


def clean_name(name: str) -> str:
    return (
        name.replace("\t", "_")
        .replace("\n", "_")
        .replace("\r", "_")
    )


def node_attributes(node) -> dict:
    return {
        attribute.name:
        helper.get_attribute_value(attribute)
        for attribute in node.attribute
    }


def build_value_info(model):
    info = {}

    values = (
        list(model.graph.input)
        + list(model.graph.value_info)
        + list(model.graph.output)
    )

    for value in values:
        tensor_type = value.type.tensor_type

        if tensor_type.elem_type == 0:
            continue

        if tensor_type.elem_type not in DTYPE_NAMES:
            continue

        shape = []

        for dimension in tensor_type.shape.dim:
            if (
                dimension.HasField("dim_value")
                and dimension.dim_value > 0
            ):
                shape.append(
                    int(dimension.dim_value)
                )
            else:
                raise ConversionError(
                    f"{value.name}: dynamic dimensions "
                    "are not supported"
                )

        info[value.name] = {
            "shape": shape,
            "dtype": DTYPE_NAMES[
                tensor_type.elem_type
            ],
        }

    return info


def tensor_from_array(
    name: str,
    array: np.ndarray,
) -> dict:
    if array.dtype == np.float32:
        dtype = "FP32"
    elif array.dtype == np.float16:
        dtype = "FP16"
    else:
        raise ConversionError(
            f"{name}: unsupported initializer "
            f"dtype {array.dtype}"
        )

    contiguous = np.ascontiguousarray(
        array.astype(
            NUMPY_DTYPES[dtype],
            copy=False,
        )
    )

    return {
        "name": name,
        "shape": list(contiguous.shape),
        "dtype": dtype,
        "role": "Constant",
        "data": contiguous,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a supported ONNX model into "
            "the VORTEX model format."
        )
    )

    parser.add_argument(
        "onnx_model",
        type=Path,
    )

    parser.add_argument(
        "output",
        type=Path,
    )

    args = parser.parse_args()

    model = onnx.load(
        args.onnx_model
    )

    onnx.checker.check_model(
        model
    )

    model = shape_inference.infer_shapes(
        model
    )

    value_info = build_value_info(
        model
    )

    initializer_arrays = {
        initializer.name:
            numpy_helper.to_array(
                initializer
            )
        for initializer
        in model.graph.initializer
    }

    initializer_names = set(
        initializer_arrays
    )

    graph_outputs = {
        value.name
        for value in model.graph.output
    }

    tensors = OrderedDict()
    nodes = []

    def add_runtime_tensor(
        name: str,
        role: str,
    ) -> None:
        if name in tensors:
            if role == "Output":
                tensors[name]["role"] = "Output"
            return

        if name not in value_info:
            raise ConversionError(
                f"Missing inferred shape for tensor: {name}"
            )

        entry = value_info[name]

        tensors[name] = {
            "name": name,
            "shape": entry["shape"],
            "dtype": entry["dtype"],
            "role": role,
            "data": None,
        }

    def add_constant(
        name: str,
        array: np.ndarray,
    ) -> None:
        if name in tensors:
            return

        tensors[name] = tensor_from_array(
            name,
            array,
        )

    # ==================================================
    # Graph inputs
    # ==================================================

    for value in model.graph.input:
        if value.name in initializer_names:
            continue

        add_runtime_tensor(
            value.name,
            "Input",
        )

    # ==================================================
    # Nodes
    # ==================================================

    for index, node in enumerate(
        model.graph.node
    ):
        op = node.op_type

        node_name = clean_name(
            node.name or
            f"{op}_{index}"
        )

        attrs = node_attributes(
            node
        )

        converted_inputs = []
        converted_outputs = list(
            node.output
        )

        # ----------------------------------------------
        # Gemm -> VORTEX Linear
        # ----------------------------------------------

        if op == "Gemm":
            if len(node.input) != 3:
                raise ConversionError(
                    "VORTEX Linear currently requires "
                    "Gemm with A, B and bias inputs"
                )

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

            if (
                alpha != 1.0
                or beta != 1.0
                or trans_a != 0
            ):
                raise ConversionError(
                    f"{node_name}: unsupported Gemm "
                    "attributes"
                )

            input_name = node.input[0]
            weight_name = node.input[1]
            bias_name = node.input[2]

            if weight_name not in initializer_arrays:
                raise ConversionError(
                    f"{node_name}: Gemm weight must "
                    "be a constant initializer"
                )

            if bias_name not in initializer_arrays:
                raise ConversionError(
                    f"{node_name}: Gemm bias must "
                    "be a constant initializer"
                )

            weight = initializer_arrays[
                weight_name
            ]

            bias = initializer_arrays[
                bias_name
            ]

            if trans_b == 1:
                # PyTorch nn.Linear commonly stores:
                #
                # weight = [out_features, in_features]
                #
                # VORTEX Linear expects:
                #
                # weight = [in_features, out_features]
                transformed_name = (
                    weight_name
                    + f"__vortex_transposed_{index}"
                )

                transformed_weight = (
                    np.ascontiguousarray(
                        weight.T
                    )
                )

                add_constant(
                    transformed_name,
                    transformed_weight,
                )

                weight_name = (
                    transformed_name
                )

            elif trans_b == 0:
                add_constant(
                    weight_name,
                    weight,
                )

            else:
                raise ConversionError(
                    f"{node_name}: invalid transB"
                )

            add_constant(
                bias_name,
                bias,
            )

            add_runtime_tensor(
                input_name,
                (
                    "Output"
                    if input_name
                    in graph_outputs
                    else "Intermediate"
                ),
            )

            converted_inputs = [
                input_name,
                weight_name,
                bias_name,
            ]

            converted_op = "Linear"

        # ----------------------------------------------
        # Directly supported operations
        # ----------------------------------------------

        elif op in {
            "MatMul",
            "Add",
            "Relu",
            "Gelu",
            "Softmax",
            "LayerNormalization",
            "Identity",
        }:
            converted_op = {
                "MatMul": "MatMul",
                "Add": "Add",
                "Relu": "ReLU",
                "Gelu": "GELU",
                "Softmax": "Softmax",
                "LayerNormalization":
                    "LayerNorm",
                "Identity": "Identity",
            }[op]

            if op == "Softmax":
                axis = int(
                    attrs.get(
                        "axis",
                        -1,
                    )
                )

                if axis not in (-1, 1):
                    raise ConversionError(
                        f"{node_name}: unsupported "
                        "Softmax axis"
                    )

            if op == "LayerNormalization":
                axis = int(
                    attrs.get(
                        "axis",
                        -1,
                    )
                )

                epsilon = float(
                    attrs.get(
                        "epsilon",
                        1e-5,
                    )
                )

                if axis != -1:
                    raise ConversionError(
                        f"{node_name}: LayerNorm "
                        "requires axis=-1"
                    )

                if abs(
                    epsilon - 1e-5
                ) > 1e-8:
                    raise ConversionError(
                        f"{node_name}: VORTEX "
                        "currently requires "
                        "LayerNorm epsilon=1e-5"
                    )

            converted_inputs = list(
                node.input
            )

            for input_name in converted_inputs:
                if input_name in initializer_arrays:
                    add_constant(
                        input_name,
                        initializer_arrays[
                            input_name
                        ],
                    )
                elif input_name:
                    role = (
                        "Output"
                        if input_name
                        in graph_outputs
                        else "Intermediate"
                    )

                    add_runtime_tensor(
                        input_name,
                        role,
                    )

        else:
            raise ConversionError(
                f"{node_name}: unsupported ONNX "
                f"operator {op}"
            )

        # ----------------------------------------------
        # Register outputs
        # ----------------------------------------------

        for output_name in converted_outputs:
            role = (
                "Output"
                if output_name in graph_outputs
                else "Intermediate"
            )

            add_runtime_tensor(
                output_name,
                role,
            )

        nodes.append(
            {
                "name": node_name,
                "op": converted_op,
                "inputs": converted_inputs,
                "outputs": converted_outputs,
            }
        )

    # ==================================================
    # Assign stable TensorIds
    # ==================================================

    tensor_ids = {
        name: index
        for index, name
        in enumerate(tensors.keys())
    }

    # ==================================================
    # Prepare output directory
    # ==================================================

    output_dir = args.output

    if output_dir.exists():
        if output_dir.is_dir():
            shutil.rmtree(
                output_dir
            )
        else:
            output_dir.unlink()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weights_path = (
        output_dir /
        "weights.bin"
    )

    manifest_path = (
        output_dir /
        "model.vtx"
    )

    # ==================================================
    # Write raw constant data
    #
    # Constants are aligned to 64 bytes to keep the
    # format friendly for future mapped/batched loading.
    # ==================================================

    offsets = {}

    with weights_path.open(
        "wb"
    ) as weights_file:

        for name, tensor in tensors.items():
            if tensor["role"] != "Constant":
                offsets[name] = (
                    -1,
                    0,
                )
                continue

            position = (
                weights_file.tell()
            )

            padding = (
                64 -
                (position % 64)
            ) % 64

            if padding:
                weights_file.write(
                    b"\x00" * padding
                )

            offset = (
                weights_file.tell()
            )

            raw = (
                tensor["data"]
                .tobytes(
                    order="C"
                )
            )

            weights_file.write(
                raw
            )

            offsets[name] = (
                offset,
                len(raw),
            )

    # ==================================================
    # Write manifest
    # ==================================================

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest:

        manifest.write(
            "VORTEX_MODEL\t1\n"
        )

        manifest.write(
            "WEIGHTS\tweights.bin\n"
        )

        manifest.write(
            f"TENSORS\t{len(tensors)}\n"
        )

        for name, tensor in tensors.items():
            tensor_id = tensor_ids[
                name
            ]

            offset, byte_count = (
                offsets[name]
            )

            shape_text = ",".join(
                str(dimension)
                for dimension
                in tensor["shape"]
            )

            manifest.write(
                "\t".join(
                    [
                        "TENSOR",
                        str(tensor_id),
                        clean_name(name),
                        tensor["role"],
                        tensor["dtype"],
                        "CUDA",
                        str(
                            len(
                                tensor["shape"]
                            )
                        ),
                        shape_text,
                        str(offset),
                        str(byte_count),
                    ]
                )
                + "\n"
            )

        manifest.write(
            f"NODES\t{len(nodes)}\n"
        )

        for node_id, node in enumerate(
            nodes
        ):
            input_ids = ",".join(
                str(
                    tensor_ids[name]
                )
                for name in node["inputs"]
                if name
            )

            output_ids = ",".join(
                str(
                    tensor_ids[name]
                )
                for name in node["outputs"]
            )

            manifest.write(
                "\t".join(
                    [
                        "NODE",
                        str(node_id),
                        clean_name(
                            node["name"]
                        ),
                        node["op"],
                        input_ids,
                        output_ids,
                    ]
                )
                + "\n"
            )

        output_ids = [
            tensor_ids[value.name]
            for value
            in model.graph.output
        ]

        manifest.write(
            "OUTPUTS\t"
            + ",".join(
                str(value)
                for value
                in output_ids
            )
            + "\n"
        )

    # ==================================================
    # Summary
    # ==================================================

    constants = sum(
        1
        for tensor
        in tensors.values()
        if tensor["role"] ==
        "Constant"
    )

    print(
        "================================="
    )
    print(
        " VORTEX ONNX Converter"
    )
    print(
        "=================================\n"
    )

    print(
        f"Source:      {args.onnx_model}"
    )

    print(
        f"Destination: {output_dir}"
    )

    print(
        f"Tensors:     {len(tensors)}"
    )

    print(
        f"Constants:   {constants}"
    )

    print(
        f"Nodes:       {len(nodes)}"
    )

    print(
        f"Outputs:     {len(model.graph.output)}"
    )

    print(
        f"Weights:     "
        f"{weights_path.stat().st_size} bytes"
    )

    print(
        "\nConverted nodes:"
    )

    for index, node in enumerate(
        nodes
    ):
        print(
            f"  {index:02d}: "
            f"{node['op']} "
            f"({node['name']})"
        )

    print(
        "\n[VORTEX] Model conversion PASSED"
    )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(
            main()
        )
    except ConversionError as error:
        print(
            f"[VORTEX][ERROR] {error}"
        )

        raise SystemExit(2)
