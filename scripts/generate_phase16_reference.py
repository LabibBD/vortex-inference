from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


def main() -> None:
    model_path = Path(
        "models/phase16_mlp.onnx"
    )

    model = onnx.load(
        model_path
    )

    # Deterministic input.
    values = np.arange(
        16,
        dtype=np.float32,
    )

    input_data = (
        (values - 7.5) * 0.1
    ).reshape(2, 8)

    evaluator = ReferenceEvaluator(
        model
    )

    outputs = evaluator.run(
        None,
        {
            "input": input_data,
        },
    )

    probabilities = np.asarray(
        outputs[0],
        dtype=np.float32,
    )

    input_path = Path(
        "models/phase16_input.bin"
    )

    output_path = Path(
        "models/phase16_reference.bin"
    )

    input_data.tofile(
        input_path
    )

    probabilities.tofile(
        output_path
    )

    print(
        "[VORTEX] Phase-16 reference generated"
    )

    print(
        f"Input shape:  {input_data.shape}"
    )

    print(
        f"Output shape: {probabilities.shape}"
    )

    print(
        "Row sums:",
        probabilities.sum(axis=1),
    )


if __name__ == "__main__":
    main()
