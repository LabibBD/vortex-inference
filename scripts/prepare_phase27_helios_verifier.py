#!/usr/bin/env python3

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


INPUT_FEATURES = 256

LABELS = [
    "SUCCESS",
    "FAILURE",
    "UNCERTAIN",
]


class HeliosOutcomeVerifier(
    nn.Module
):
    """
    Compact classification head intended to consume an
    embedding produced by the HELIOS perception/state
    pipeline.

    This deterministic checkpoint validates backend
    integration. It is not presented as a verifier trained
    on real HELIOS demonstrations.
    """

    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(
            INPUT_FEATURES,
            128,
        )

        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(
            128,
            64,
        )

        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(
            64,
            len(LABELS),
        )

        self.softmax = nn.Softmax(
            dim=-1
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        features = self.relu1(
            self.fc1(features)
        )

        features = self.relu2(
            self.fc2(features)
        )

        return self.softmax(
            self.fc3(features)
        )


def create_reference_features() -> torch.Tensor:
    """
    Representative deterministic feature vector used only
    for integration and numerical validation.
    """

    positions = torch.linspace(
        -1.0,
        1.0,
        steps=INPUT_FEATURES,
        dtype=torch.float32,
    )

    features = (
        0.55 * positions
        + 0.25 * torch.sin(
            positions * 7.0
        )
        + 0.10 * torch.cos(
            positions * 19.0
        )
    )

    return features.reshape(
        1,
        INPUT_FEATURES,
    )


def main() -> None:
    torch.manual_seed(2701)

    model = HeliosOutcomeVerifier()
    model.eval()

    features = (
        create_reference_features()
    )

    with torch.inference_mode():
        probabilities = model(
            features
        )

    prediction = int(
        probabilities.argmax(
            dim=-1
        ).item()
    )

    confidence = float(
        probabilities[
            0,
            prediction,
        ]
    )

    model_root = Path(
        "models"
    )

    package_root = (
        model_root /
        "phase27_helios"
    )

    model_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    package_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        package_root /
        "verifier_state.pt",
    )

    features.numpy().astype(
        np.float32
    ).tofile(
        package_root /
        "input.bin"
    )

    probabilities.numpy().astype(
        np.float32
    ).tofile(
        package_root /
        "reference.bin"
    )

    (
        package_root /
        "labels.txt"
    ).write_text(
        "\n".join(LABELS)
        + "\n",
        encoding="utf-8",
    )

    metadata = {
        "name":
            "HELIOS Outcome Verifier",
        "input_name":
            "verification_features",
        "input_shape":
            [1, INPUT_FEATURES],
        "output_name":
            "outcome_probabilities",
        "output_shape":
            [1, len(LABELS)],
        "labels":
            LABELS,
        "precision":
            "FP32",
        "purpose": (
            "Interface-level integration model for the "
            "VORTEX backend. Replace its checkpoint with "
            "a verifier trained on real HELIOS data before "
            "claiming task-level accuracy."
        ),
    }

    (
        package_root /
        "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    onnx_path = (
        model_root /
        "phase27_helios_verifier.onnx"
    )

    torch.onnx.export(
        model,
        features,
        onnx_path,
        input_names=[
            "verification_features"
        ],
        output_names=[
            "outcome_probabilities"
        ],
        opset_version=18,
        dynamo=False,
    )

    print(
        "================================="
    )

    print(
        " VORTEX × HELIOS Verifier Model"
    )

    print(
        "=================================\n"
    )

    print(
        "Architecture: "
        "256 -> 128 -> 64 -> 3"
    )

    print(
        f"Input shape:  "
        f"{tuple(features.shape)}"
    )

    print(
        f"Output shape: "
        f"{tuple(probabilities.shape)}"
    )

    print(
        "Probabilities:"
    )

    for label, probability in zip(
        LABELS,
        probabilities[0].tolist(),
        strict=True,
    ):
        print(
            f"  {label:9s}: "
            f"{probability:.8f}"
        )

    print(
        f"\nPredicted outcome: "
        f"{LABELS[prediction]}"
    )

    print(
        f"Confidence:        "
        f"{confidence:.8f}"
    )

    print(
        "\n[VORTEX] HELIOS verifier "
        "interface model exported"
    )


if __name__ == "__main__":
    main()
