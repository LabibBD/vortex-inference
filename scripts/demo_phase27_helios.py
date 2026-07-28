#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

from python.vortex_backend import (  # noqa: E402
    VortexVerifierBackend,
)


def recovery_decision(
    prediction: str,
) -> str:
    if prediction == "SUCCESS":
        return (
            "Accept the action result and continue "
            "to the next instruction."
        )

    if prediction == "FAILURE":
        return (
            "Trigger the HELIOS recovery policy and "
            "retry the failed action."
        )

    return (
        "Capture another observation before deciding "
        "whether to retry."
    )


def main() -> None:
    backend = VortexVerifierBackend(
        executable=(
            ROOT /
            "build" /
            "vortex_infer"
        ),
        model_directory=(
            ROOT /
            "models" /
            "phase27_helios_verifier.vortex"
        ),
        labels_path=(
            ROOT /
            "models" /
            "phase27_helios" /
            "labels.txt"
        ),
    )

    input_path = (
        ROOT /
        "models" /
        "phase27_helios" /
        "input.bin"
    )

    result = backend.infer_file(
        input_path
    )

    decision = recovery_decision(
        result["prediction"]
    )

    print(
        "================================="
    )

    print(
        " HELIOS × VORTEX Demo"
    )

    print(
        "=================================\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )

    print(
        "\nHELIOS decision:"
    )

    print(
        f"  {decision}"
    )


if __name__ == "__main__":
    main()
