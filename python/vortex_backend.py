from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


class VortexVerifierBackend:
    """
    Python subprocess bridge between HELIOS and the
    VORTEX C++/CUDA inference executable.
    """

    def __init__(
        self,
        executable: str | Path,
        model_directory: str | Path,
        labels_path: str | Path,
    ) -> None:
        self.executable = Path(
            executable
        ).resolve()

        self.model_directory = Path(
            model_directory
        ).resolve()

        self.labels_path = Path(
            labels_path
        ).resolve()

        if not self.executable.exists():
            raise FileNotFoundError(
                f"Missing VORTEX executable: "
                f"{self.executable}"
            )

        if not self.model_directory.exists():
            raise FileNotFoundError(
                f"Missing VORTEX model: "
                f"{self.model_directory}"
            )

        if not self.labels_path.exists():
            raise FileNotFoundError(
                f"Missing labels: "
                f"{self.labels_path}"
            )

    def infer_file(
        self,
        input_path: str | Path,
    ) -> dict:
        command = [
            str(self.executable),
            "--model",
            str(self.model_directory),
            "--input",
            str(
                Path(input_path).resolve()
            ),
            "--labels",
            str(self.labels_path),
            "--json",
        ]

        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )

        if result.returncode != 0:
            raise RuntimeError(
                "VORTEX inference failed.\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        try:
            return json.loads(
                result.stdout
            )
        except json.JSONDecodeError as error:
            raise RuntimeError(
                "VORTEX returned invalid JSON:\n"
                f"{result.stdout}"
            ) from error

    def predict(
        self,
        features: Iterable[float],
    ) -> dict:
        values = [
            float(value)
            for value in features
        ]

        if len(values) != 256:
            raise ValueError(
                "HELIOS verifier requires exactly "
                "256 input features"
            )

        payload = struct.pack(
            f"<{len(values)}f",
            *values,
        )

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=".bin",
                delete=False,
            ) as temporary:
                temporary.write(
                    payload
                )

                temporary_path = Path(
                    temporary.name
                )

            return self.infer_file(
                temporary_path
            )
        finally:
            if (
                temporary_path is not None
                and temporary_path.exists()
            ):
                temporary_path.unlink()
