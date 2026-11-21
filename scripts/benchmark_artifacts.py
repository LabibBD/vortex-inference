"""CPU-testable output isolation and provenance for GPU benchmark reports."""

from __future__ import annotations

import platform
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def create_output_directory(
    requested: Path | None,
    results_root: Path,
    gpu_name: str,
) -> Path:
    """Never overwrite archived benchmark outputs or a previous result directory."""
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=False)
        return requested.resolve()

    root = results_root / "gpu-benchmarks"
    root.mkdir(parents=True, exist_ok=True)
    safe_gpu = re.sub(r"[^a-z0-9]+", "-", gpu_name.lower()).strip("-")
    label = safe_gpu[:48] or "gpu"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(
        tempfile.mkdtemp(prefix=f"{stamp}-{label}-", dir=root)
    ).resolve()


def _command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=False, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def get_provenance(torch_module: Any, root: Path) -> dict[str, Any]:
    """Record system identity; none of these fields is a performance metric."""
    gpu = torch_module.cuda.get_device_properties(0)
    driver = _command_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    revision = _command_output(["git", "-C", str(root), "rev-parse", "HEAD"])
    return {
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_name": torch_module.cuda.get_device_name(0),
        "gpu_compute_capability": list(
            torch_module.cuda.get_device_capability(0)
        ),
        "gpu_vram_gib": round(gpu.total_memory / (1024**3), 3),
        "nvidia_driver": driver.splitlines()[0] if driver else None,
        "pytorch_version": str(torch_module.__version__),
        "pytorch_cuda_version": torch_module.version.cuda,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": revision,
        "synchronization": "synchronized end-to-end",
        "source": "new local GPU measurement, not archived Phase 24 values",
    }
