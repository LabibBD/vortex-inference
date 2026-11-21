#!/usr/bin/env bash
# Validate VORTEX on an NVIDIA GPU without modifying archived benchmark results.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'HELP'
Usage: bash scripts/verify_gpu_release.sh [OUTPUT_ROOT]

Builds VORTEX with CUDA, runs all CTest targets, and validates the
bundled HELIOS verifier against its saved reference output.

Requires: NVIDIA GPU and driver, CUDA toolkit (nvcc), CMake, C++ compiler.
Default evidence directory: results/local-gpu-verification/<UTC timestamp>
Benchmark performance is NOT measured by this script.
HELP
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

for tool in cmake nvcc nvidia-smi; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "[VORTEX][ERROR] Missing required tool: $tool" >&2
    exit 1
  fi
done

if ! nvidia-smi -L >/dev/null 2>&1; then
  echo "[VORTEX][ERROR] Could not detect an NVIDIA GPU/driver" >&2
  exit 1
fi

OUTPUT_ROOT="${1:-$ROOT/results/local-gpu-verification}"
mkdir -p "$OUTPUT_ROOT"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
REPORT_DIR="$(mktemp -d "$OUTPUT_ROOT/$STAMP-XXXXXXXX")"
BUILD_DIR="$ROOT/build"

echo "[VORTEX] Evidence directory: $REPORT_DIR"

{
  printf 'UTC timestamp: %s\n' "$STAMP"
  printf 'Git revision: '
  git -C "$ROOT" rev-parse HEAD || true
  printf '\nNVIDIA GPU:\n'
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
  printf '\nCUDA compiler:\n'
  nvcc --version
  printf '\nCMake:\n'
  cmake --version | head -n 1
} >"$REPORT_DIR/environment.txt" 2>&1

echo "[VORTEX] Configuring CUDA release build..."
cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release \
  2>&1 | tee "$REPORT_DIR/configure.log"

echo "[VORTEX] Building complete C++/CUDA suite..."
cmake --build "$BUILD_DIR" --parallel "${VORTEX_BUILD_JOBS:-2}" \
  2>&1 | tee "$REPORT_DIR/build.log"

echo "[VORTEX] Running CTest suite..."
ctest --test-dir "$BUILD_DIR" --output-on-failure \
  2>&1 | tee "$REPORT_DIR/ctest.log"

echo "[VORTEX] Checking HELIOS verifier output..."
VORTEX_SYNC_MODE=boundary "$BUILD_DIR/vortex_infer" \
  --model "$ROOT/models/phase27_helios_verifier.vortex" \
  --input "$ROOT/models/phase27_helios/input.bin" \
  --labels "$ROOT/models/phase27_helios/labels.txt" \
  --reference "$ROOT/models/phase27_helios/reference.bin" \
  --tolerance 0.0001 \
  --json \
  2>&1 | tee "$REPORT_DIR/helios_verifier.log"

echo "[VORTEX] GPU validation passed."
echo "[VORTEX] Logs: $REPORT_DIR"
