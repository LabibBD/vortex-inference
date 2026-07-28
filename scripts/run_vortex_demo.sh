#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target vortex_infer --parallel 2

VORTEX_SYNC_MODE=boundary \
./build/vortex_infer \
    --model models/phase27_helios_verifier.vortex \
    --input models/phase27_helios/input.bin \
    --labels models/phase27_helios/labels.txt \
    --reference models/phase27_helios/reference.bin \
    --tolerance 0.0001 \
    --json

python3 scripts/demo_phase27_helios.py

echo
printf '%s\n' '[VORTEX] Demo completed successfully'
