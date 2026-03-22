#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
MACRO_DIR="${PROJECT_DIR}/macros/generated/phase3_stacked"
EXECUTABLE="${BUILD_DIR}/MCSHighland"
DATA_DIR="${PROJECT_DIR}/data/phase3_stacked"

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found. Run 'cd build && cmake .. && make' first."
    exit 1
fi

mkdir -p "${DATA_DIR}"

echo "Phase 3: Stacked rectilinear 20% — nLayers = 1, 2, 4, 10, 20 (5 macros)..."
for macro in "${MACRO_DIR}"/stacked_*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${PROJECT_DIR}"
    "${EXECUTABLE}" "${macro}" > "${DATA_DIR}/${name}.log" 2>&1
    # Move ROOT output to data directory (output lands in CWD)
    for ext in root csv; do
        if ls "${PROJECT_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${PROJECT_DIR}"/*.${ext} "${DATA_DIR}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

echo ""
echo "Phase 3 complete (5 macros)."
echo "  Output: ${DATA_DIR}/"
