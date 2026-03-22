#!/bin/bash
set -euo pipefail

conda activate g4highland

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
MACRO_DIR="${PROJECT_DIR}/macros/generated/phase01_thin_wall"
DATA_DIR="${PROJECT_DIR}/data/phase01_thin_wall"
EXECUTABLE="${BUILD_DIR}/MCSHighland"

# Generate macros first
echo "Generating Phase 0.1 thin-wall macros..."
cd "${PROJECT_DIR}"
python3 scripts/generate_thin_wall_macros.py

# Ensure output directory exists
mkdir -p "${DATA_DIR}"

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found. Run 'cd build && cmake .. && make' first."
    exit 1
fi

echo ""
echo "Running Phase 0.1: thin-wall kappa_M parametrisation (27 macros)..."

for macro in "${MACRO_DIR}"/*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./MCSHighland "${macro}" > "${DATA_DIR}/${name}.log" 2>&1
    # Move ROOT output to data directory
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_DIR}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

echo ""
echo "Phase 0.1 complete. Output: ${DATA_DIR}/"
