#!/bin/bash
set -euo pipefail


PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
MACRO_DIR="${PROJECT_DIR}/macros/generated/phase04_model_comparison"
DATA_DIR="${PROJECT_DIR}/data/phase04_model_comparison"
EXECUTABLE="${BUILD_DIR}/MCSHighland"

# Generate macros first
echo "Generating Phase 0.4 model-comparison macros..."
cd "${PROJECT_DIR}"
python3 scripts/generate_model_comparison_macros.py

# Build the project
echo ""
echo "Building project..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
make -j$(sysctl -n hw.ncpu)

# Ensure output directory exists
mkdir -p "${DATA_DIR}"

echo ""
echo "Running Phase 0.4: EM physics model comparison (3 options x 2 geometries = 6 runs)..."

SEED=42
EM_OPTIONS=(0 3 4)
MACROS=("model_solid_4GeV" "model_rect40_4GeV")

for opt in "${EM_OPTIONS[@]}"; do
    for macro_name in "${MACROS[@]}"; do
        # Derive output filename: model_opt{N}_solid_4GeV or model_opt{N}_rect40_4GeV
        out_name=$(echo "${macro_name}" | sed "s/^model_/model_opt${opt}_/")
        echo "  Running: ${out_name} (EM option ${opt})..."

        cd "${PROJECT_DIR}"
        "${EXECUTABLE}" "${MACRO_DIR}/${macro_name}.mac" "${SEED}" "${opt}" \
            > "${DATA_DIR}/${out_name}.log" 2>&1

        # The macro writes fileName = model_{geom}_4GeV.root; rename to include opt
        src_root="${PROJECT_DIR}/${macro_name}.root"
        if [ -f "${src_root}" ]; then
            mv "${src_root}" "${DATA_DIR}/${out_name}.root"
        fi

        echo "  Done: ${out_name}"
    done
done

echo ""
echo "Phase 0.4 complete. Output: ${DATA_DIR}/"
ls -lh "${DATA_DIR}"/*.root 2>/dev/null || echo "  (no ROOT files found — check logs)"
