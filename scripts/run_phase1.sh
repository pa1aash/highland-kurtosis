#!/bin/bash
set -euo pipefail

conda activate g4highland

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
MACRO_DIR="${PROJECT_DIR}/macros/generated/phase1"
EXECUTABLE="${BUILD_DIR}/MCSHighland"

# Generate macros first
echo "Generating Phase 1 macros..."
cd "${PROJECT_DIR}"
python3 scripts/generate_phase1_macros.py

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: ${EXECUTABLE} not found. Run 'cd build && cmake .. && make' first."
    exit 1
fi

# Phase 1.1 — Silicon
DATA_SI="${PROJECT_DIR}/data/phase1_silicon"
mkdir -p "${DATA_SI}"
echo ""
echo "Phase 1.1: Silicon lattices (4 macros)..."
for macro in "${MACRO_DIR}"/si_*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./MCSHighland "${macro}" > "${DATA_SI}/${name}.log" 2>&1
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_SI}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

# Phase 1.2 — Tungsten
DATA_W="${PROJECT_DIR}/data/phase1_tungsten"
mkdir -p "${DATA_W}"
echo ""
echo "Phase 1.2: Tungsten lattices (4 macros)..."
for macro in "${MACRO_DIR}"/w_*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./MCSHighland "${macro}" > "${DATA_W}/${name}.log" 2>&1
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_W}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

# Phase 1.3 — Muons
DATA_MU="${PROJECT_DIR}/data/phase1_muons"
mkdir -p "${DATA_MU}"
echo ""
echo "Phase 1.3: Muons (2 macros)..."
for macro in "${MACRO_DIR}"/muon_*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./MCSHighland "${macro}" > "${DATA_MU}/${name}.log" 2>&1
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_MU}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

# Phase 1.4 — Thickness variation
DATA_TH="${PROJECT_DIR}/data/phase1_thickness"
mkdir -p "${DATA_TH}"
echo ""
echo "Phase 1.4: Thickness variation (4 macros)..."
for macro in "${MACRO_DIR}"/thick_*.mac; do
    name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./MCSHighland "${macro}" > "${DATA_TH}/${name}.log" 2>&1
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_TH}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
done

echo ""
echo "Phase 1 complete (14 macros)."
echo "  Silicon:   ${DATA_SI}/"
echo "  Tungsten:  ${DATA_W}/"
echo "  Muons:     ${DATA_MU}/"
echo "  Thickness: ${DATA_TH}/"
