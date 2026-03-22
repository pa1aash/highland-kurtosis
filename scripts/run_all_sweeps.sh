#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_DIR}/build"
DATA_DIR="${PROJECT_DIR}/data"
EXECUTABLE="${BUILD_DIR}/MCSHighland"

BUILD_ONLY=false
SWEEP=""
NJOBS=1

while [[ $# -gt 0 ]]; do
    case $1 in
        --build-only) BUILD_ONLY=true; shift ;;
        --sweep)      SWEEP="$2"; shift 2 ;;
        --jobs)       NJOBS="$2"; shift 2 ;;
        *)            echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "Building MCSHighland..."

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake "${PROJECT_DIR}" -DWITH_GEANT4_UIVIS=ON
make -j"${NJOBS}"

if [ "${BUILD_ONLY}" = true ]; then
    echo "Build complete. Executable: ${EXECUTABLE}"
    exit 0
fi

if [ ! -f "${EXECUTABLE}" ]; then
    echo "ERROR: Build failed -- ${EXECUTABLE} not found"
    exit 1
fi

echo ""
echo "Generating macro files..."
cd "${PROJECT_DIR}"
python3 scripts/generate_macros.py

mkdir -p "${DATA_DIR}"

run_macro() {
    local macro="$1"
    local name=$(basename "${macro}" .mac)
    echo "  Running: ${name}..."
    cd "${BUILD_DIR}"
    ./"MCSHighland" "${macro}" > "${DATA_DIR}/${name}.log" 2>&1
    for ext in root csv; do
        if ls "${BUILD_DIR}"/*.${ext} 1>/dev/null 2>&1; then
            mv "${BUILD_DIR}"/*.${ext} "${DATA_DIR}/" 2>/dev/null || true
        fi
    done
    echo "  Done: ${name}"
}

run_sweep() {
    local sweep_name="$1"
    local pattern="$2"

    echo ""
    echo "Running ${sweep_name}..."

    local macro_dir="${PROJECT_DIR}/macros/generated"
    local count=0

    for macro in "${macro_dir}"/${pattern}; do
        if [ -f "${macro}" ]; then
            if [ "${NJOBS}" -gt 1 ]; then
                run_macro "${macro}" &
                count=$((count + 1))
                if [ $((count % NJOBS)) -eq 0 ]; then
                    wait
                fi
            else
                run_macro "${macro}"
            fi
        fi
    done
    wait
    echo "${sweep_name} complete."
}

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "0" ]; then
    echo ""
    echo "Sweep 0: Ray-trace path-length distributions"
    cd "${PROJECT_DIR}"
    python3 scripts/ray_trace_sweep0.py --output-dir "${DATA_DIR}/sweep0"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "1" ]; then
    run_sweep "Sweep 1: Physics validation" "sweep1*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "2" ]; then
    run_sweep "Sweep 2: Infill scan" "sweep2*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "3" ]; then
    run_sweep "Sweep 3: Geometry comparison" "sweep3*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "4" ]; then
    run_sweep "Sweep 4: Cell size scaling" "sweep4*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "5" ]; then
    run_sweep "Sweep 5: Energy dependence" "sweep5*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "6" ]; then
    run_sweep "Sweep 6: Thickness optimization" "sweep6*.mac"
fi

if [ -z "${SWEEP}" ] || [ "${SWEEP}" = "controls" ]; then
    echo ""
    echo "Running control configurations..."
    cd "${BUILD_DIR}"
    run_macro "${PROJECT_DIR}/macros/run_solid_control.mac"
    run_macro "${PROJECT_DIR}/macros/run_air_control.mac"
fi

if [ -z "${SWEEP}" ]; then
    echo ""
    echo "Running analysis pipeline..."
    cd "${PROJECT_DIR}"
    python3 analysis/analyze_mcs.py \
        --input-dir "${DATA_DIR}" \
        --output-dir "${PROJECT_DIR}/results" \
        --sweep0 "${DATA_DIR}/sweep0/sweep0_summary.json"

    echo ""
    echo "Simulation campaign complete."
    echo "Data:    ${DATA_DIR}/"
    echo "Results: ${PROJECT_DIR}/results/"
fi
