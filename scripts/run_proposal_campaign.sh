#!/bin/bash
set -euo pipefail

BUILD_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../build" && pwd)"
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/proposal"
EXECUTABLE="${BUILD_DIR}/MCSHighland"

NEVENTS=50000

mkdir -p "${DATA_DIR}"

run_sim() {
    local tag="$1"
    local geom="$2"
    local infill="$3"
    local cell="$4"
    local energy="$5"
    local thick="${6:-10}"

    local seed=$(echo -n "${tag}" | cksum | cut -d' ' -f1)

    local macfile="${DATA_DIR}/${tag}.mac"
    cat > "${macfile}" << EOF
/control/verbose 0
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/MCS/phys/rangeFactor 0.04
/MCS/det/geometry ${geom}
/MCS/det/infill ${infill}
EOF
    if [ "${cell}" != "0" ]; then
        echo "/MCS/det/cellSize ${cell} mm" >> "${macfile}"
    fi
    cat >> "${macfile}" << EOF
/MCS/det/wallThickness 0.4 mm
/MCS/det/sampleThickness ${thick} mm
/MCS/det/sampleWidth 20 mm
/run/initialize
/gun/energy ${energy} GeV
/MCS/gun/beamSigma 5 mm
/MCS/output/fileName ${DATA_DIR}/${tag}
/run/beamOn ${NEVENTS}
EOF

    echo "  [START] ${tag} (seed=${seed})"
    cd "${BUILD_DIR}"
    ./MCSHighland "${macfile}" "${seed}" > "${DATA_DIR}/${tag}.log" 2>&1
    echo "  [DONE]  ${tag}"
}

run_sim_n() {
    local tag="$1"
    local geom="$2"
    local infill="$3"
    local cell="$4"
    local energy="$5"
    local thick="${6:-10}"
    local nevt="${7:-${NEVENTS}}"

    local seed=$(echo -n "${tag}" | cksum | cut -d' ' -f1)

    local macfile="${DATA_DIR}/${tag}.mac"
    cat > "${macfile}" << EOF
/control/verbose 0
/run/verbose 0
/event/verbose 0
/tracking/verbose 0
/MCS/phys/rangeFactor 0.04
/MCS/det/geometry ${geom}
/MCS/det/infill ${infill}
EOF
    if [ "${cell}" != "0" ]; then
        echo "/MCS/det/cellSize ${cell} mm" >> "${macfile}"
    fi
    cat >> "${macfile}" << EOF
/MCS/det/wallThickness 0.4 mm
/MCS/det/sampleThickness ${thick} mm
/MCS/det/sampleWidth 20 mm
/run/initialize
/gun/energy ${energy} GeV
/MCS/gun/beamSigma 5 mm
/MCS/output/fileName ${DATA_DIR}/${tag}
/run/beamOn ${nevt}
EOF

    echo "  [START] ${tag} (seed=${seed}, N=${nevt})"
    cd "${BUILD_DIR}"
    ./MCSHighland "${macfile}" "${seed}" > "${DATA_DIR}/${tag}.log" 2>&1
    echo "  [DONE]  ${tag}"
}

echo "Proposal Validation Campaign"
echo "Events per config: ${NEVENTS}"
echo "Output: ${DATA_DIR}/"

echo ""
echo "--- Controls ---"
run_sim "control_solid_2GeV" "solid" 100 0 2
run_sim "control_solid_4GeV" "solid" 100 0 4
run_sim "control_solid_6GeV" "solid" 100 0 6
run_sim "control_air_4GeV"   "air"   0   0 4

echo ""
echo "--- Rectilinear infill scan (4 GeV) ---"
run_sim "rect_20pct_4GeV" "rectilinear" 20 3.79 4
run_sim "rect_40pct_4GeV" "rectilinear" 40 1.78 4
run_sim "rect_60pct_4GeV" "rectilinear" 60 1.09 4
run_sim "rect_80pct_4GeV" "rectilinear" 80 0.72 4

echo ""
echo "--- Honeycomb infill scan (4 GeV) ---"
run_sim "hc_20pct_4GeV" "honeycomb" 20 4.0 4
run_sim "hc_40pct_4GeV" "honeycomb" 40 2.0 4
run_sim "hc_60pct_4GeV" "honeycomb" 60 1.33 4
run_sim "hc_80pct_4GeV" "honeycomb" 80 1.0 4

echo ""
echo "--- Gyroid infill scan (4 GeV) ---"
run_sim "gyr_20pct_4GeV" "gyroid" 20 7.0 4
run_sim "gyr_40pct_4GeV" "gyroid" 40 4.5 4
run_sim "gyr_60pct_4GeV" "gyroid" 60 3.0 4
run_sim "gyr_80pct_4GeV" "gyroid" 80 2.0 4

echo ""
echo "--- 3D Grid infill scan (4 GeV) ---"
run_sim "cub_20pct_4GeV" "cubic" 20 6.0 4
run_sim "cub_40pct_4GeV" "cubic" 40 3.0 4
run_sim "cub_60pct_4GeV" "cubic" 60 2.0 4
run_sim "cub_80pct_4GeV" "cubic" 80 1.5 4

echo ""
echo "--- Voronoi infill scan (4 GeV, 200k events) ---"
run_sim_n "vor_20pct_4GeV" "voronoi" 20 4.0 4 10 200000
run_sim_n "vor_40pct_4GeV" "voronoi" 40 2.5 4 10 200000
run_sim_n "vor_60pct_4GeV" "voronoi" 60 1.8 4 10 200000
run_sim_n "vor_80pct_4GeV" "voronoi" 80 1.2 4 10 200000

echo ""
echo "--- Energy scan: 40% rectilinear ---"
run_sim "rect_40pct_2GeV" "rectilinear" 40 1.78 2
run_sim "rect_40pct_6GeV" "rectilinear" 40 1.78 6

echo ""
echo "--- Energy scan: 40% gyroid ---"
run_sim "gyr_40pct_2GeV" "gyroid" 40 4.5 2
run_sim "gyr_40pct_6GeV" "gyroid" 40 4.5 6

echo ""
echo "Campaign complete! ${DATA_DIR}/"
ls -la "${DATA_DIR}"/*.root 2>/dev/null | wc -l | xargs echo "ROOT files:"
