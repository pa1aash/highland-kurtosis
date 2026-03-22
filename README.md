# Beam Test of the Highland Formula in 3D-Printed Lattice Materials

**Team Highland and Seek** -- BL4S 2026

Geant4 simulation framework for validating the Highland multiple Coulomb scattering formula in 3D-printed PLA lattice structures at DESY II test beam energies (2-6 GeV electrons).

## Physics

The Highland formula predicts the RMS scattering angle for particles traversing homogeneous material. In lattice metamaterials, particles sample different path lengths depending on entry position, producing a Gaussian scale mixture. The primary observable is the excess kurtosis of the angular distribution, which is guaranteed positive by Jensen's inequality for any non-degenerate mixture.

The excess kurtosis separates into a material term and a geometry term:

```
kappa = kappa_M + kappa_geo * (1 + kappa_M / 3)
```

For binary lattices (infill fraction f): `kappa = (3 + kappa_M) / f - 3`

## Prerequisites

- **Geant4** >= 11.0 (with data libraries)
- **ROOT** (for `.root` output)
- **CMake** >= 3.16
- **Python 3** with: numpy, scipy, matplotlib, uproot, scikit-image, numpy-stl

## Installation

```bash
./scripts/install_geant4.sh conda
./scripts/install_geant4.sh --check
```

## Building

```bash
conda activate g4highland
mkdir -p build && cd build
cmake ..
make -j$(sysctl -n hw.ncpu)    # macOS
# make -j$(nproc)               # Linux
```

## Running Simulations

### Quick test

```bash
cd build
./MCSHighland ../macros/run_solid_control.mac
```

### Proposal campaign (27 configs, 50k events each)

```bash
./scripts/run_proposal_campaign.sh
```

### All sweeps

```bash
./scripts/run_all_sweeps.sh
./scripts/run_all_sweeps.sh --sweep 2
./scripts/run_all_sweeps.sh --jobs 4
```

## Simulation Matrix

5 geometries x 4 infill levels (20/40/60/80%) + controls, at 3 beam energies.

| Geometry | Method | Infill levels (%) |
|----------|--------|-------------------|
| Rectilinear | CSG wall slabs | 20, 40, 60, 80 |
| Honeycomb | Voxelised hex walls | 20, 40, 60, 80 |
| Gyroid | Voxelised TPMS | 20, 40, 60, 80 |
| Cubic (3D Grid) | CSG orthogonal walls | 20, 40, 60, 80 |
| Voronoi | Voxelised CVT | 20, 40, 60, 80 |

**Controls:** Solid PLA (100%), Air-only (null test)

**Beam:** 2, 4, 6 GeV electrons (DESY II test beam), Gaussian spot sigma = 5 mm

## Project Structure

```
Geant4Highland/
├── CMakeLists.txt
├── MCSHighland.cc
├── include/
│   ├── DetectorConstruction.hh
│   ├── DetectorMessenger.hh
│   ├── PhysicsList.hh
│   ├── PrimaryGeneratorAction.hh
│   ├── ActionInitialization.hh
│   ├── RunAction.hh
│   ├── EventAction.hh
│   └── SteppingAction.hh
├── src/
│   ├── DetectorConstruction.cc
│   ├── DetectorMessenger.cc
│   ├── PhysicsList.cc
│   ├── PrimaryGeneratorAction.cc
│   ├── ActionInitialization.cc
│   ├── RunAction.cc
│   ├── EventAction.cc
│   └── SteppingAction.cc
├── macros/
│   ├── init.mac
│   ├── vis.mac
│   ├── run_solid_control.mac
│   └── run_air_control.mac
├── scripts/
│   ├── install_geant4.sh
│   ├── generate_macros.py
│   ├── generate_geometry.py
│   ├── ray_trace_sweep0.py
│   ├── run_all_sweeps.sh
│   └── run_proposal_campaign.sh
├── analysis/
│   ├── generate_stls.py
│   ├── visualize_stls.py
│   ├── ray_trace_sweep0.py
│   ├── proposal_analysis.py
│   ├── proposal_figures.py
│   ├── generate_proposal_figures.py
│   ├── generate_theory_comparison.py
│   ├── n_scaling_analysis.py
│   ├── bl4s_predictions.py
│   ├── universal_equation_final.py
│   └── analyze_mcs.py
├── stl_outputs/
├── proposal_targets/
├── proposal_figures/
├── data/
└── results/
```

## Macro Commands

| Command | Values | Default |
|---------|--------|---------|
| `/MCS/det/geometry` | `solid`, `air`, `rectilinear`, `honeycomb`, `gyroid`, `cubic`, `voronoi` | `solid` |
| `/MCS/det/infill` | 0-100 (%) | 40 |
| `/MCS/det/cellSize` | mm | 2.0 |
| `/MCS/det/wallThickness` | mm | 0.4 |
| `/MCS/det/sampleThickness` | mm | 10 |
| `/MCS/det/sampleWidth` | mm | 20 |
| `/gun/energy` | GeV | 4 |
| `/MCS/gun/beamSigma` | mm | 5 |
| `/MCS/gun/pencilBeam` | true/false | false |
| `/MCS/output/fileName` | string | MCSOutput |

## Analysis Workflow

```bash
python analysis/ray_trace_sweep0.py
python analysis/generate_stls.py
python analysis/visualize_stls.py
./scripts/run_proposal_campaign.sh
python analysis/proposal_analysis.py
python analysis/proposal_figures.py
```

## Output Format

ROOT ntuple `scattering` with columns:

| Column | Unit | Description |
|--------|------|-------------|
| `theta_x` | rad | Projected scattering angle (x) |
| `theta_y` | rad | Projected scattering angle (y) |
| `theta_space` | rad | Space scattering angle |
| `energy_out` | GeV | Exit kinetic energy |
| `entry_x` | mm | Entry x position |
| `entry_y` | mm | Entry y position |
| `pla_path` | mm | Total PLA path length |

## Physics Settings

- **Physics list:** FTFP_BERT with G4EmStandardPhysics_option4 (EMZ)
- **MCS model:** WentzelVI + single Coulomb scattering
- **Step control:** RangeFactor = 0.04, MaxStep = 0.1 mm
- **Production cuts:** 1.0 mm global, 0.1 mm in target region
- **Material:** PLA (C3H4O2)n, rho = 1.24 g/cm3, X0 = 315 mm
- **Beam:** Gaussian profile (sigma = 5 mm), 2/4/6 GeV electrons

## 3D Printing

STL files for all target geometries (20x20x10 mm PLA blocks):

```bash
pip install numpy scikit-image numpy-stl
python analysis/generate_stls.py
```

Print at 100% infill -- the STL IS the geometry. See `stl_outputs/README.md` for printing instructions.
