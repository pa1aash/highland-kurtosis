# MCS Highland — Geant4 Lattice Scattering Simulation

Geant4 simulation framework for validating the Highland multiple Coulomb scattering formula in 3D-printed PLA lattice structures (rectilinear, honeycomb, gyroid, cubic, voronoi) at DESY II test beam energies (2-6 GeV electrons). The primary observable is excess kurtosis of the angular distribution, which reveals non-Gaussian tails from path-length variation across lattice geometries. See `REPO_AUDIT.md` for the full codebase inventory.

## Environment

Always run `conda activate g4highland` before any build or simulation commands. If a shell script runs cmake, make, or the MCSHighland executable, it should include `conda activate g4highland` at the top (after `set -euo pipefail`).

## Build

```bash
conda activate g4highland          # or source geant4 environment
mkdir -p build && cd build
cmake ..                           # add -DWITH_CADMESH=ON for STL import
make -j$(sysctl -n hw.ncpu)        # macOS
```

## Run

```bash
cd build
./MCSHighland ../macros/run_solid_control.mac          # batch mode
./MCSHighland ../macros/run_solid_control.mac 12345     # with seed
./MCSHighland                                           # interactive (vis.mac)
```

Macros configure geometry, beam, and output via `/MCS/...` commands before `/run/initialize`.

## Layout

| Directory | Contents |
|-----------|----------|
| `src/`, `include/` | Geant4 C++ source (8 classes) |
| `macros/` | Hand-written macros; `generated/` subdir from `generate_macros.py` |
| `scripts/` | Build/run infrastructure, ray-trace, macro generation |
| `analysis/` | Python analysis pipeline (analyze_mcs.py is the main entry) |
| `data/` | Simulation outputs; `sweep0/` has ray-trace predictions |
| `results/` | Analysis outputs (created by scripts, gitignored) |
| `proposal_targets/` | STL files for 3D printing |
| `proposal_figures/` | Proposal publication figures |

## Output ntuple

File: `{fileName}.root`, tree name: `scattering`

| Branch | Type | Unit | Description |
|--------|------|------|-------------|
| `theta_x` | Double | rad | Projected scattering angle (x) |
| `theta_y` | Double | rad | Projected scattering angle (y) |
| `theta_space` | Double | rad | Space scattering angle |
| `energy_out` | Double | GeV | Exit kinetic energy |
| `entry_x` | Double | mm | Beam entry x position |
| `entry_y` | Double | mm | Beam entry y position |
| `pla_path` | Double | mm | Total PLA path length |

## Physics settings

- **EM physics:** G4EmStandardPhysics_option4 (EMZ) — hardcoded
- **MCS model:** WentzelVI + single Coulomb scattering
- **Step control:** RangeFactor=0.04, MaxStep=0.1mm, StepLimitType=UseSafetyPlus, Skin=3
- **Production cuts:** 1.0mm global, 0.1mm in target region
- **Material:** PLA (C₃H₄O₂)ₙ ρ=1.24 g/cm³ X₀≈315mm (default), silicon (G4_Si) ρ=2.33 g/cm³ X₀=93.7mm, tungsten (G4_W) ρ=19.3 g/cm³ X₀=3.5mm — selectable via `/MCS/det/material`
- **Particle:** e⁻ (hardcoded), default 4 GeV
- **Beam:** Gaussian spot σ=5mm, direction +z, origin z=-50mm

## Key macro commands

```
/MCS/det/geometry solid|air|rectilinear|honeycomb|gyroid|cubic|voronoi
/MCS/det/material PLA|silicon|tungsten
/MCS/det/infill 40              # percent
/MCS/det/sampleThickness 10 mm
/MCS/det/cellSize 4.0 mm
/MCS/det/wallThickness 0.4 mm
/MCS/output/fileName MCSOutput
/MCS/gun/pencilBeam false
/MCS/gun/beamSigma 5 mm
/MCS/phys/rangeFactor 0.04
/gun/energy 4 GeV               # Geant4 built-in
```


## Post-task rule

After completing any task that modifies source code, analysis scripts, macros,
figures, or paper content, update the following files to reflect the changes:

- `REPO_AUDIT.md` — update the relevant section (e.g. if you added a new material
  to DetectorConstruction.cc, update Section 2 and Section 7 Gap 2)
- `CLAUDE.md` — update if build instructions, physics settings, macro commands,
  or ntuple schema changed
- `paper/PAPER_AUDIT.md` — update if any figures were regenerated, tables recomputed,
  or hardcoded values in the .tex changed

Keep updates minimal and precise. Don't rewrite the entire file — just update the
lines that are now stale. If nothing changed in a file, don't touch it.