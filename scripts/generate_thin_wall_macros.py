#!/usr/bin/env python3
"""Generate macros for Phase 0.1: thin-wall kappa_M parametrisation study.

Produces 27 macros (9 thicknesses x 3 energies) for solid PLA slabs.
Output directory: macros/generated/phase01_thin_wall/
"""

from pathlib import Path

MACRO_DIR = Path(__file__).parent.parent / "macros" / "generated" / "phase01_thin_wall"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

THICKNESSES_MM = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
ENERGIES_GEV = [2, 4, 6]
N_EVENTS = 100_000


def thickness_label(t):
    """Format thickness for filenames: 0.1 -> '0.1', 1.0 -> '1', 10.0 -> '10'."""
    return f"{t:g}"


def write_macro(thickness_mm, energy_gev):
    t_label = thickness_label(thickness_mm)
    tag = f"thin_wall_solid_{t_label}mm_{energy_gev}GeV"
    fname = f"{tag}.mac"

    content = f"""# Phase 0.1: thin-wall kappa_M parametrisation
# Solid PLA slab, thickness={t_label} mm, energy={energy_gev} GeV
/control/execute macros/init.mac

/MCS/det/geometry solid
/MCS/det/sampleThickness {thickness_mm} mm
/MCS/det/sampleWidth 20 mm

/run/initialize

/gun/energy {energy_gev} GeV
/MCS/output/fileName {tag}

/run/beamOn {N_EVENTS}
"""
    filepath = MACRO_DIR / fname
    filepath.write_text(content)
    return filepath


def main():
    macros = []
    for thickness in THICKNESSES_MM:
        for energy in ENERGIES_GEV:
            m = write_macro(thickness, energy)
            macros.append(m)

    print(f"Phase 0.1: generated {len(macros)} thin-wall macros in {MACRO_DIR}/")

    manifest = MACRO_DIR / "manifest.txt"
    with open(manifest, "w") as f:
        for m in macros:
            f.write(f"{m.name}\n")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
