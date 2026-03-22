#!/usr/bin/env python3
"""Generate macros for Phase 1: material and particle expansion.

Produces 14 macros across 4 sub-phases:
  Phase 1.1 — Silicon lattices (4 macros)
  Phase 1.2 — Tungsten lattices (4 macros)
  Phase 1.3 — Muons (2 macros)
  Phase 1.4 — Thickness variation (4 macros)

Output directory: macros/generated/phase1/
"""

from pathlib import Path

MACRO_DIR = Path(__file__).parent.parent / "macros" / "generated" / "phase1"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

N_EVENTS = 100_000

# Rectilinear cell sizes from generate_macros.py
RECTILINEAR_CELL_SIZES = {40: 1.78, 60: 1.09, 80: 0.72}


def write_macro(filename, tag, geometry, energy_gev, *,
                material="PLA", particle="e-", infill=None,
                cell_size=None, sample_thickness_mm=10):
    """Write a single Geant4 macro file."""
    lines = [
        f"# Phase 1: {tag}",
        f"# material={material}, particle={particle}, geometry={geometry}, "
        f"energy={energy_gev} GeV",
        "/control/execute macros/init.mac",
        "",
    ]

    lines.append(f"/MCS/det/geometry {geometry}")
    lines.append(f"/MCS/det/material {material}")
    lines.append(f"/MCS/gun/particle {particle}")

    if infill is not None:
        lines.append(f"/MCS/det/infill {infill}")
    if cell_size is not None and cell_size > 0:
        lines.append(f"/MCS/det/cellSize {cell_size} mm")

    lines.append("/MCS/det/wallThickness 0.4 mm")
    lines.append(f"/MCS/det/sampleThickness {sample_thickness_mm} mm")
    lines.append("/MCS/det/sampleWidth 20 mm")
    lines.append("")
    lines.append("/run/initialize")
    lines.append("")
    lines.append(f"/gun/energy {energy_gev} GeV")
    lines.append(f"/MCS/output/fileName {tag}")
    lines.append("")
    lines.append(f"/run/beamOn {N_EVENTS}")
    lines.append("")

    content = "\n".join(lines)
    filepath = MACRO_DIR / filename
    filepath.write_text(content)
    return filepath


def generate_phase1_1_silicon():
    """Phase 1.1: Silicon lattices — rectilinear 40/60/80% + solid control."""
    macros = []
    for infill in [40, 60, 80]:
        cell = RECTILINEAR_CELL_SIZES[infill]
        tag = f"si_rect_{infill}pct_4GeV"
        m = write_macro(f"{tag}.mac", tag, "rectilinear", 4,
                        material="silicon", infill=infill, cell_size=cell)
        macros.append(m)

    tag = "si_solid_4GeV"
    m = write_macro(f"{tag}.mac", tag, "solid", 4, material="silicon")
    macros.append(m)
    return macros


def generate_phase1_2_tungsten():
    """Phase 1.2: Tungsten lattices — rectilinear 40/60/80% + solid control."""
    macros = []
    for infill in [40, 60, 80]:
        cell = RECTILINEAR_CELL_SIZES[infill]
        tag = f"w_rect_{infill}pct_4GeV"
        m = write_macro(f"{tag}.mac", tag, "rectilinear", 4,
                        material="tungsten", infill=infill, cell_size=cell)
        macros.append(m)

    tag = "w_solid_4GeV"
    m = write_macro(f"{tag}.mac", tag, "solid", 4, material="tungsten")
    macros.append(m)
    return macros


def generate_phase1_3_muons():
    """Phase 1.3: Muons — PLA rectilinear 40% + solid control."""
    macros = []

    tag = "muon_rect_40pct_4GeV"
    cell = RECTILINEAR_CELL_SIZES[40]
    m = write_macro(f"{tag}.mac", tag, "rectilinear", 4,
                    particle="mu-", infill=40, cell_size=cell)
    macros.append(m)

    tag = "muon_solid_4GeV"
    m = write_macro(f"{tag}.mac", tag, "solid", 4, particle="mu-")
    macros.append(m)
    return macros


def generate_phase1_4_thickness():
    """Phase 1.4: Thickness variation — PLA rectilinear 40% at 5/10/20/40 mm."""
    macros = []
    cell = RECTILINEAR_CELL_SIZES[40]
    for thick in [5, 10, 20, 40]:
        tag = f"thick_{thick}mm_rect_40pct_4GeV"
        m = write_macro(f"{tag}.mac", tag, "rectilinear", 4,
                        infill=40, cell_size=cell,
                        sample_thickness_mm=thick)
        macros.append(m)
    return macros


def main():
    all_macros = []

    print("Phase 1.1: Silicon lattices (4 macros)...")
    all_macros += generate_phase1_1_silicon()

    print("Phase 1.2: Tungsten lattices (4 macros)...")
    all_macros += generate_phase1_2_tungsten()

    print("Phase 1.3: Muons (2 macros)...")
    all_macros += generate_phase1_3_muons()

    print("Phase 1.4: Thickness variation (4 macros)...")
    all_macros += generate_phase1_4_thickness()

    print(f"\nPhase 1: generated {len(all_macros)} macros in {MACRO_DIR}/")

    manifest = MACRO_DIR / "manifest.txt"
    with open(manifest, "w") as f:
        for m in all_macros:
            f.write(f"{m.name}\n")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
