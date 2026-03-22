from pathlib import Path

MACRO_DIR = Path(__file__).parent.parent / "macros" / "generated"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRIES = ["rectilinear", "honeycomb", "gyroid", "cubic", "voronoi"]
INFILL_LEVELS = [20, 40, 60, 80, 100]
ENERGIES = [2, 4, 6]

CELL_SIZES = {
    "rectilinear": {20: 3.79, 40: 1.78, 60: 1.09, 80: 0.72, 100: 0},
    "honeycomb":   {20: 4.0, 40: 2.0, 60: 1.33, 80: 1.0, 100: 0},
    "gyroid":      {20: 7.0, 40: 4.5, 60: 3.0, 80: 2.0, 100: 0},
    "cubic":       {20: 6.0, 40: 3.0, 60: 2.0, 80: 1.5, 100: 0},
    "voronoi":     {20: 4.0, 40: 2.5, 60: 1.8, 80: 1.2, 100: 0},
}

N_EVENTS = 1_000_000

def write_macro(filename, geometry, infill, energy_gev, cell_size,
                n_events=N_EVENTS, sample_thickness_mm=10,
                extra_settings=""):
    tag = f"{geometry}_{infill}pct_{energy_gev}GeV"
    content = f"""# auto-generated macro: {tag}
# geometry: {geometry}, Infill: {infill}%, Energy: {energy_gev} GeV
/control/execute macros/init.mac

/MCS/det/geometry {geometry}
/MCS/det/infill {infill}
"""
    if cell_size > 0:
        content += f"/MCS/det/cellSize {cell_size} mm\n"

    content += f"""/MCS/det/wallThickness 0.4 mm
/MCS/det/sampleThickness {sample_thickness_mm} mm
/MCS/det/sampleWidth 20 mm
{extra_settings}
/run/initialize

/gun/energy {energy_gev} GeV
/MCS/output/fileName {tag}

/run/beamOn {n_events}
"""
    filepath = MACRO_DIR / filename
    filepath.write_text(content)
    return filepath

def generate_primary_matrix():
    macros = []
    for geom in GEOMETRIES:
        for infill in INFILL_LEVELS:
            cell = CELL_SIZES[geom][infill]
            for energy in ENERGIES:
                fname = f"{geom}_{infill}pct_{energy}GeV.mac"
                m = write_macro(fname, geom, infill, energy, cell)
                macros.append(m)
    return macros

def generate_controls():
    macros = []
    for energy in ENERGIES:
        fname = f"control_solid_{energy}GeV.mac"
        m = write_macro(fname, "solid", 100, energy, 0)
        macros.append(m)
        fname = f"control_air_{energy}GeV.mac"
        m = write_macro(fname, "air", 0, energy, 0, extra_settings="")
        macros.append(m)
    return macros

def generate_sweep1_physics():
    macros = []
    configs = [
        ("1a", 0.02, "finest reference"),
        ("1b", 0.04, "production baseline"),
        ("1c", 0.08, "EMZ default"),
        ("1d", 0.20, "coarse speed test"),
    ]
    for label, rf, desc in configs:
        fname = f"sweep1{label}_rf{str(rf).replace('.','')}.mac"
        extra = f"/MCS/phys/rangeFactor {rf}\n# {desc}"
        m = write_macro(fname, "gyroid", 40, 4, 4.5,
                        extra_settings=extra)
        macros.append(m)
    return macros

def generate_sweep2_infill():
    macros = []
    for infill in range(5, 101, 5):
        cell = 4.5 * (40.0 / max(infill, 5)) ** 0.5 if infill < 100 else 0
        cell = round(cell, 1)
        fname = f"sweep2_gyroid_{infill}pct_4GeV.mac"
        m = write_macro(fname, "gyroid", infill, 4, cell)
        macros.append(m)
    return macros

def generate_sweep3_geometry():
    macros = []
    for infill in [20, 40, 60]:
        for geom in GEOMETRIES:
            cell = CELL_SIZES[geom][infill]
            fname = f"sweep3_{geom}_{infill}pct_4GeV.mac"
            m = write_macro(fname, geom, infill, 4, cell)
            macros.append(m)
    return macros

def generate_sweep4_cellsize():
    macros = []
    for cell in [0.5, 1.0, 2.0, 4.0, 8.0]:
        fname = f"sweep4_gyroid_40pct_cell{cell}mm_4GeV.mac"
        m = write_macro(fname, "gyroid", 40, 4, cell)
        macros.append(m)
    return macros

def generate_sweep5_energy():
    macros = []
    for infill in [20, 50, 100]:
        geom = "solid" if infill == 100 else "gyroid"
        cell = 7.0 if infill == 20 else (3.6 if infill == 50 else 0)
        for energy in [1, 2, 3, 4, 5, 6]:
            fname = f"sweep5_{geom}_{infill}pct_{energy}GeV.mac"
            m = write_macro(fname, geom, infill, energy, cell)
            macros.append(m)
    return macros

def generate_sweep6_thickness():
    macros = []
    for thick in [2, 5, 10, 20, 40]:
        fname = f"sweep6_gyroid_20pct_thick{thick}mm_4GeV.mac"
        m = write_macro(fname, "gyroid", 20, 4, 7.0,
                        sample_thickness_mm=thick)
        macros.append(m)
    return macros

def main():
    all_macros = []

    print("Generating primary matrix (75 macros)...")
    all_macros += generate_primary_matrix()

    print("Generating controls (6 macros)...")
    all_macros += generate_controls()

    print("Generating Sweep 1: physics validation (4 macros)...")
    all_macros += generate_sweep1_physics()

    print("Generating Sweep 2: infill scan (20 macros)...")
    all_macros += generate_sweep2_infill()

    print("Generating Sweep 3: geometry comparison (15 macros)...")
    all_macros += generate_sweep3_geometry()

    print("Generating Sweep 4: cell size scaling (5 macros)...")
    all_macros += generate_sweep4_cellsize()

    print("Generating Sweep 5: energy dependence (18 macros)...")
    all_macros += generate_sweep5_energy()

    print("Generating Sweep 6: thickness optimization (5 macros)...")
    all_macros += generate_sweep6_thickness()

    print(f"\nTotal: {len(all_macros)} macro files in {MACRO_DIR}/")

    master = MACRO_DIR / "all_macros.txt"
    with open(master, 'w') as f:
        for m in all_macros:
            f.write(f"{m.name}\n")
    print(f"Master list: {master}")

if __name__ == "__main__":
    main()

