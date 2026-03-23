#!/usr/bin/env python3
"""Generate macros for Phase 0.4: EM physics model comparison.

Produces 2 macros (solid control + rectilinear 40%) at 4 GeV.
The physics option (0, 3, 4) is a command-line argument, not a macro command,
so run_model_comparison.sh handles the EM option loop.

Output directory: macros/generated/phase04_model_comparison/
"""

from pathlib import Path

MACRO_DIR = Path(__file__).parent.parent / "macros" / "generated" / "phase04_model_comparison"
MACRO_DIR.mkdir(parents=True, exist_ok=True)

ENERGY_GEV = 4
N_EVENTS = 100_000

CONFIGS = [
    {
        "tag": "model_solid_4GeV",
        "geometry": "solid",
        "description": "Solid PLA control",
        "extra_cmds": "",
    },
    {
        "tag": "model_rect40_4GeV",
        "geometry": "rectilinear",
        "description": "Rectilinear 40% infill",
        "extra_cmds": (
            "/MCS/det/infill 40\n"
            "/MCS/det/cellSize 1.78 mm\n"
            "/MCS/det/wallThickness 0.4 mm\n"
        ),
    },
]


def write_macro(cfg):
    content = f"""# Phase 0.4: EM physics model comparison
# {cfg['description']}, energy={ENERGY_GEV} GeV
# Physics option selected via command-line argument (0=Urban, 3=opt3, 4=opt4)
/control/execute macros/init.mac

/MCS/det/geometry {cfg['geometry']}
{cfg['extra_cmds']}/MCS/det/sampleThickness 10 mm
/MCS/det/sampleWidth 20 mm

/run/initialize

/gun/energy {ENERGY_GEV} GeV
/MCS/output/fileName {cfg['tag']}

/run/beamOn {N_EVENTS}
"""
    filepath = MACRO_DIR / f"{cfg['tag']}.mac"
    filepath.write_text(content)
    return filepath


def main():
    macros = []
    for cfg in CONFIGS:
        m = write_macro(cfg)
        macros.append(m)

    print(f"Phase 0.4: generated {len(macros)} model-comparison macros in {MACRO_DIR}/")

    manifest = MACRO_DIR / "manifest.txt"
    with open(manifest, "w") as f:
        for m in macros:
            f.write(f"{m.name}\n")
    print(f"Manifest: {manifest}")


if __name__ == "__main__":
    main()
