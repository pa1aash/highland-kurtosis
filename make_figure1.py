#!/usr/bin/env python3
"""
Generate Figure 1: XY cross-sections of five lattice geometries.
Publication-quality figure for NIM-A (2 rows x 5 columns, 170 mm wide).
"""

import numpy as np
from scipy.spatial import KDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ── Global constants ──────────────────────────────────────────────────────────
SAMPLE_W = 20.0   # mm
SAMPLE_T = 10.0   # mm
WALL_T   = 0.4    # mm
HALF_W   = SAMPLE_W / 2.0
HALF_T   = SAMPLE_T / 2.0
HALF_WALL = WALL_T / 2.0

RES = 0.05  # mm voxel resolution for publication quality

# Parameters from ray_trace_sweep0.py PARAMS dict
PARAMS = {
    "rectilinear": {
        20: {"cell_size": 3.79},
        40: {"cell_size": 1.78},
    },
    "honeycomb": {
        40: {"d": 1.76914},
    },
    "gyroid": {
        20: {"cell_size": 7.0, "threshold": 0.305398},
        40: {"cell_size": 4.5, "threshold": 0.61701},
    },
    "cubic": {
        40: {"cell_size": 2.5},
    },
    "voronoi": {
        20: {"cell_size": 4.0, "wall_thresh": 0.305441},
        40: {"cell_size": 2.5, "wall_thresh": 0.395338},
        60: {"cell_size": 1.8, "wall_thresh": 0.446486},
    },
}


# ── Geometry voxelisation functions ───────────────────────────────────────────

def _wall_positions(half_extent, cell_size):
    """Wall positions along one axis, clamped to sample boundary."""
    n_cells = max(1, int(2 * half_extent / cell_size))
    walls = []
    for i in range(n_cells + 1):
        pos = -half_extent + i * cell_size
        pos = np.clip(pos, -half_extent + HALF_WALL, half_extent - HALF_WALL)
        walls.append(pos)
    return np.array(walls)


def make_mask_rectilinear(cell_size, res=RES):
    """2D boolean mask for rectilinear grid (extruded along z)."""
    n = int(SAMPLE_W / res)
    x = np.linspace(-HALF_W + res / 2, HALF_W - res / 2, n)
    y = x.copy()
    XX, YY = np.meshgrid(x, y, indexing='ij')
    walls = _wall_positions(HALF_W, cell_size)
    mask = np.zeros((n, n), dtype=bool)
    for w in walls:
        mask |= (np.abs(XX - w) < HALF_WALL)
        mask |= (np.abs(YY - w) < HALF_WALL)
    return mask, x, y


def make_mask_honeycomb(d, res=RES):
    """2D boolean mask for honeycomb lattice (extruded along z)."""
    n = int(SAMPLE_W / res)
    x = np.linspace(-HALF_W + res / 2, HALF_W - res / 2, n)
    y = x.copy()
    XX, YY = np.meshgrid(x, y, indexing='ij')

    a = d / np.sqrt(3.0)
    halfA = a / 2.0
    cos60, sin60 = 0.5, np.sqrt(3.0) / 2.0
    cos120, sin120 = -0.5, np.sqrt(3.0) / 2.0
    colSp = 1.5 * a
    rowSp = d

    mask = np.zeros((n, n), dtype=bool)
    n_cols = int(SAMPLE_W / colSp) + 4
    n_rows = int(SAMPLE_W / rowSp) + 4

    for c in range(-n_cols, n_cols + 1):
        cx = c * colSp
        yOff = rowSp * 0.5 if (abs(c) % 2) else 0.0
        for r in range(-n_rows, n_rows + 1):
            cy = r * rowSp + yOff
            dx1 = XX - cx
            dy1 = YY - (cy + d * 0.5)
            hit1 = (np.abs(dx1) < halfA) & (np.abs(dy1) < HALF_WALL)

            dx2 = XX - (cx + 0.75 * a)
            dy2 = YY - (cy + d * 0.25)
            lx2 = cos120 * dx2 + sin120 * dy2
            ly2 = -sin120 * dx2 + cos120 * dy2
            hit2 = (np.abs(lx2) < halfA) & (np.abs(ly2) < HALF_WALL)

            dx3 = XX - (cx + 0.75 * a)
            dy3 = YY - (cy - d * 0.25)
            lx3 = cos60 * dx3 + sin60 * dy3
            ly3 = -sin60 * dx3 + cos60 * dy3
            hit3 = (np.abs(lx3) < halfA) & (np.abs(ly3) < HALF_WALL)

            mask |= hit1 | hit2 | hit3

    in_sample = (np.abs(XX) <= HALF_W) & (np.abs(YY) <= HALF_W)
    mask &= in_sample
    return mask, x, y


def make_mask_gyroid_slice(cell_size, threshold, z_val, res=RES):
    """2D boolean mask: XY slice of gyroid TPMS at given z."""
    n = int(SAMPLE_W / res)
    x = np.linspace(-HALF_W + res / 2, HALF_W - res / 2, n)
    y = x.copy()
    XX, YY = np.meshgrid(x, y, indexing='ij')

    k = 2.0 * np.pi / cell_size
    F = (np.sin(k * XX) * np.cos(k * YY)
       + np.sin(k * YY) * np.cos(k * z_val)
       + np.sin(k * z_val) * np.cos(k * XX))
    mask = np.abs(F) < threshold
    return mask, x, y


def make_mask_cubic(cell_size, z_val, res=RES):
    """2D boolean mask for cubic lattice at given z.
    Shows x-walls, y-walls, and z-wall if z is near a z-wall position."""
    n = int(SAMPLE_W / res)
    x = np.linspace(-HALF_W + res / 2, HALF_W - res / 2, n)
    y = x.copy()
    XX, YY = np.meshgrid(x, y, indexing='ij')

    xy_walls = _wall_positions(HALF_W, cell_size)
    z_walls = _wall_positions(HALF_T, cell_size)

    # x- and y-walls (same as rectilinear)
    in_xwall = np.zeros((n, n), dtype=bool)
    in_ywall = np.zeros((n, n), dtype=bool)
    for w in xy_walls:
        in_xwall |= (np.abs(XX - w) < HALF_WALL)
        in_ywall |= (np.abs(YY - w) < HALF_WALL)

    # Check if z_val is on a z-wall
    on_zwall = any(abs(z_val - zw) < HALF_WALL for zw in z_walls)
    if on_zwall:
        # z-wall fills gaps not covered by x/y walls
        mask = np.ones((n, n), dtype=bool)
    else:
        mask = in_xwall | in_ywall
    return mask, x, y


def make_voronoi_seeds(cell_size, n_lloyd=5, n_probe_lloyd=50000):
    """Generate CVT-relaxed Voronoi seeds in 3D sample volume (seed=42)."""
    rng = np.random.default_rng(42)
    cell_vol = cell_size ** 3
    samp_vol = SAMPLE_W * SAMPLE_W * SAMPLE_T
    n_seeds = max(4, int(samp_vol / cell_vol))

    seeds = np.column_stack([
        rng.uniform(-HALF_W, HALF_W, n_seeds),
        rng.uniform(-HALF_W, HALF_W, n_seeds),
        rng.uniform(-HALF_T, HALF_T, n_seeds),
    ])

    for _ in range(n_lloyd):
        pts = np.column_stack([
            rng.uniform(-HALF_W, HALF_W, n_probe_lloyd),
            rng.uniform(-HALF_W, HALF_W, n_probe_lloyd),
            rng.uniform(-HALF_T, HALF_T, n_probe_lloyd),
        ])
        tree = KDTree(seeds)
        _, closest = tree.query(pts)
        for s in range(n_seeds):
            m = closest == s
            if np.any(m):
                seeds[s] = pts[m].mean(axis=0)

    return seeds


def make_mask_voronoi_slice(cell_size, wall_thresh, z_val, res=RES):
    """2D boolean mask: XY slice of Voronoi foam at given z."""
    seeds = make_voronoi_seeds(cell_size)
    tree = KDTree(seeds)

    n = int(SAMPLE_W / res)
    x = np.linspace(-HALF_W + res / 2, HALF_W - res / 2, n)
    y = x.copy()
    XX, YY = np.meshgrid(x, y, indexing='ij')

    pts = np.column_stack([
        XX.ravel(),
        YY.ravel(),
        np.full(n * n, z_val),
    ])
    dd, _ = tree.query(pts, k=2)
    d1 = dd[:, 0]
    d2 = dd[:, 1]
    mask = ((d2 - d1) < wall_thresh).reshape(n, n)
    return mask, x, y


# ── Panel definitions ─────────────────────────────────────────────────────────

PANELS = [
    # Row 1
    {"label": "(a)", "geom": "Rectilinear, 20%",     "type": "rectilinear", "infill": 20},
    {"label": "(b)", "geom": "Rectilinear, 40%",     "type": "rectilinear", "infill": 40},
    {"label": "(c)", "geom": "Honeycomb, 40%",        "type": "honeycomb",   "infill": 40},
    {"label": "(d)", "geom": "Gyroid, 20%",           "type": "gyroid",      "infill": 20, "z": 0.0,
     "z_note": "z = 0 mm"},
    {"label": "(e)", "geom": "Gyroid, 40%",           "type": "gyroid",      "infill": 40, "z": 0.0,
     "z_note": "z = 0 mm"},
    # Row 2
    {"label": "(f)", "geom": "Gyroid, 40%",           "type": "gyroid",      "infill": 40, "z": 1.125,
     "z_note": "z = 1.125 mm"},
    {"label": "(g)", "geom": "Cubic, 40%",            "type": "cubic",       "infill": 40,
     "z": 1.25, "z_note": "z = 1.25 mm"},
    {"label": "(h)", "geom": "Voronoi, 20%",          "type": "voronoi",     "infill": 20, "z": 0.0},
    {"label": "(i)", "geom": "Voronoi, 40%",          "type": "voronoi",     "infill": 40, "z": 0.0},
    {"label": "(j)", "geom": "Voronoi, 60%",          "type": "voronoi",     "infill": 60, "z": 0.0},
]


def generate_panel_mask(panel):
    """Return (mask, x, y) for a given panel specification."""
    t = panel["type"]
    inf = panel["infill"]
    p = PARAMS[t][inf]

    if t == "rectilinear":
        return make_mask_rectilinear(p["cell_size"])
    elif t == "honeycomb":
        return make_mask_honeycomb(p["d"])
    elif t == "gyroid":
        return make_mask_gyroid_slice(p["cell_size"], p["threshold"], panel["z"])
    elif t == "cubic":
        return make_mask_cubic(p["cell_size"], panel["z"])
    elif t == "voronoi":
        return make_mask_voronoi_slice(p["cell_size"], p["wall_thresh"], panel["z"])
    else:
        raise ValueError(f"Unknown geometry type: {t}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_figure():
    # NIM-A formatting
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'stix',
        'font.size': 9,
        'axes.labelsize': 9,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.top': True,
        'ytick.right': True,
        'xtick.major.size': 3,
        'ytick.major.size': 3,
    })

    fig_w_mm = 170.0
    fig_w_in = fig_w_mm / 25.4  # 6.693 in
    ncols, nrows = 5, 2
    panel_w_in = fig_w_in / ncols
    panel_h_in = panel_w_in  # square panels
    fig_h_in = nrows * panel_h_in + 0.35  # extra for annotation row

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w_in, fig_h_in),
                             gridspec_kw={'wspace': 0.05, 'hspace': 0.25})

    cmap = ListedColormap(['white', '#C8963C'])  # air, PLA

    for idx, panel in enumerate(PANELS):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        print(f"  Generating {panel['label']} {panel['geom']}...", flush=True)

        mask, x, y = generate_panel_mask(panel)
        extent = [x[0] - RES / 2, x[-1] + RES / 2,
                  y[0] - RES / 2, y[-1] + RES / 2]

        ax.imshow(mask.T, origin='lower', cmap=cmap, vmin=0, vmax=1,
                  extent=extent, aspect='equal', interpolation='nearest')

        # Panel label — bold, upper-left
        ax.text(0.04, 0.96, panel["label"], transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1))

        # Geometry annotation — italic, below panel
        ann = panel["geom"]
        if "z_note" in panel:
            ann += f"\n{panel['z_note']}"
        ax.set_xlabel(ann, fontsize=7.5, fontstyle='italic', labelpad=2)

        # Shared axis labels
        if col == 0:
            ax.set_ylabel(r'$y$ (mm)', fontsize=9)
        else:
            ax.set_yticklabels([])

        if row == nrows - 1:
            pass  # xlabel is the geometry annotation
        else:
            ax.set_xticklabels([])

        # Consistent view: full 20x20 mm sample
        ax.set_xlim(-HALF_W, HALF_W)
        ax.set_ylim(-HALF_W, HALF_W)
        ax.set_xticks([-8, -4, 0, 4, 8])
        ax.set_yticks([-8, -4, 0, 4, 8])

    # Add shared x-axis label
    fig.text(0.5, -0.01, r'$x$ (mm)', ha='center', fontsize=9)

    # Save
    fig.savefig('figure_1_geometry_cross_sections.pdf',
                bbox_inches='tight', pad_inches=0.02)
    fig.savefig('figure_1_geometry_cross_sections.png',
                dpi=300, bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)
    print("Saved: figure_1_geometry_cross_sections.pdf")
    print("Saved: figure_1_geometry_cross_sections.png")


if __name__ == "__main__":
    make_figure()
