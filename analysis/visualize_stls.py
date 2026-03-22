import numpy as np
from pathlib import Path
from stl import mesh as stl_mesh

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STL_DIR = Path("stl_outputs")
OUT_DIR = Path("results/stl_verification")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_W = 20.0
SAMPLE_T = 10.0
VOXEL_RES = 0.1

def stl_to_voxels(stl_path):
    return None

def make_voxel_grid_rectilinear(cell_size):
    halfW = SAMPLE_W / 2.0
    nx = int(SAMPLE_W / VOXEL_RES)
    ny = int(SAMPLE_W / VOXEL_RES)
    nz = int(SAMPLE_T / VOXEL_RES)
    x = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, nx)
    y = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, ny)

    n_cells = max(1, int(SAMPLE_W / cell_size))
    walls = []
    for i in range(n_cells + 1):
        pos = -halfW + i * cell_size
        pos = max(-halfW + 0.2, min(pos, halfW - 0.2))
        walls.append(pos)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    mask = np.zeros((nx, ny), dtype=bool)
    for w in walls:
        mask |= (np.abs(XX - w) < 0.2)
        mask |= (np.abs(YY - w) < 0.2)
    return mask, x, y

def make_voxel_grid_honeycomb(d):
    halfW = SAMPLE_W / 2.0
    a = d / np.sqrt(3.0)
    halfA = a / 2.0
    cos60, sin60 = 0.5, np.sqrt(3.0) / 2.0
    cos120, sin120 = -0.5, np.sqrt(3.0) / 2.0
    colSp = 1.5 * a
    rowSp = d

    nx = int(SAMPLE_W / VOXEL_RES)
    ny = int(SAMPLE_W / VOXEL_RES)
    x = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, nx)
    y = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, ny)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    mask = np.zeros((nx, ny), dtype=bool)

    n_cols = int(SAMPLE_W / colSp) + 4
    n_rows = int(SAMPLE_W / rowSp) + 4
    for c in range(-n_cols, n_cols + 1):
        cx = c * colSp
        yOff = rowSp * 0.5 if (abs(c) % 2) else 0.0
        for r in range(-n_rows, n_rows + 1):
            cy = r * rowSp + yOff
            dx1 = XX - cx
            dy1 = YY - (cy + d * 0.5)
            hit1 = (np.abs(dx1) < halfA) & (np.abs(dy1) < 0.2)
            dx2 = XX - (cx + 0.75 * a)
            dy2 = YY - (cy + d * 0.25)
            lx2 = cos120 * dx2 + sin120 * dy2
            ly2 = -sin120 * dx2 + cos120 * dy2
            hit2 = (np.abs(lx2) < halfA) & (np.abs(ly2) < 0.2)
            dx3 = XX - (cx + 0.75 * a)
            dy3 = YY - (cy - d * 0.25)
            lx3 = cos60 * dx3 + sin60 * dy3
            ly3 = -sin60 * dx3 + cos60 * dy3
            hit3 = (np.abs(lx3) < halfA) & (np.abs(ly3) < 0.2)
            mask |= hit1 | hit2 | hit3

    in_sample = (np.abs(XX) <= halfW) & (np.abs(YY) <= halfW)
    mask &= in_sample
    return mask, x, y

def make_voxel_grid_gyroid(cell_size, threshold):
    halfW = SAMPLE_W / 2.0
    halfT = SAMPLE_T / 2.0
    nx = int(SAMPLE_W / VOXEL_RES)
    ny = int(SAMPLE_W / VOXEL_RES)
    nz = int(SAMPLE_T / VOXEL_RES)
    x = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, nx)
    y = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, ny)
    z = np.linspace(-halfT + VOXEL_RES/2, halfT - VOXEL_RES/2, nz)

    k = 2.0 * np.pi / cell_size
    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')
    F = (np.sin(k * XX) * np.cos(k * YY) +
         np.sin(k * YY) * np.cos(k * ZZ) +
         np.sin(k * ZZ) * np.cos(k * XX))
    volume = np.abs(F) < threshold
    return volume, x, y, z

def make_voxel_grid_slicer(cell_size):
    halfW = SAMPLE_W / 2.0
    halfT = SAMPLE_T / 2.0
    nx = int(SAMPLE_W / VOXEL_RES)
    ny = int(SAMPLE_W / VOXEL_RES)
    nz = int(SAMPLE_T / VOXEL_RES)
    x = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, nx)
    y = np.linspace(-halfW + VOXEL_RES/2, halfW - VOXEL_RES/2, ny)
    z = np.linspace(-halfT + VOXEL_RES/2, halfT - VOXEL_RES/2, nz)

    n_cells = max(1, int(SAMPLE_W / cell_size))
    walls = []
    for i in range(n_cells + 1):
        pos = -halfW + i * cell_size
        pos = max(-halfW + 0.2, min(pos, halfW - 0.2))
        walls.append(pos)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    xwall = np.zeros((nx, ny), dtype=bool)
    ywall = np.zeros((nx, ny), dtype=bool)
    for w in walls:
        xwall |= (np.abs(XX - w) < 0.2)
        ywall |= (np.abs(YY - w) < 0.2)

    volume = np.zeros((nx, ny, nz), dtype=bool)
    layer_thick = 0.2
    for iz in range(nz):
        z_center = z[iz] + halfT
        layer_idx = int(z_center / layer_thick)
        volume[:, :, iz] = xwall if (layer_idx % 2 == 0) else ywall

    return volume, x, y, z

RECT_PARAMS = {20: 3.794, 40: 1.778, 60: 1.062}
HC_PARAMS = {20: 3.81409, 40: 1.76914, 60: 1.08829}
GYR_PARAMS = {
    20: (7.0, 0.305398),
    40: (4.5, 0.61701),
    60: (3.0, 0.913538),
}

def plot_2d_geometry(ax, mask, x, y, title, zoom_extent=None):
    ax.imshow(mask.T, origin='lower', cmap='YlOrBr',
              extent=[x[0] - VOXEL_RES/2, x[-1] + VOXEL_RES/2,
                      y[0] - VOXEL_RES/2, y[-1] + VOXEL_RES/2],
              aspect='equal', interpolation='nearest')
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.set_xlabel('x (mm)', fontsize=8)
    ax.set_ylabel('y (mm)', fontsize=8)
    if zoom_extent:
        ax.set_xlim(zoom_extent[0], zoom_extent[1])
        ax.set_ylim(zoom_extent[2], zoom_extent[3])
    ax.tick_params(labelsize=7)

def main():
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))

    configs = [
        ("rectilinear 20%", "rect", 20),
        ("rectilinear 40%", "rect", 40),
        ("rectilinear 60%", "rect", 60),
        ("honeycomb 20%", "hc", 20),
        ("honeycomb 40%", "hc", 40),
        ("honeycomb 60%", "hc", 60),
        ("gyroid 20%", "gyr", 20),
        ("gyroid 40%", "gyr", 40),
        ("gyroid 60%", "gyr", 60),
        ("slicer rect 40% (even layer)", "slicer_even", 40),
        ("slicer rect 40% (odd layer)", "slicer_odd", 40),
        ("solid 100%", "solid", 100),
    ]

    for idx, (title, geom, pct) in enumerate(configs):
        row, col = divmod(idx, 4)
        ax = axes[row, col]

        if geom == "rect":
            mask, x, y = make_voxel_grid_rectilinear(RECT_PARAMS[pct])
            plot_2d_geometry(ax, mask, x, y, title)
        elif geom == "hc":
            mask, x, y = make_voxel_grid_honeycomb(HC_PARAMS[pct])
            plot_2d_geometry(ax, mask, x, y, title)
        elif geom == "gyr":
            vol, x, y, z = make_voxel_grid_gyroid(*GYR_PARAMS[pct])
            iz_mid = len(z) // 2
            plot_2d_geometry(ax, vol[:, :, iz_mid], x, y, title)
        elif geom == "slicer_even":
            vol, x, y, z = make_voxel_grid_slicer(RECT_PARAMS[40])
            plot_2d_geometry(ax, vol[:, :, 50], x, y, title)
        elif geom == "slicer_odd":
            vol, x, y, z = make_voxel_grid_slicer(RECT_PARAMS[40])
            plot_2d_geometry(ax, vol[:, :, 51], x, y, title)
        elif geom == "solid":
            nx = int(SAMPLE_W / VOXEL_RES)
            x = np.linspace(-10 + VOXEL_RES/2, 10 - VOXEL_RES/2, nx)
            mask = np.ones((nx, nx), dtype=bool)
            plot_2d_geometry(ax, mask, x, x, title)

    fig.suptitle('STL verification: XY cross-sections at z = 0 (mid-plane)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    outpath = OUT_DIR / "stl_xy_cross_sections.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {outpath}")

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    zoom = (-3, 3, -3, 3)

    zoom_configs = [
        ("rectilinear 40% (zoom)", "rect", 40),
        ("honeycomb 40% (zoom)", "hc", 40),
        ("gyroid 40% z=0 (zoom)", "gyr_z0", 40),
        ("gyroid 40% z=2.5 (zoom)", "gyr_z25", 40),
        ("slicer 40% even layer (zoom)", "slicer_even", 40),
        ("slicer 40% odd layer (zoom)", "slicer_odd", 40),
    ]

    for idx, (title, geom, pct) in enumerate(zoom_configs):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if geom == "rect":
            mask, x, y = make_voxel_grid_rectilinear(RECT_PARAMS[pct])
            plot_2d_geometry(ax, mask, x, y, title, zoom)
        elif geom == "hc":
            mask, x, y = make_voxel_grid_honeycomb(HC_PARAMS[pct])
            plot_2d_geometry(ax, mask, x, y, title, zoom)
        elif geom == "gyr_z0":
            vol, x, y, z = make_voxel_grid_gyroid(*GYR_PARAMS[pct])
            iz_mid = len(z) // 2
            plot_2d_geometry(ax, vol[:, :, iz_mid], x, y, title, zoom)
        elif geom == "gyr_z25":
            vol, x, y, z = make_voxel_grid_gyroid(*GYR_PARAMS[pct])
            iz = np.argmin(np.abs(z - 2.5))
            plot_2d_geometry(ax, vol[:, :, iz], x, y, title, zoom)
        elif geom == "slicer_even":
            vol, x, y, z = make_voxel_grid_slicer(RECT_PARAMS[40])
            plot_2d_geometry(ax, vol[:, :, 50], x, y, title, zoom)
        elif geom == "slicer_odd":
            vol, x, y, z = make_voxel_grid_slicer(RECT_PARAMS[40])
            plot_2d_geometry(ax, vol[:, :, 51], x, y, title, zoom)

    fig.suptitle('STL verification: zoomed XY cross-sections (central 6x6 mm)',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    outpath = OUT_DIR / "stl_xy_zoomed.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {outpath}")

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))

    xz_configs = [
        ("rectilinear 40% XZ", "rect", 40),
        ("honeycomb 40% XZ", "hc", 40),
        ("gyroid 40% XZ", "gyr", 40),
        ("slicer 40% XZ", "slicer", 40),
        ("gyroid 20% XZ", "gyr", 20),
        ("gyroid 60% XZ", "gyr", 60),
    ]

    for idx, (title, geom, pct) in enumerate(xz_configs):
        row, col = divmod(idx, 3)
        ax = axes[row, col]

        if geom == "rect":
            mask_2d, x, y = make_voxel_grid_rectilinear(RECT_PARAMS[pct])
            iy_mid = len(y) // 2
            nz = int(SAMPLE_T / VOXEL_RES)
            z = np.linspace(-SAMPLE_T/2 + VOXEL_RES/2, SAMPLE_T/2 - VOXEL_RES/2, nz)
            xz_slice = np.repeat(mask_2d[:, iy_mid:iy_mid+1], nz, axis=1)
            ax.imshow(xz_slice.T, origin='lower', cmap='YlOrBr',
                      extent=[x[0], x[-1], z[0], z[-1]], aspect='auto',
                      interpolation='nearest')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('x (mm)', fontsize=8)
            ax.set_ylabel('z (mm)', fontsize=8)
        elif geom == "hc":
            mask_2d, x, y = make_voxel_grid_honeycomb(HC_PARAMS[pct])
            iy_mid = len(y) // 2
            nz = int(SAMPLE_T / VOXEL_RES)
            z = np.linspace(-SAMPLE_T/2 + VOXEL_RES/2, SAMPLE_T/2 - VOXEL_RES/2, nz)
            xz_slice = np.repeat(mask_2d[:, iy_mid:iy_mid+1], nz, axis=1)
            ax.imshow(xz_slice.T, origin='lower', cmap='YlOrBr',
                      extent=[x[0], x[-1], z[0], z[-1]], aspect='auto',
                      interpolation='nearest')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('x (mm)', fontsize=8)
            ax.set_ylabel('z (mm)', fontsize=8)
        elif geom == "gyr":
            vol, x, y, z = make_voxel_grid_gyroid(*GYR_PARAMS[pct])
            iy_mid = len(y) // 2
            xz_slice = vol[:, iy_mid, :]
            ax.imshow(xz_slice.T, origin='lower', cmap='YlOrBr',
                      extent=[x[0], x[-1], z[0], z[-1]], aspect='auto',
                      interpolation='nearest')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('x (mm)', fontsize=8)
            ax.set_ylabel('z (mm)', fontsize=8)
        elif geom == "slicer":
            vol, x, y, z = make_voxel_grid_slicer(RECT_PARAMS[40])
            iy_mid = len(y) // 2
            xz_slice = vol[:, iy_mid, :]
            ax.imshow(xz_slice.T, origin='lower', cmap='YlOrBr',
                      extent=[x[0], x[-1], z[0], z[-1]], aspect='auto',
                      interpolation='nearest')
            ax.set_title(title, fontsize=10, fontweight='bold')
            ax.set_xlabel('x (mm)', fontsize=8)
            ax.set_ylabel('z (mm)', fontsize=8)

        ax.tick_params(labelsize=7)

    fig.suptitle('STL verification: XZ cross-sections at y = 0',
                 fontsize=14, fontweight='bold')
    fig.tight_layout()
    outpath = OUT_DIR / "stl_xz_cross_sections.png"
    fig.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {outpath}")

    print(f"\nall verification plots saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()

