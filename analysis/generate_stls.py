import numpy as np
from pathlib import Path
from stl import mesh as stl_mesh
from skimage.measure import marching_cubes

SAMPLE_W = 20.0
SAMPLE_T = 10.0
WALL_T = 0.4
HALF_WALL = WALL_T / 2.0

VOXEL_RES = 0.1

OUT_DIR = Path("stl_outputs")

RECT_PARAMS = {
    20: {"cell_size": 3.794},
    40: {"cell_size": 1.778},
    60: {"cell_size": 1.062},
}

HONEYCOMB_PARAMS = {
    20: {"d": 3.81409},
    40: {"d": 1.76914},
    60: {"d": 1.08829},
}

GYROID_PARAMS = {
    20: {"cell_size": 7.0, "threshold": 0.305398},
    40: {"cell_size": 4.5, "threshold": 0.61701},
    60: {"cell_size": 3.0, "threshold": 0.913538},
}

def make_grid():
    nx = int(SAMPLE_W / VOXEL_RES)
    ny = int(SAMPLE_W / VOXEL_RES)
    nz = int(SAMPLE_T / VOXEL_RES)
    x = np.linspace(-SAMPLE_W/2 + VOXEL_RES/2, SAMPLE_W/2 - VOXEL_RES/2, nx)
    y = np.linspace(-SAMPLE_W/2 + VOXEL_RES/2, SAMPLE_W/2 - VOXEL_RES/2, ny)
    z = np.linspace(-SAMPLE_T/2 + VOXEL_RES/2, SAMPLE_T/2 - VOXEL_RES/2, nz)
    return x, y, z, nx, ny, nz

def voxels_to_stl(volume, x, y, z, filename):
    infill_frac = np.mean(volume)

    padded = np.pad(volume.astype(float), pad_width=1, mode='constant',
                    constant_values=0.0)

    verts, faces, _, _ = marching_cubes(padded, level=0.5,
                                        spacing=(VOXEL_RES, VOXEL_RES, VOXEL_RES))

    verts[:, 0] += x[0] - VOXEL_RES
    verts[:, 1] += y[0] - VOXEL_RES
    verts[:, 2] += z[0] - VOXEL_RES

    stl_obj = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_obj.vectors[i][j] = verts[f[j]]

    stl_obj.save(str(filename))
    return infill_frac

def generate_solid(out_dir):
    x, y, z, nx, ny, nz = make_grid()
    volume = np.ones((nx, ny, nz), dtype=bool)
    fname = out_dir / "solid_100pct.stl"
    infill = voxels_to_stl(volume, x, y, z, fname)
    return fname.name, "solid", 100, infill

def generate_rectilinear(infill_pct, out_dir):
    cell_size = RECT_PARAMS[infill_pct]["cell_size"]
    x, y, z, nx, ny, nz = make_grid()

    halfW = SAMPLE_W / 2.0

    n_cells = max(1, int(SAMPLE_W / cell_size))
    x_walls = []
    for i in range(n_cells + 1):
        pos = -halfW + i * cell_size
        pos = max(-halfW + HALF_WALL, min(pos, halfW - HALF_WALL))
        x_walls.append(pos)
    y_walls = x_walls

    XX, YY = np.meshgrid(x, y, indexing='ij')
    mask_2d = np.zeros((nx, ny), dtype=bool)

    for xw in x_walls:
        mask_2d |= (np.abs(XX - xw) < HALF_WALL)
    for yw in y_walls:
        mask_2d |= (np.abs(YY - yw) < HALF_WALL)

    volume = np.repeat(mask_2d[:, :, np.newaxis], nz, axis=2)

    fname = out_dir / f"rectilinear_{infill_pct}pct.stl"
    infill = voxels_to_stl(volume, x, y, z, fname)
    return fname.name, "rectilinear", infill_pct, infill

def generate_honeycomb(infill_pct, out_dir):
    d = HONEYCOMB_PARAMS[infill_pct]["d"]
    a = d / np.sqrt(3.0)
    halfA = a / 2.0

    cos60, sin60 = 0.5, np.sqrt(3.0) / 2.0
    cos120, sin120 = -0.5, np.sqrt(3.0) / 2.0

    colSp = 1.5 * a
    rowSp = d

    x, y, z, nx, ny, nz = make_grid()
    XX, YY = np.meshgrid(x, y, indexing='ij')

    mask_2d = np.zeros((nx, ny), dtype=bool)

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

            mask_2d |= hit1 | hit2 | hit3

    in_sample = (np.abs(XX) <= SAMPLE_W/2) & (np.abs(YY) <= SAMPLE_W/2)
    mask_2d &= in_sample

    volume = np.repeat(mask_2d[:, :, np.newaxis], nz, axis=2)

    fname = out_dir / f"honeycomb_{infill_pct}pct.stl"
    infill = voxels_to_stl(volume, x, y, z, fname)
    return fname.name, "honeycomb", infill_pct, infill

def generate_gyroid(infill_pct, out_dir):
    cell_size = GYROID_PARAMS[infill_pct]["cell_size"]
    threshold = GYROID_PARAMS[infill_pct]["threshold"]

    x, y, z, nx, ny, nz = make_grid()
    k = 2.0 * np.pi / cell_size

    XX, YY, ZZ = np.meshgrid(x, y, z, indexing='ij')

    F = (np.sin(k * XX) * np.cos(k * YY) +
         np.sin(k * YY) * np.cos(k * ZZ) +
         np.sin(k * ZZ) * np.cos(k * XX))

    volume = np.abs(F) < threshold

    fname = out_dir / f"gyroid_{infill_pct}pct.stl"
    infill = voxels_to_stl(volume, x, y, z, fname)
    return fname.name, "gyroid", infill_pct, infill

def generate_rectilinear_slicer(out_dir):
    cell_size = RECT_PARAMS[40]["cell_size"]
    x, y, z, nx, ny, nz = make_grid()
    halfW = SAMPLE_W / 2.0
    layer_thick = 0.2

    n_cells = max(1, int(SAMPLE_W / cell_size))
    wall_positions_list = []
    for i in range(n_cells + 1):
        pos = -halfW + i * cell_size
        pos = max(-halfW + HALF_WALL, min(pos, halfW - HALF_WALL))
        wall_positions_list.append(pos)

    XX, YY = np.meshgrid(x, y, indexing='ij')

    xwall_2d = np.zeros((nx, ny), dtype=bool)
    for wp in wall_positions_list:
        xwall_2d |= (np.abs(XX - wp) < HALF_WALL)

    ywall_2d = np.zeros((nx, ny), dtype=bool)
    for wp in wall_positions_list:
        ywall_2d |= (np.abs(YY - wp) < HALF_WALL)

    volume = np.zeros((nx, ny, nz), dtype=bool)

    for iz in range(nz):
        z_center = z[iz] + SAMPLE_T / 2.0
        layer_idx = int(z_center / layer_thick)
        if layer_idx % 2 == 0:
            volume[:, :, iz] = xwall_2d
        else:
            volume[:, :, iz] = ywall_2d

    fname = out_dir / "rectilinear_slicer_40pct.stl"
    infill = voxels_to_stl(volume, x, y, z, fname)
    return fname.name, "rectilinear_slicer", 40, infill

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    print("generating solid control...")
    results.append(generate_solid(OUT_DIR))

    for pct in [20, 40, 60]:
        print(f"generating rectilinear {pct}%...")
        results.append(generate_rectilinear(pct, OUT_DIR))

    for pct in [20, 40, 60]:
        print(f"generating honeycomb {pct}%...")
        results.append(generate_honeycomb(pct, OUT_DIR))

    for pct in [20, 40, 60]:
        print(f"generating gyroid {pct}%...")
        results.append(generate_gyroid(pct, OUT_DIR))

    print("generating rectilinear slicer comparison (40%)...")
    results.append(generate_rectilinear_slicer(OUT_DIR))

    total_vol = SAMPLE_W * SAMPLE_W * SAMPLE_T
    print(f"\n{'filename':<35} {'geometry':<22} {'target':>6} {'actual':>7} {'volume_mm3':>10}")
    for fname, geom, target, infill in results:
        vol = infill * total_vol
        print(f"{fname:<35} {geom:<22} {target:>5}% {infill*100:>6.1f}% {vol:>10.1f}")

    print(f"\nall STL files saved to {OUT_DIR}/")
    print(f"total files: {len(results)}")

if __name__ == "__main__":
    main()

