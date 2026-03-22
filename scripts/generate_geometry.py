#!/usr/bin/env python3

import numpy as np
from pathlib import Path
import sys

OUTPUT_DIR = Path(__file__).parent.parent / "geometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_WIDTH = 20.0
SAMPLE_DEPTH = 10.0
WALL_THICK = 0.4
RESOLUTION = 0.1

def generate_gyroid_stl(period_mm, target_infill_pct, filename):
    try:
        from skimage.measure import marching_cubes
        from stl import mesh as stl_mesh
    except ImportError:
        print("ERROR: Install dependencies: pip install scikit-image numpy-stl")
        return None

    res = RESOLUTION
    nx = int(SAMPLE_WIDTH / res)
    ny = int(SAMPLE_WIDTH / res)
    nz = int(SAMPLE_DEPTH / res)

    print(f"  Gyroid: period={period_mm}mm, target={target_infill_pct}%, "
          f"grid={nx}x{ny}x{nz}")

    x = np.linspace(0, SAMPLE_WIDTH, nx)
    y = np.linspace(0, SAMPLE_WIDTH, ny)
    z = np.linspace(0, SAMPLE_DEPTH, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    k = 2 * np.pi / period_mm
    F = (np.sin(k*X) * np.cos(k*Y) +
         np.sin(k*Y) * np.cos(k*Z) +
         np.sin(k*Z) * np.cos(k*X))

    target_frac = target_infill_pct / 100.0
    lo, hi = 0.0, 1.5
    for _ in range(30):
        mid = (lo + hi) / 2.0
        frac = np.mean(np.abs(F) < mid)
        if frac < target_frac:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2.0
    actual_frac = np.mean(np.abs(F) < threshold)
    print(f"  Threshold={threshold:.4f}, actual infill={actual_frac*100:.1f}%")

    solid = (np.abs(F) < threshold).astype(np.float64)

    try:
        verts, faces, normals, _ = marching_cubes(solid, level=0.5,
                                                    spacing=(res, res, res))
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None

    gyroid_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            gyroid_mesh.vectors[i][j] = verts[f[j]]

    gyroid_mesh.save(str(filename))
    print(f"  Saved: {filename} ({faces.shape[0]} triangles)")
    return filename

def generate_voronoi_stl(mean_cell_size_mm, target_infill_pct, filename,
                          strut_diameter=None):
    try:
        from skimage.measure import marching_cubes
        from stl import mesh as stl_mesh
        from scipy.spatial import KDTree
    except ImportError:
        print("ERROR: Install deps: pip install scikit-image numpy-stl scipy")
        return None

    if strut_diameter is None:
        strut_diameter = WALL_THICK * 2

    cell_vol = mean_cell_size_mm ** 3
    sample_vol = SAMPLE_WIDTH * SAMPLE_WIDTH * SAMPLE_DEPTH
    n_seeds = max(4, int(sample_vol / cell_vol))

    print(f"  Voronoi: cell={mean_cell_size_mm}mm, seeds={n_seeds}, "
          f"target={target_infill_pct}%")

    rng = np.random.default_rng(42)
    seeds = np.column_stack([
        rng.uniform(0, SAMPLE_WIDTH, n_seeds),
        rng.uniform(0, SAMPLE_WIDTH, n_seeds),
        rng.uniform(0, SAMPLE_DEPTH, n_seeds),
    ])

    for iteration in range(10):
        tree = KDTree(seeds)
        n_probe = min(100000, n_seeds * 100)
        probes = np.column_stack([
            rng.uniform(0, SAMPLE_WIDTH, n_probe),
            rng.uniform(0, SAMPLE_WIDTH, n_probe),
            rng.uniform(0, SAMPLE_DEPTH, n_probe),
        ])
        _, labels = tree.query(probes)
        for s in range(n_seeds):
            mask = labels == s
            if np.any(mask):
                seeds[s] = probes[mask].mean(axis=0)

    res = RESOLUTION * 2
    nx = int(SAMPLE_WIDTH / res)
    ny = int(SAMPLE_WIDTH / res)
    nz = int(SAMPLE_DEPTH / res)

    print(f"  Voxel grid: {nx}x{ny}x{nz}, resolution={res}mm")

    x = np.linspace(0, SAMPLE_WIDTH, nx)
    y = np.linspace(0, SAMPLE_WIDTH, ny)
    z = np.linspace(0, SAMPLE_DEPTH, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    tree = KDTree(seeds)
    dists, _ = tree.query(points, k=2)
    d1 = dists[:, 0].reshape(nx, ny, nz)
    d2 = dists[:, 1].reshape(nx, ny, nz)
    boundary_dist = d2 - d1

    target_frac = target_infill_pct / 100.0
    lo, hi = 0.0, mean_cell_size_mm
    for _ in range(30):
        mid = (lo + hi) / 2.0
        frac = np.mean(boundary_dist < mid)
        if frac < target_frac:
            lo = mid
        else:
            hi = mid
    wall_thresh = (lo + hi) / 2.0
    actual_frac = np.mean(boundary_dist < wall_thresh)
    print(f"  Wall threshold={wall_thresh:.3f}mm, actual infill={actual_frac*100:.1f}%")

    solid = (boundary_dist < wall_thresh).astype(np.float64)

    try:
        verts, faces, normals, _ = marching_cubes(solid, level=0.5,
                                                    spacing=(res, res, res))
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None

    vor_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            vor_mesh.vectors[i][j] = verts[f[j]]

    vor_mesh.save(str(filename))
    print(f"  Saved: {filename} ({faces.shape[0]} triangles)")
    return filename

def generate_rectilinear_stl(cell_size_mm, wall_thick_mm, filename):
    try:
        from skimage.measure import marching_cubes
        from stl import mesh as stl_mesh
    except ImportError:
        print("ERROR: Install deps: pip install scikit-image numpy-stl")
        return None

    res = RESOLUTION
    nx = int(SAMPLE_WIDTH / res)
    ny = int(SAMPLE_WIDTH / res)
    nz = int(SAMPLE_DEPTH / res)

    x = np.linspace(0, SAMPLE_WIDTH, nx)
    y = np.linspace(0, SAMPLE_WIDTH, ny)
    z = np.linspace(0, SAMPLE_DEPTH, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    x_walls = np.abs(np.mod(X, cell_size_mm)) < wall_thick_mm / 2
    y_walls = np.abs(np.mod(Y, cell_size_mm)) < wall_thick_mm / 2
    solid = (x_walls | y_walls).astype(np.float64)

    actual_frac = np.mean(solid)
    print(f"  Rectilinear: cell={cell_size_mm}mm, wall={wall_thick_mm}mm, "
          f"infill={actual_frac*100:.1f}%")

    try:
        verts, faces, normals, _ = marching_cubes(solid, level=0.5,
                                                    spacing=(res, res, res))
    except Exception as e:
        print(f"  Marching cubes failed: {e}")
        return None

    grid_mesh = stl_mesh.Mesh(np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            grid_mesh.vectors[i][j] = verts[f[j]]

    grid_mesh.save(str(filename))
    print(f"  Saved: {filename} ({faces.shape[0]} triangles)")
    return filename

GYROID_PARAMS = {
    20: {"period": 7.0},
    40: {"period": 4.5},
    60: {"period": 3.0},
    80: {"period": 2.0},
}

VORONOI_PARAMS = {
    20: {"cell": 4.0},
    40: {"cell": 2.5},
    60: {"cell": 1.8},
    80: {"cell": 1.2},
}

def main():
    print("Lattice Geometry STL Generator")

    generated = []

    print("\nGyroid geometries")
    for infill, params in GYROID_PARAMS.items():
        fname = OUTPUT_DIR / f"gyroid_{infill}pct.stl"
        result = generate_gyroid_stl(params["period"], infill, fname)
        if result:
            generated.append(result)

    print("\nVoronoi (CVT) geometries")
    for infill, params in VORONOI_PARAMS.items():
        fname = OUTPUT_DIR / f"voronoi_{infill}pct.stl"
        result = generate_voronoi_stl(params["cell"], infill, fname)
        if result:
            generated.append(result)

    print("\nRectilinear geometries (for STL validation)")
    rect_cells = {20: 4.0, 40: 2.0, 60: 1.33, 80: 1.0}
    for infill, cell in rect_cells.items():
        fname = OUTPUT_DIR / f"rectilinear_{infill}pct.stl"
        result = generate_rectilinear_stl(cell, WALL_THICK, fname)
        if result:
            generated.append(result)

    print(f"\nGenerated {len(generated)} STL files in {OUTPUT_DIR}/")
    for g in generated:
        size_mb = g.stat().st_size / 1e6 if g.exists() else 0
        print(f"  {g.name}: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()

