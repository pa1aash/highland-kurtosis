import numpy as np
from pathlib import Path
import json
import argparse

SAMPLE_WIDTH = 20.0
SAMPLE_DEPTH = 10.0
WALL_THICK = 0.4
VOXEL_SIZE = 0.05

def highland_sigma(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    beta = 1.0
    return (13.6e-3 / (beta * p_gev)) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))

def compute_kurtosis_from_sigma_distribution(sigmas):
    sigma2 = sigmas ** 2
    mean_sigma2 = np.mean(sigma2)
    if mean_sigma2 == 0:
        return 0.0
    var_sigma2 = np.var(sigma2)
    kappa = 3.0 * var_sigma2 / (mean_sigma2 ** 2)
    return kappa

def make_rectilinear(infill_pct, cell_size_mm):
    nx = int(SAMPLE_WIDTH / VOXEL_SIZE)
    ny = int(SAMPLE_WIDTH / VOXEL_SIZE)
    nz = int(SAMPLE_DEPTH / VOXEL_SIZE)

    x = np.arange(nx) * VOXEL_SIZE
    y = np.arange(ny) * VOXEL_SIZE

    x_wall = np.abs(np.mod(x, cell_size_mm) - cell_size_mm/2) > (cell_size_mm - WALL_THICK)/2
    y_wall = np.abs(np.mod(y, cell_size_mm) - cell_size_mm/2) > (cell_size_mm - WALL_THICK)/2

    grid = np.zeros((nx, ny, nz), dtype=bool)
    for iz in range(nz):
        grid[:, :, iz] = x_wall[:, np.newaxis] | y_wall[np.newaxis, :]

    return grid

def make_honeycomb(infill_pct, cell_size_mm):
    nx = int(SAMPLE_WIDTH / VOXEL_SIZE)
    ny = int(SAMPLE_WIDTH / VOXEL_SIZE)
    nz = int(SAMPLE_DEPTH / VOXEL_SIZE)

    x = np.arange(nx) * VOXEL_SIZE
    y = np.arange(ny) * VOXEL_SIZE
    X, Y = np.meshgrid(x, y, indexing='ij')

    half_wall = WALL_THICK / 2.0
    grid_2d = np.zeros((nx, ny), dtype=bool)

    for angle_deg in [0, 60, 120]:
        angle = np.radians(angle_deg)
        proj = X * np.sin(angle) + Y * np.cos(angle)
        dist_to_wall = np.abs(np.mod(proj, cell_size_mm) - cell_size_mm/2)
        grid_2d |= (dist_to_wall > (cell_size_mm - WALL_THICK)/2)

    grid = np.repeat(grid_2d[:, :, np.newaxis], nz, axis=2)
    return grid

def make_gyroid(infill_pct, period_mm):
    nx = int(SAMPLE_WIDTH / VOXEL_SIZE)
    ny = int(SAMPLE_WIDTH / VOXEL_SIZE)
    nz = int(SAMPLE_DEPTH / VOXEL_SIZE)

    x = np.linspace(0, SAMPLE_WIDTH, nx)
    y = np.linspace(0, SAMPLE_WIDTH, ny)
    z = np.linspace(0, SAMPLE_DEPTH, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    k = 2 * np.pi / period_mm
    F = np.sin(k*X)*np.cos(k*Y) + np.sin(k*Y)*np.cos(k*Z) + np.sin(k*Z)*np.cos(k*X)

    target_frac = infill_pct / 100.0
    lo, hi = 0.0, 1.5
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if np.mean(np.abs(F) < mid) < target_frac:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2.0

    return np.abs(F) < threshold

def make_cubic(infill_pct, cell_size_mm):
    nx = int(SAMPLE_WIDTH / VOXEL_SIZE)
    ny = int(SAMPLE_WIDTH / VOXEL_SIZE)
    nz = int(SAMPLE_DEPTH / VOXEL_SIZE)

    x = np.arange(nx) * VOXEL_SIZE
    y = np.arange(ny) * VOXEL_SIZE
    z = np.arange(nz) * VOXEL_SIZE
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    Xr = X * np.cos(np.pi/4) - Y * np.sin(np.pi/4)
    Yr = X * np.sin(np.pi/4) + Y * np.cos(np.pi/4)

    half_wall = WALL_THICK / 2.0
    x_wall = np.abs(np.mod(Xr, cell_size_mm) - cell_size_mm/2) > (cell_size_mm - WALL_THICK)/2
    y_wall = np.abs(np.mod(Yr, cell_size_mm) - cell_size_mm/2) > (cell_size_mm - WALL_THICK)/2
    z_wall = np.abs(np.mod(Z,  cell_size_mm) - cell_size_mm/2) > (cell_size_mm - WALL_THICK)/2

    return x_wall | y_wall | z_wall

def make_voronoi(infill_pct, mean_cell_mm):
    from scipy.spatial import KDTree

    nx = int(SAMPLE_WIDTH / VOXEL_SIZE)
    ny = int(SAMPLE_WIDTH / VOXEL_SIZE)
    nz = int(SAMPLE_DEPTH / VOXEL_SIZE)

    actual_voxel = VOXEL_SIZE
    while nx * ny * nz > 5_000_000:
        actual_voxel *= 2
        nx = int(SAMPLE_WIDTH / actual_voxel)
        ny = int(SAMPLE_WIDTH / actual_voxel)
        nz = int(SAMPLE_DEPTH / actual_voxel)

    cell_vol = mean_cell_mm ** 3
    sample_vol = SAMPLE_WIDTH * SAMPLE_WIDTH * SAMPLE_DEPTH
    n_seeds = max(4, int(sample_vol / cell_vol))

    rng = np.random.default_rng(42)
    seeds = np.column_stack([
        rng.uniform(0, SAMPLE_WIDTH, n_seeds),
        rng.uniform(0, SAMPLE_WIDTH, n_seeds),
        rng.uniform(0, SAMPLE_DEPTH, n_seeds),
    ])

    for _ in range(10):
        tree = KDTree(seeds)
        probes = np.column_stack([
            rng.uniform(0, SAMPLE_WIDTH, 50000),
            rng.uniform(0, SAMPLE_WIDTH, 50000),
            rng.uniform(0, SAMPLE_DEPTH, 50000),
        ])
        _, labels = tree.query(probes)
        for s in range(n_seeds):
            mask = labels == s
            if np.any(mask):
                seeds[s] = probes[mask].mean(axis=0)

    x = np.linspace(0, SAMPLE_WIDTH, nx)
    y = np.linspace(0, SAMPLE_WIDTH, ny)
    z = np.linspace(0, SAMPLE_DEPTH, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    tree = KDTree(seeds)
    dists, _ = tree.query(points, k=2)
    boundary = (dists[:, 1] - dists[:, 0]).reshape(nx, ny, nz)

    target_frac = infill_pct / 100.0
    lo, hi = 0.0, mean_cell_mm
    for _ in range(30):
        mid = (lo + hi) / 2.0
        if np.mean(boundary < mid) < target_frac:
            lo = mid
        else:
            hi = mid
    thresh = (lo + hi) / 2.0

    return boundary < thresh

def trace_rays(voxel_grid, n_rays, voxel_size=VOXEL_SIZE):
    nx, ny, nz = voxel_grid.shape
    rng = np.random.default_rng(123)

    ix = rng.integers(0, nx, n_rays)
    iy = rng.integers(0, ny, n_rays)

    path_lengths = column_sums[ix, iy]

    return path_lengths

CONFIGS = {
    "rectilinear": {
        20: {"cell": 4.0},
        40: {"cell": 2.0},
        60: {"cell": 1.33},
        80: {"cell": 1.0},
    },
    "honeycomb": {
        20: {"cell": 5.0},
        40: {"cell": 3.0},
        60: {"cell": 2.0},
        80: {"cell": 1.3},
    },
    "gyroid": {
        20: {"cell": 7.0},
        40: {"cell": 4.5},
        60: {"cell": 3.0},
        80: {"cell": 2.0},
    },
    "cubic": {
        20: {"cell": 6.0},
        40: {"cell": 3.0},
        60: {"cell": 2.0},
        80: {"cell": 1.5},
    },
    "voronoi": {
        20: {"cell": 4.0},
        40: {"cell": 2.5},
        60: {"cell": 1.8},
        80: {"cell": 1.2},
    },
}

GEOMETRY_BUILDERS = {
    "rectilinear": lambda inf, cell: make_rectilinear(inf, cell),
    "honeycomb":   lambda inf, cell: make_honeycomb(inf, cell),
    "gyroid":      lambda inf, cell: make_gyroid(inf, cell),
    "cubic":       lambda inf, cell: make_cubic(inf, cell),
    "voronoi":     lambda inf, cell: make_voronoi(inf, cell),
}

def analyze_one_config(geom_name, infill, cell_size, n_rays, p_gev=4.0):
    print(f"\n--- {geom_name} {infill}% (cell={cell_size}mm) ---")

    builder = GEOMETRY_BUILDERS[geom_name]
    grid = builder(infill, cell_size)
    actual_infill = np.mean(grid) * 100
    print(f"  Actual infill: {actual_infill:.1f}%")

    actual_voxel = SAMPLE_DEPTH / grid.shape[2]

    paths = trace_rays(grid, n_rays, actual_voxel)

    x_over_X0 = paths / PLA_X0_MM

    sigmas = np.array([highland_sigma(t, p_gev) for t in x_over_X0])

    mean_path = np.mean(paths)
    std_path = np.std(paths)
    mean_xX0 = np.mean(x_over_X0)
    var_xX0 = np.var(x_over_X0)

    highland_pred = highland_sigma(mean_xX0, p_gev)

    kappa = compute_kurtosis_from_sigma_distribution(sigmas)

    rms_sigma = np.sqrt(np.mean(sigmas**2))

    result = {
        "geometry": geom_name,
        "infill_target_pct": infill,
        "infill_actual_pct": round(actual_infill, 1),
        "cell_size_mm": cell_size,
        "beam_energy_GeV": p_gev,
        "n_rays": n_rays,
        "mean_PLA_path_mm": round(mean_path, 4),
        "std_PLA_path_mm": round(std_path, 4),
        "mean_x_over_X0": round(mean_xX0, 6),
        "var_x_over_X0": round(var_xX0, 8),
        "highland_prediction_mrad": round(highland_pred * 1000, 4),
        "rms_sigma_mrad": round(rms_sigma * 1000, 4),
        "predicted_kurtosis": round(kappa, 4),
        "path_length_percentiles": {
            "p0": round(np.min(paths), 4),
            "p5": round(np.percentile(paths, 5), 4),
            "p25": round(np.percentile(paths, 25), 4),
            "p50": round(np.percentile(paths, 50), 4),
            "p75": round(np.percentile(paths, 75), 4),
            "p95": round(np.percentile(paths, 95), 4),
            "p100": round(np.max(paths), 4),
        },
    }

    print(f"  Mean path: {mean_path:.3f} mm (x/X0 = {mean_xX0:.5f})")
    print(f"  Path std:  {std_path:.3f} mm")
    print(f"  Highland prediction: {highland_pred*1e6:.1f} μrad")
    print(f"  RMS sigma:          {rms_sigma*1e6:.1f} μrad")
    print(f"  Predicted kurtosis: κ = {kappa:.4f}")

    return result, paths

def main():
    parser = argparse.ArgumentParser(description="Sweep 0: Ray-trace path-length distributions")
    parser.add_argument("--n-rays", type=int, default=1_000_000, help="Rays per config")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--energy", type=float, default=4.0, help="Beam energy (GeV)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent.parent / "data" / "sweep0"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    print("\nCONTROL: Solid PLA")
    solid_path = SAMPLE_DEPTH
    solid_xX0 = solid_path / PLA_X0_MM
    solid_sigma = highland_sigma(solid_xX0, args.energy)
    print(f"  x/X0 = {solid_xX0:.5f}")
    print(f"  Highland σ = {solid_sigma*1e6:.1f} μrad")
    all_results.append({
        "geometry": "solid",
        "infill_target_pct": 100,
        "infill_actual_pct": 100,
        "mean_x_over_X0": round(solid_xX0, 6),
        "var_x_over_X0": 0,
        "highland_prediction_mrad": round(solid_sigma * 1000, 4),
        "predicted_kurtosis": 0.0,
    })

    for geom_name, infill_configs in CONFIGS.items():
        for infill, params in infill_configs.items():
            try:
                result, paths = analyze_one_config(
                    geom_name, infill, params["cell"],
                    args.n_rays, args.energy)
                all_results.append(result)

                hist_file = output_dir / f"paths_{geom_name}_{infill}pct.npz"
                np.savez_compressed(str(hist_file), paths=paths)

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

    summary_file = output_dir / "sweep0_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

    print(f"{'Geometry':<15} {'Infill%':>8} {'⟨x/X₀⟩':>10} {'σ_Highland':>12} "
          f"{'σ_RMS':>10} {'κ':>8}")
    for r in all_results:
        geom = r["geometry"]
        infill = r["infill_target_pct"]
        xX0 = r.get("mean_x_over_X0", 0)
        highland = r.get("highland_prediction_mrad", 0)
        rms = r.get("rms_sigma_mrad", highland)
        kappa = r.get("predicted_kurtosis", 0)
        print(f"{geom:<15} {infill:>7}% {xX0:>10.5f} {highland:>10.3f} mrad "
              f"{rms:>8.3f} {kappa:>8.4f}")

if __name__ == "__main__":
    main()

