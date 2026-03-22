import json
import numpy as np
from pathlib import Path

SAMPLE_W = 20.0
SAMPLE_T = 10.0
WALL_T = 0.4
BEAM_SIGMA = 5.0
X0_PLA = 315.0
HALF_W = SAMPLE_W / 2
HALF_T = SAMPLE_T / 2
HALF_WALL = WALL_T / 2
N_RAYS = 1_000_000
NZ_BINS = 200

PARAMS = {
    "rectilinear": {
        20: {"cell_size": 3.79},
        40: {"cell_size": 1.78},
        60: {"cell_size": 1.09},
        80: {"cell_size": 0.72},
    },
    "honeycomb": {
        20: {"d": 3.81409},
        40: {"d": 1.76914},
        60: {"d": 1.08829},
        80: {"d": 0.721858},
    },
    "gyroid": {
        20: {"cell_size": 7.0, "threshold": 0.305398},
        40: {"cell_size": 4.5, "threshold": 0.61701},
        60: {"cell_size": 3.0, "threshold": 0.913538},
        80: {"cell_size": 2.0, "threshold": 1.19732},
    },
    "cubic": {
        20: {"cell_size": 5.81482},
        40: {"cell_size": 2.5},
        60: {"cell_size": 1.52714},
        80: {"cell_size": 0.953214},
    },
    "voronoi": {
        20: {"cell_size": 4.0, "wall_thresh": 0.305441},
        40: {"cell_size": 2.5, "wall_thresh": 0.395338},
        60: {"cell_size": 1.8, "wall_thresh": 0.446486},
        80: {"cell_size": 1.2, "wall_thresh": 0.441343},
    },
}

def highland_s(t):
    x_X0 = t / X0_PLA
    mask = x_X0 > 0
    s = np.zeros_like(t)
    log_term = 1.0 + 0.038 * np.log(x_X0[mask])
    s[mask] = x_X0[mask] * log_term**2
    return s

def compute_kurtosis(paths):
    s = highland_s(paths)
    mean_s = np.mean(s)
    if mean_s < 1e-30:
        return 0.0
    var_s = np.var(s)
    return 3.0 * var_s / mean_s**2

def compute_stats(paths):
    mean_pla = np.mean(paths)
    std_pla = np.std(paths)
    infill = mean_pla / SAMPLE_T * 100
    x_X0 = paths / X0_PLA
    mean_x_X0 = np.mean(x_X0)
    var_x_X0 = np.var(x_X0)
    kappa = compute_kurtosis(paths)
    pcts = np.percentile(paths, [0, 5, 25, 50, 75, 95, 100])
    hit_frac = np.mean(paths > 0)
    return {
        "mean_PLA_path_mm": round(mean_pla, 4),
        "std_PLA_path_mm": round(std_pla, 4),
        "mean_x_over_X0": round(mean_x_X0, 6),
        "var_x_over_X0": round(var_x_X0, 8),
        "predicted_kurtosis": round(kappa, 4),
        "infill_actual_pct": round(infill, 1),
        "hit_fraction": round(hit_frac, 6),
        "path_length_percentiles": {
            "p0": round(pcts[0], 2), "p5": round(pcts[1], 2),
            "p25": round(pcts[2], 2), "p50": round(pcts[3], 2),
            "p75": round(pcts[4], 2), "p95": round(pcts[5], 2),
            "p100": round(pcts[6], 2),
        },
    }

def wall_positions(half_extent, cell_size):
    n_cells = max(1, int(half_extent * 2 / cell_size))
    walls = []
    for i in range(n_cells + 1):
        pos = -half_extent + i * cell_size
        pos = np.clip(pos, -half_extent + HALF_WALL, half_extent - HALF_WALL)
        walls.append(pos)
    return np.array(walls)

def raytrace_rectilinear(cell_size, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    xwalls = wall_positions(HALF_W, cell_size)
    ywalls = wall_positions(HALF_W, cell_size)

    in_xwall = np.zeros(n_rays, dtype=bool)
    for xw in xwalls:
        in_xwall |= np.abs(x - xw) < HALF_WALL
    in_ywall = np.zeros(n_rays, dtype=bool)
    for yw in ywalls:
        in_ywall |= np.abs(y - yw) < HALF_WALL

    paths = np.where(in_xwall | in_ywall, SAMPLE_T, 0.0)
    return paths

def raytrace_honeycomb(d, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    a = d / np.sqrt(3.0)
    col_sp = 1.5 * a
    row_sp = d
    half_a = a / 2.0
    sin60 = np.sqrt(3.0) / 2.0

    c_near = np.rint(x / col_sp).astype(int)
    r_near = np.rint(y / row_sp).astype(int)

    hit = np.zeros(n_rays, dtype=bool)
    for dc in range(-2, 3):
        c = c_near + dc
        cx = c * col_sp
        y_off = np.where(np.abs(c) % 2 == 1, row_sp * 0.5, 0.0)
        for dr in range(-2, 3):
            cy = (r_near + dr) * row_sp + y_off
            dx = x - cx
            dy = y - cy

            ey = dy - d / 2.0
            e1 = (np.abs(dx) < half_a) & (np.abs(ey) < HALF_WALL)

            dx2 = dx - 0.75 * a
            dy2 = dy - 0.25 * d
            lx2 = -0.5 * dx2 + sin60 * dy2
            ly2 = -sin60 * dx2 - 0.5 * dy2
            e2 = (np.abs(lx2) < half_a) & (np.abs(ly2) < HALF_WALL)

            dx3 = dx - 0.75 * a
            dy3 = dy + 0.25 * d
            lx3 = 0.5 * dx3 + sin60 * dy3
            ly3 = -sin60 * dx3 + 0.5 * dy3
            e3 = (np.abs(lx3) < half_a) & (np.abs(ly3) < HALF_WALL)

            hit |= e1 | e2 | e3

    paths = np.where(hit, SAMPLE_T, 0.0)
    return paths

def raytrace_gyroid(cell_size, threshold, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    k = 2.0 * np.pi / cell_size
    dz = SAMPLE_T / NZ_BINS
    paths = np.zeros(n_rays)

    for iz in range(NZ_BINS):
        z = -HALF_T + (iz + 0.5) * dz
        F = (np.sin(k * x) * np.cos(k * y) +
             np.sin(k * y) * np.cos(k * z) +
             np.sin(k * z) * np.cos(k * x))
        paths += np.where(np.abs(F) < threshold, dz, 0.0)

    return paths

def raytrace_cubic(cell_size, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    xwalls = wall_positions(HALF_W, cell_size)
    ywalls = wall_positions(HALF_W, cell_size)
    zwalls = wall_positions(HALF_T, cell_size)

    in_xwall = np.zeros(n_rays, dtype=bool)
    for xw in xwalls:
        in_xwall |= np.abs(x - xw) < HALF_WALL
    in_ywall = np.zeros(n_rays, dtype=bool)
    for yw in ywalls:
        in_ywall |= np.abs(y - yw) < HALF_WALL

    in_xy = in_xwall | in_ywall

    z_path = 0.0
    for zw in zwalls:
        z_lo = max(zw - HALF_WALL, -HALF_T)
        z_hi = min(zw + HALF_WALL, HALF_T)
        z_path += max(0.0, z_hi - z_lo)

    paths = np.where(in_xy, SAMPLE_T, z_path)
    return paths

def raytrace_voronoi(cell_size, wall_thresh, n_rays=N_RAYS):
    from scipy.spatial import KDTree
    rng = np.random.default_rng(42)

    cell_vol = cell_size**3
    samp_vol = SAMPLE_W * SAMPLE_W * SAMPLE_T
    n_seeds = max(4, int(samp_vol / cell_vol))

    seeds = np.column_stack([
        rng.uniform(-HALF_W, HALF_W, n_seeds),
        rng.uniform(-HALF_W, HALF_W, n_seeds),
        rng.uniform(-HALF_T, HALF_T, n_seeds),
    ])

    n_lloyd = 50000
    for _ in range(5):
        pts = np.column_stack([
            rng.uniform(-HALF_W, HALF_W, n_lloyd),
            rng.uniform(-HALF_W, HALF_W, n_lloyd),
            rng.uniform(-HALF_T, HALF_T, n_lloyd),
        ])
        tree = KDTree(seeds)
        _, closest = tree.query(pts)
        for s in range(n_seeds):
            mask = closest == s
            if np.any(mask):
                seeds[s] = pts[mask].mean(axis=0)

    tree = KDTree(seeds)

    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    dz = SAMPLE_T / NZ_BINS
    paths = np.zeros(n_rays)

    chunk = 50000
    for i0 in range(0, n_rays, chunk):
        i1 = min(i0 + chunk, n_rays)
        xc = x[i0:i1]
        yc = y[i0:i1]
        nc = i1 - i0
        chunk_paths = np.zeros(nc)

        for iz in range(NZ_BINS):
            z = -HALF_T + (iz + 0.5) * dz
            pts = np.column_stack([xc, yc, np.full(nc, z)])
            dd, _ = tree.query(pts, k=2)
            d1 = dd[:, 0]
            d2 = dd[:, 1]
            in_wall = (d2 - d1) < wall_thresh
            chunk_paths += np.where(in_wall, dz, 0.0)

        paths[i0:i1] = chunk_paths

    return paths

def raytrace_stacked_rectilinear(cell_size, n_layers, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    layer_thick = SAMPLE_T / n_layers
    paths = np.zeros(n_rays)

    for layer in range(n_layers):
        x_offset = rng.uniform(0, cell_size)
        y_offset = rng.uniform(0, cell_size)

        x_shifted = x + x_offset
        y_shifted = y + y_offset

        xwalls = wall_positions(HALF_W + cell_size, cell_size)
        ywalls = wall_positions(HALF_W + cell_size, cell_size)

        in_xwall = np.zeros(n_rays, dtype=bool)
        for xw in xwalls:
            in_xwall |= np.abs(x_shifted - xw) < HALF_WALL
        in_ywall = np.zeros(n_rays, dtype=bool)
        for yw in ywalls:
            in_ywall |= np.abs(y_shifted - yw) < HALF_WALL

        hit = in_xwall | in_ywall
        paths += np.where(hit, layer_thick, 0.0)

    return paths

def raytrace_gyroid_independent_periods(n_periods, threshold, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    lam = SAMPLE_T / n_periods
    k = 2.0 * np.pi / lam

    dx_shifts = rng.uniform(0, lam, n_periods)
    dy_shifts = rng.uniform(0, lam, n_periods)

    nz_total = max(NZ_BINS, n_periods * 20, int(SAMPLE_T / 0.01))
    dz = SAMPLE_T / nz_total
    paths = np.zeros(n_rays)

    for iz in range(nz_total):
        z_global = (iz + 0.5) * dz
        period_idx = min(int(z_global / lam), n_periods - 1)
        z_local = z_global - period_idx * lam

        x_shifted = x + dx_shifts[period_idx]
        y_shifted = y + dy_shifts[period_idx]

        F = (np.sin(k * x_shifted) * np.cos(k * y_shifted) +
             np.sin(k * y_shifted) * np.cos(k * z_local) +
             np.sin(k * z_local) * np.cos(k * x_shifted))
        paths += np.where(np.abs(F) < threshold, dz, 0.0)

    return paths

def raytrace_gyroid_stacked(n_slabs, threshold, n_rays=N_RAYS):
    rng = np.random.default_rng(42)
    x = rng.normal(0, BEAM_SIGMA, n_rays)
    y = rng.normal(0, BEAM_SIGMA, n_rays)

    lam = SAMPLE_T / n_slabs
    k = 2.0 * np.pi / lam
    nz_per_slab = max(20, int(lam / 0.01))
    dz = lam / nz_per_slab

    paths = np.zeros(n_rays)
    for slab in range(n_slabs):
        dx = rng.uniform(0, lam)
        dy = rng.uniform(0, lam)
        x_shifted = x + dx
        y_shifted = y + dy

        for iz in range(nz_per_slab):
            z_local = (iz + 0.5) * dz
            F = (np.sin(k * x_shifted) * np.cos(k * y_shifted) +
                 np.sin(k * y_shifted) * np.cos(k * z_local) +
                 np.sin(k * z_local) * np.cos(k * x_shifted))
            paths += np.where(np.abs(F) < threshold, dz, 0.0)

    return paths

def run_n_scaling_sweep():
    output_dir = Path("data/sweep0")
    output_dir.mkdir(parents=True, exist_ok=True)

    N_values = [1, 2, 3, 5, 10, 20, 50, 100]
    infills = [20, 40, 60]
    n_rays = 100_000

    results = []
    for infill in infills:
        cell_size = PARAMS["rectilinear"][infill]["cell_size"]
        single_paths = raytrace_rectilinear(cell_size, n_rays=n_rays)
        f_hit = np.mean(single_paths > 0)

        print(f"\n  stacked_rectilinear {infill}% (cell={cell_size}mm, f_hit={f_hit:.4f})")
        for N in N_values:
            print(f"    N={N:3d} ...", end="", flush=True)
            paths = raytrace_stacked_rectilinear(cell_size, N, n_rays=n_rays)
            stats = compute_stats(paths)
            kappa_analytic = 3.0 * (1.0 - f_hit) / (N * f_hit) if f_hit > 0 else 0.0

            entry = {
                "geometry": "stacked_rectilinear",
                "infill_target_pct": infill,
                "n_layers": N,
                "cell_size_mm": cell_size,
                "hit_fraction": round(f_hit, 6),
                "kappa_analytic": round(kappa_analytic, 6),
                "n_rays": n_rays,
            }
            entry.update(stats)
            results.append(entry)
            print(f" κ_ray={stats['predicted_kurtosis']:.4f}, κ_analytic={kappa_analytic:.4f}")

    with open(output_dir / "n_scaling_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'n_scaling_summary.json'}")
    return results

def run_gyroid_period_sweep():
    output_dir = Path("data/sweep0")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_infill = 40.0
    periods = [10.0, 5.0, 2.5, 1.25, 0.625]
    n_rays = 100_000

    results = []
    print(f"\n  gyroid DETERMINISTIC period sweep at {target_infill}% infill")
    for period in periods:
        lo, hi = 0.01, 2.0
        for _ in range(30):
            mid = (lo + hi) / 2.0
            paths = raytrace_gyroid(period, mid, n_rays=n_rays)
            actual = np.mean(paths) / SAMPLE_T * 100
            if actual < target_infill:
                lo = mid
            else:
                hi = mid
        threshold = (lo + hi) / 2.0
        paths = raytrace_gyroid(period, threshold, n_rays=n_rays)
        stats = compute_stats(paths)
        n_cells_z = SAMPLE_T / period

        entry = {
            "geometry": "gyroid_deterministic",
            "infill_target_pct": target_infill,
            "cell_size_mm": period,
            "gyroid_threshold": round(threshold, 6),
            "n_cells_z": round(n_cells_z, 2),
            "n_rays": n_rays,
        }
        entry.update(stats)
        results.append(entry)
        print(f"    λ={period:6.3f}mm (N_z={n_cells_z:5.1f}): "
              f"infill={stats['infill_actual_pct']:.1f}%, "
              f"κ={stats['predicted_kurtosis']:.4f}, thresh={threshold:.4f}")

    with open(output_dir / "gyroid_period_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'gyroid_period_summary.json'}")
    return results

def run_gyroid_independent_period_sweep():
    output_dir = Path("data/sweep0")
    output_dir.mkdir(parents=True, exist_ok=True)

    target_infill = 40.0
    N_values = [1, 2, 4, 8, 16]
    n_rays = 100_000

    print(f"\n  bisecting threshold for 40% infill gyroid ...")
    lo, hi = 0.01, 2.0
    for _ in range(30):
        mid = (lo + hi) / 2.0
        paths = raytrace_gyroid_independent_periods(1, mid, n_rays=n_rays)
        actual = np.mean(paths) / SAMPLE_T * 100
        if actual < target_infill:
            lo = mid
        else:
            hi = mid
    threshold = (lo + hi) / 2.0
    print(f"    threshold = {threshold:.6f}")

    paths_single = raytrace_gyroid_independent_periods(1, threshold, n_rays=n_rays)
    kappa_single = compute_kurtosis(paths_single)
    print(f"    κ_single = {kappa_single:.4f}")

    results_indep = []
    results_stacked = []

    print(f"\n  gyroid INDEPENDENT period sweep at {target_infill}% infill")
    for N in N_values:
        print(f"    N={N:2d} independent ...", end="", flush=True)
        paths = raytrace_gyroid_independent_periods(N, threshold, n_rays=n_rays)
        stats = compute_stats(paths)
        kappa_predicted = kappa_single / N

        entry = {
            "geometry": "gyroid_independent",
            "infill_target_pct": target_infill,
            "n_periods": N,
            "gyroid_threshold": round(threshold, 6),
            "kappa_single": round(kappa_single, 6),
            "kappa_predicted_1_over_N": round(kappa_predicted, 6),
            "n_rays": n_rays,
        }
        entry.update(stats)
        results_indep.append(entry)
        print(f" κ={stats['predicted_kurtosis']:.4f}, "
              f"κ_pred={kappa_predicted:.4f}, "
              f"infill={stats['infill_actual_pct']:.1f}%")

        print(f"    N={N:2d} stacked    ...", end="", flush=True)
        paths_s = raytrace_gyroid_stacked(N, threshold, n_rays=n_rays)
        stats_s = compute_stats(paths_s)

        entry_s = {
            "geometry": "gyroid_stacked",
            "infill_target_pct": target_infill,
            "n_periods": N,
            "gyroid_threshold": round(threshold, 6),
            "kappa_single": round(kappa_single, 6),
            "kappa_predicted_1_over_N": round(kappa_predicted, 6),
            "n_rays": n_rays,
        }
        entry_s.update(stats_s)
        results_stacked.append(entry_s)
        print(f" κ={stats_s['predicted_kurtosis']:.4f}, "
              f"infill={stats_s['infill_actual_pct']:.1f}%")

    all_results = {
        "threshold": round(threshold, 6),
        "kappa_single": round(kappa_single, 6),
        "independent": results_indep,
        "stacked": results_stacked,
    }
    with open(output_dir / "gyroid_independent_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {output_dir / 'gyroid_independent_summary.json'}")
    return all_results

def main():
    output_dir = Path("data/sweep0")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    results.append({
        "geometry": "solid",
        "infill_target_pct": 100,
        "infill_actual_pct": 100,
        "mean_x_over_X0": round(SAMPLE_T / X0_PLA, 6),
        "var_x_over_X0": 0,
        "predicted_kurtosis": 0.0,
    })

    N_3D = 200_000
    raytrace_funcs = {
        "rectilinear": lambda p, _: raytrace_rectilinear(p["cell_size"]),
        "honeycomb":   lambda p, _: raytrace_honeycomb(p["d"]),
        "gyroid":      lambda p, _: raytrace_gyroid(p["cell_size"], p["threshold"], N_3D),
        "cubic":       lambda p, _: raytrace_cubic(p["cell_size"]),
        "voronoi":     lambda p, _: raytrace_voronoi(p["cell_size"], p["wall_thresh"], N_3D),
    }

    for geom in ["rectilinear", "honeycomb", "gyroid", "cubic", "voronoi"]:
        for infill in [20, 40, 60, 80]:
            params = PARAMS[geom][infill]
            print(f"  Ray-tracing {geom} {infill}% ...", end="", flush=True)

            paths = raytrace_funcs[geom](params, infill)
            stats = compute_stats(paths)

            entry = {
                "geometry": geom,
                "infill_target_pct": infill,
                "beam_energy_GeV": 4.0,
                "n_rays": N_RAYS,
            }
            if "cell_size" in params:
                entry["cell_size_mm"] = params["cell_size"]
            if "d" in params:
                entry["cell_size_mm"] = params["d"]
            if "threshold" in params:
                entry["gyroid_threshold"] = params["threshold"]
            if "wall_thresh" in params:
                entry["voronoi_wall_thresh"] = params["wall_thresh"]

            entry.update(stats)
            results.append(entry)

            np.savez_compressed(
                output_dir / f"paths_{geom}_{infill}pct.npz",
                paths=paths.astype(np.float32))

            print(f" infill={stats['infill_actual_pct']:.1f}%, "
                  f"κ={stats['predicted_kurtosis']:.3f}")

    with open(output_dir / "sweep0_summary.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir / 'sweep0_summary.json'}")

    import sys
    if "--n-scaling" in sys.argv or "--all" in sys.argv:
        print("N-SCALING SWEEP: stacked rectilinear")
        run_n_scaling_sweep()

    if "--gyroid-period" in sys.argv or "--all" in sys.argv:
        print("GYROID DETERMINISTIC PERIOD SWEEP (control)")
        run_gyroid_period_sweep()

    if "--gyroid-independent" in sys.argv or "--all" in sys.argv:
        print("GYROID INDEPENDENT PERIOD SWEEP")
        run_gyroid_independent_period_sweep()

if __name__ == "__main__":
    main()

