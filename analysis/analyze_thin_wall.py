#!/usr/bin/env python3
"""
Thin-wall Molière kurtosis analysis.

Analyses solid PLA slabs at 9 thicknesses (0.1–10 mm) × 3 energies (2/4/6 GeV)
to extract the thickness-dependent Molière kurtosis κ_M(ℓ/X₀, E) and fit a
logarithmic parametrisation.  Replaces the ±12% systematic band in the paper
with a first-principles thin-wall correction.

Usage:
    python analysis/analyze_thin_wall.py
    python analysis/analyze_thin_wall.py --data-dir data/phase01_thin_wall
"""

import numpy as np
from scipy import stats, optimize
from pathlib import Path
import json
import argparse
import re

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping plots")

try:
    import uproot
except ImportError:
    raise SystemExit("ERROR: uproot not available. Install: pip install uproot")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLA_X0_MM = 315.0
BOOT_N = 1000
SEED = 42

THICKNESSES_MM = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 5.0, 10.0]
ENERGIES_GEV = [2.0, 4.0, 6.0]

# Solid-slab κ_M from existing Geant4 proposal simulations (10 mm solid PLA)
KAPPA_M_SOLID_10MM = {2.0: 4.394, 4.0: 3.616, 6.0: 4.097}

BINARY_GEOS = {"rectilinear", "honeycomb"}

# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

def highland_sigma_rad(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))


def excess_kurtosis(data):
    return stats.kurtosis(data, fisher=True, bias=False)


def bootstrap_kurtosis_se(tx, ty, n_boot=BOOT_N, seed=SEED):
    """Bootstrap SE of the averaged (tx, ty) excess kurtosis."""
    rng = np.random.default_rng(seed)
    n = len(tx)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = (excess_kurtosis(tx[idx]) + excess_kurtosis(ty[idx])) / 2.0
    return np.std(boots, ddof=1)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def parse_filename(fname):
    """Extract (thickness_mm, energy_GeV) from e.g. thin_wall_solid_0p4mm_4GeV.root"""
    stem = Path(fname).stem
    m = re.search(r'(\d+(?:p\d+)?)mm_(\d+)GeV', stem)
    if not m:
        return None, None
    t_str = m.group(1).replace('p', '.')
    return float(t_str), float(m.group(2))


def load_and_cut(filepath, p_gev, thickness_mm):
    """Load ROOT file, apply standard quality cuts, return cut arrays."""
    f = uproot.open(str(filepath))
    d = f["scattering"].arrays(library="np")

    tx, ty = d["theta_x"], d["theta_y"]
    energy = d["energy_out"]
    ex, ey = d["entry_x"], d["entry_y"]

    # Highland sigma for the actual slab thickness (scales cut with slab)
    sigma_h = highland_sigma_rad(thickness_mm / PLA_X0_MM, p_gev)
    cut_rad = 10.0 * sigma_h

    mask = (energy > 0.9 * p_gev)
    mask &= (np.abs(ex) < 5.0) & (np.abs(ey) < 10.0)
    mask &= (np.abs(tx) < cut_rad) & (np.abs(ty) < cut_rad)

    n_before = len(tx)
    n_after = int(np.sum(mask))
    print(f"    Cuts: {n_before} -> {n_after} events "
          f"({100*n_after/n_before:.1f}% kept, angle_max={cut_rad*1e3:.2f} mrad)")

    return tx[mask], ty[mask]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def analyze_all(data_dir):
    """Analyse all 27 thin-wall ROOT files and return results list."""
    data_dir = Path(data_dir)
    root_files = sorted(data_dir.glob("*.root"))
    if not root_files:
        raise SystemExit(f"ERROR: No ROOT files found in {data_dir}")

    print(f"Found {len(root_files)} ROOT files in {data_dir}\n")

    results = []
    for rf in root_files:
        t_mm, e_gev = parse_filename(rf)
        if t_mm is None:
            print(f"  Skipping {rf.name} (could not parse filename)")
            continue

        print(f"  {rf.name}  (thickness={t_mm} mm, energy={e_gev} GeV)")
        tx, ty = load_and_cut(rf, e_gev, t_mm)

        if len(tx) < 100:
            print(f"    WARNING: Only {len(tx)} events after cuts — skipping")
            continue

        kappa_x = excess_kurtosis(tx)
        kappa_y = excess_kurtosis(ty)
        kappa_M = (kappa_x + kappa_y) / 2.0
        kappa_SE = bootstrap_kurtosis_se(tx, ty)

        ell_over_X0 = t_mm / PLA_X0_MM

        results.append({
            "thickness_mm": t_mm,
            "energy_GeV": e_gev,
            "ell_over_X0": round(ell_over_X0, 6),
            "kappa_M": round(kappa_M, 4),
            "kappa_M_SE": round(kappa_SE, 4),
            "kappa_x": round(kappa_x, 4),
            "kappa_y": round(kappa_y, 4),
            "n_events": len(tx),
        })

        print(f"    κ_M = {kappa_M:.4f} ± {kappa_SE:.4f}  "
              f"(κ_x={kappa_x:.4f}, κ_y={kappa_y:.4f}, N={len(tx)})")

    return results


# ---------------------------------------------------------------------------
# Parametrisation fit
# ---------------------------------------------------------------------------

def fit_log_parametrisation(results):
    """Fit κ_M = a + b·ln(ℓ/X₀) at each energy.  Return dict of fit params."""
    fits = {}
    for e in ENERGIES_GEV:
        subset = [r for r in results if r["energy_GeV"] == e]
        if len(subset) < 3:
            continue
        x = np.array([r["ell_over_X0"] for r in subset])
        y = np.array([r["kappa_M"] for r in subset])
        w = np.array([1.0 / max(r["kappa_M_SE"], 0.01) for r in subset])

        log_x = np.log(x)

        # Weighted least-squares: y = a + b*log_x
        def model(lx, a, b):
            return a + b * lx

        popt, pcov = optimize.curve_fit(model, log_x, y, sigma=1.0/w, absolute_sigma=True)
        a, b = popt
        perr = np.sqrt(np.diag(pcov))

        y_pred = model(log_x, a, b)
        residuals = y - y_pred
        chi2 = np.sum((residuals * w) ** 2)
        ndf = len(y) - 2

        fits[e] = {
            "a": round(a, 4),
            "b": round(b, 4),
            "a_err": round(perr[0], 4),
            "b_err": round(perr[1], 4),
            "chi2_ndf": round(chi2 / ndf, 3) if ndf > 0 else None,
            "residuals": [round(r, 4) for r in residuals],
            "thicknesses_mm": [s["thickness_mm"] for s in subset],
        }

        print(f"\n  Fit at {e:.0f} GeV: κ_M = {a:.4f} + {b:.4f}·ln(ℓ/X₀)")
        print(f"    Errors: a ± {perr[0]:.4f}, b ± {perr[1]:.4f}")
        print(f"    χ²/ndf = {chi2:.2f}/{ndf} = {chi2/ndf:.3f}" if ndf > 0 else "    (no dof)")
        print(f"    Residuals: {', '.join(f'{r:.4f}' for r in residuals)}")

    return fits


def kappa_M_from_fit(fits, ell_over_X0, energy_GeV):
    """Evaluate the fitted κ_M at given ℓ/X₀ and energy."""
    f = fits.get(energy_GeV)
    if f is None:
        return None
    return f["a"] + f["b"] * np.log(ell_over_X0)


# ---------------------------------------------------------------------------
# Figure: κ_M vs ℓ/X₀
# ---------------------------------------------------------------------------

def make_figure(results, fits, output_stem):
    if not HAS_MPL:
        print("  Skipping figure (matplotlib not available)")
        return

    fig, ax = plt.subplots(figsize=(8, 5.5))

    colors = {2.0: 'C0', 4.0: 'C1', 6.0: 'C2'}
    markers = {2.0: 's', 4.0: 'o', 6.0: '^'}

    for e in ENERGIES_GEV:
        subset = sorted([r for r in results if r["energy_GeV"] == e],
                        key=lambda r: r["ell_over_X0"])
        if not subset:
            continue

        x = np.array([r["ell_over_X0"] for r in subset])
        y = np.array([r["kappa_M"] for r in subset])
        yerr = np.array([r["kappa_M_SE"] for r in subset])

        ax.errorbar(x, y, yerr=yerr,
                    fmt=markers[e], color=colors[e], capsize=3, markersize=7,
                    label=f'{e:.0f} GeV data')

        # Overlay fit curve
        if e in fits:
            x_fine = np.geomspace(x.min() * 0.8, x.max() * 1.2, 200)
            y_fit = fits[e]["a"] + fits[e]["b"] * np.log(x_fine)
            ax.plot(x_fine, y_fit, '-', color=colors[e], alpha=0.6,
                    label=f'{e:.0f} GeV fit: $\\kappa_M = {fits[e]["a"]:.2f} '
                          f'{fits[e]["b"]:+.2f}\\,\\ln(\\ell/X_0)$')

    ax.set_xscale('log')
    ax.set_xlabel('$\\ell / X_0$', fontsize=13)
    ax.set_ylabel('Solid-slab Molière kurtosis $\\kappa_M$', fontsize=13)
    ax.set_title('Thin-wall Molière kurtosis vs material thickness', fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3, which='both')
    fig.tight_layout()

    for ext in ['png', 'pdf']:
        path = f"{output_stem}.{ext}"
        fig.savefig(path, dpi=200)
        print(f"  Figure saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Step 5: Thin-wall vs solid comparison
# ---------------------------------------------------------------------------

def print_thin_wall_vs_solid(results):
    print("\n" + "=" * 80)
    print("THIN-WALL vs SOLID COMPARISON: κ_M at 0.4 mm vs 10 mm")
    print("=" * 80)
    print(f"{'Energy':>8s}  {'κ_M(0.4mm)':>11s}  {'κ_M(10mm)':>10s}  "
          f"{'κ_M(10mm,G4)':>13s}  {'ratio 0.4/10':>13s}  {'Δκ_M':>8s}")
    print("-" * 80)

    for e in ENERGIES_GEV:
        r04 = [r for r in results if r["thickness_mm"] == 0.4 and r["energy_GeV"] == e]
        r10 = [r for r in results if r["thickness_mm"] == 10.0 and r["energy_GeV"] == e]

        k04 = r04[0]["kappa_M"] if r04 else float('nan')
        k04_se = r04[0]["kappa_M_SE"] if r04 else float('nan')
        k10 = r10[0]["kappa_M"] if r10 else float('nan')
        k10_ref = KAPPA_M_SOLID_10MM.get(e, float('nan'))

        ratio = k04 / k10 if k10 != 0 else float('nan')
        delta = k04 - k10

        print(f"{e:>7.0f}   {k04:>10.4f}±{k04_se:.3f}  {k10:>10.4f}  "
              f"{k10_ref:>13.3f}  {ratio:>13.3f}  {delta:>+8.4f}")

    print("\nThis ratio replaces the '±12% systematic band' in the paper.")
    print("The thin-wall κ_M at wall thickness (0.4 mm) should be used for")
    print("predicting binary-lattice kurtosis instead of the 10 mm solid value.\n")


# ---------------------------------------------------------------------------
# Step 6: Recompute Table 1 predictions
# ---------------------------------------------------------------------------

def recompute_table1_predictions(results):
    """Recompute predicted kurtosis for all binary lattice configs in Table 1
    using κ_M(thin-wall, 0.4 mm) instead of κ_M(solid, 10 mm)."""

    print("\n" + "=" * 80)
    print("RECOMPUTED TABLE 1 PREDICTIONS: κ_M(thin-wall) vs κ_M(solid)")
    print("  Universal equation (binary): κ = (3 + κ_M) / f  −  3")
    print("=" * 80)

    # Load ray-trace data for f_hit
    sweep0_path = Path(__file__).parent.parent / "data" / "sweep0" / "sweep0_summary.json"
    try:
        with open(sweep0_path) as fp:
            sweep0 = json.load(fp)
    except FileNotFoundError:
        print(f"  ERROR: {sweep0_path} not found — cannot recompute Table 1")
        return None

    rt_lookup = {}
    for entry in sweep0:
        rt_lookup[(entry["geometry"], entry.get("infill_target_pct", 100))] = entry

    # Extract thin-wall κ_M at 0.4 mm for each energy
    kM_thin = {}
    for e in ENERGIES_GEV:
        r04 = [r for r in results if r["thickness_mm"] == 0.4 and r["energy_GeV"] == e]
        if r04:
            kM_thin[e] = r04[0]["kappa_M"]
        else:
            # Fallback: use solid
            kM_thin[e] = KAPPA_M_SOLID_10MM.get(e, 3.6)

    print(f"\n  κ_M(thin-wall, 0.4 mm): "
          + ", ".join(f"{e:.0f} GeV: {kM_thin[e]:.4f}" for e in ENERGIES_GEV))
    print(f"  κ_M(solid, 10 mm):      "
          + ", ".join(f"{e:.0f} GeV: {KAPPA_M_SOLID_10MM[e]:.3f}" for e in ENERGIES_GEV)
          + "\n")

    # Table 1 binary configs: geometry, infill %, energies
    # At 4 GeV: all 5 geom × 4 infills (but only binary use (3+κ_M)/f−3)
    # Energy scan: rect_40% at 2,6 GeV; gyroid_40% at 2,6 GeV

    configs = []
    for geo in ["rectilinear", "honeycomb"]:
        for inf in [20, 40, 60, 80]:
            configs.append((geo, inf, 4.0))
    # Energy scan for binary
    for geo in ["rectilinear"]:
        for inf in [40]:
            for e in [2.0, 6.0]:
                configs.append((geo, inf, e))

    # Also include non-binary for completeness (these use ray-trace κ_geo)
    for geo in ["gyroid", "cubic", "voronoi"]:
        for inf in [20, 40, 60, 80]:
            configs.append((geo, inf, 4.0))
    for geo in ["gyroid"]:
        for inf in [40]:
            for e in [2.0, 6.0]:
                configs.append((geo, inf, e))

    print(f"{'Geometry':<14s} {'Infill':>6s} {'E(GeV)':>7s} {'f_hit':>6s} "
          f"{'κ_old':>7s} {'κ_new':>7s} {'Δ(new-old)':>10s} {'Type':>10s}")
    print("-" * 80)

    recomputed = []
    for geo, inf, e in configs:
        key = (geo, inf)
        if key not in rt_lookup:
            continue

        entry = rt_lookup[key]
        f_hit = entry.get("hit_fraction", 1.0)
        kgeo_ray = entry.get("predicted_kurtosis", 0.0)

        kM_old = KAPPA_M_SOLID_10MM.get(e, 3.6)
        kM_new = kM_thin.get(e, kM_old)

        is_binary = geo in BINARY_GEOS

        if is_binary:
            k_old = (3.0 + kM_old) / f_hit - 3.0
            k_new = (3.0 + kM_new) / f_hit - 3.0
            gtype = "binary"
        else:
            # For continuous/ternary: κ = κ_M + κ_geo*(1 + κ_M/3)
            k_old = kM_old + kgeo_ray * (1.0 + kM_old / 3.0)
            k_new = kM_new + kgeo_ray * (1.0 + kM_new / 3.0)
            gtype = "continuous"

        delta = k_new - k_old

        print(f"{geo:<14s} {inf:>5d}% {e:>7.0f} {f_hit:>6.3f} "
              f"{k_old:>7.2f} {k_new:>7.2f} {delta:>+10.3f} {gtype:>10s}")

        recomputed.append({
            "geometry": geo, "infill_pct": inf, "energy_GeV": e,
            "f_hit": round(f_hit, 4),
            "kappa_M_solid": round(kM_old, 4),
            "kappa_M_thin_wall": round(kM_new, 4),
            "kappa_pred_old": round(k_old, 4),
            "kappa_pred_new": round(k_new, 4),
            "delta_kappa": round(delta, 4),
            "geometry_type": gtype,
        })

    # Summary statistics for binary configs
    binary_rows = [r for r in recomputed if r["geometry_type"] == "binary"]
    if binary_rows:
        deltas = [r["delta_kappa"] for r in binary_rows]
        rel_changes = [r["delta_kappa"] / r["kappa_pred_old"] * 100
                       for r in binary_rows if r["kappa_pred_old"] != 0]
        print(f"\n  Binary lattice summary:")
        print(f"    Mean Δκ (new−old): {np.mean(deltas):+.3f}")
        print(f"    Mean relative change: {np.mean(rel_changes):+.1f}%")
        print(f"    Range: {np.min(rel_changes):+.1f}% to {np.max(rel_changes):+.1f}%")

    return recomputed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Thin-wall Moliere kurtosis analysis")
    parser.add_argument("--data-dir", type=str,
                        default="data/phase01_thin_wall",
                        help="Directory with thin-wall ROOT files")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    data_dir = base / args.data_dir

    # Step 1: Analyse all 27 ROOT files
    print("=" * 80)
    print("STEP 1: Analysing thin-wall ROOT files")
    print("=" * 80)
    results = analyze_all(data_dir)

    # Step 2: Save JSON results
    json_path = data_dir / "thin_wall_results.json"
    out = [{k: v for k, v in r.items()} for r in results]
    with open(json_path, 'w') as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nResults saved to {json_path}")

    # Step 3: Fit logarithmic parametrisation
    print("\n" + "=" * 80)
    print("STEP 3: Fitting κ_M = a + b·ln(ℓ/X₀) at each energy")
    print("=" * 80)
    fits = fit_log_parametrisation(results)

    # Save fits to JSON alongside results
    fits_path = data_dir / "thin_wall_fits.json"
    with open(fits_path, 'w') as f:
        json.dump(fits, f, indent=2, default=float)
    print(f"\nFits saved to {fits_path}")

    # Step 4: Create figure
    print("\n" + "=" * 80)
    print("STEP 4: Creating figure")
    print("=" * 80)
    fig_stem = str(base / "paper" / "figure_thin_wall_parametrisation")
    make_figure(results, fits, fig_stem)

    # Step 5: Thin-wall vs solid comparison
    print_thin_wall_vs_solid(results)

    # Step 6: Recompute Table 1 predictions
    recomputed = recompute_table1_predictions(results)

    # Save recomputed predictions
    if recomputed:
        recomp_path = data_dir / "thin_wall_recomputed_table1.json"
        with open(recomp_path, 'w') as f:
            json.dump(recomputed, f, indent=2, default=float)
        print(f"\nRecomputed predictions saved to {recomp_path}")

    # Summary table of all κ_M values
    print("\n" + "=" * 80)
    print("FULL κ_M RESULTS TABLE")
    print("=" * 80)
    print(f"{'Thickness':>10s} {'ℓ/X₀':>10s}  "
          f"{'κ_M(2GeV)':>10s} {'κ_M(4GeV)':>10s} {'κ_M(6GeV)':>10s}")
    print("-" * 60)

    for t in THICKNESSES_MM:
        row = f"{t:>9.1f}  {t/PLA_X0_MM:>10.6f}  "
        for e in ENERGIES_GEV:
            r = [x for x in results if x["thickness_mm"] == t and x["energy_GeV"] == e]
            if r:
                row += f"{r[0]['kappa_M']:>9.4f}±{r[0]['kappa_M_SE']:.3f}"
            else:
                row += f"{'—':>13s}"
            row += "  " if e < 6.0 else ""
        print(row)


if __name__ == "__main__":
    main()
