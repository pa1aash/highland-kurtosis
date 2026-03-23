#!/usr/bin/env python3
"""
Phase 3 stacked-layer analysis: compare Geant4 N-scaling with ray-trace predictions.

For each N in {1, 2, 4, 10, 20}, loads ROOT file, applies standard cuts,
computes excess kurtosis with bootstrap SE, and extracts the geometric kurtosis:

    kappa_geo(N) = (kappa_total(N) - kappa_M) / (1 + kappa_M/3)

where kappa_M is the MCS baseline kurtosis from the solid control.
kappa_geo should follow 1/N scaling.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

# ── Paths ──────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "phase3_stacked"
SWEEP0_DIR = BASE / "data" / "sweep0"
RESULTS_DIR = BASE / "results"
PAPER_DIR = BASE / "paper"
SOLID_CONTROL = BASE / "data" / "proposal" / "control_solid_4GeV.root"

# ── Constants ──────────────────────────────────────────────────────────────
PLA_X0_MM = 315.0
BEAM_ENERGY = 4.0  # GeV
N_BOOTSTRAP = 1000
SEED = 42
N_VALUES = [1, 2, 4, 10, 20]

# Standard cuts
ENERGY_CUT_FRAC = 0.90
ANGLE_CUT_SIGMA = 10.0
FIDUCIAL_X = 5.0
FIDUCIAL_Y = 10.0


def highland_sigma_rad(x_over_X0, p_gev):
    """Highland projected-angle sigma [rad]."""
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))


def load_root_file(filepath):
    """Load ROOT file and return dict of numpy arrays."""
    if not HAS_UPROOT:
        print("ERROR: uproot not available", file=sys.stderr)
        sys.exit(1)
    f = uproot.open(str(filepath))
    tree = None
    for name in ["scattering", "ntuple", "tree", "T"]:
        if name in f:
            tree = f[name]
            break
    if tree is None:
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, 'arrays'):
                tree = obj
                break
    if tree is None:
        raise ValueError(f"No TTree found in {filepath}")
    return tree.arrays(library="np")


def apply_cuts(data):
    """Apply standard analysis cuts. Returns filtered dict."""
    n_before = len(data["theta_x"])
    mask = np.ones(n_before, dtype=bool)

    # Energy cut
    mask &= data["energy_out"] > ENERGY_CUT_FRAC * BEAM_ENERGY

    # Fiducial cuts
    mask &= (np.abs(data["entry_x"]) < FIDUCIAL_X) & (np.abs(data["entry_y"]) < FIDUCIAL_Y)

    # Angle cut: use Highland sigma for solid 10mm PLA
    sigma_h = highland_sigma_rad(10.0 / PLA_X0_MM, BEAM_ENERGY)
    angle_max = ANGLE_CUT_SIGMA * sigma_h
    mask &= (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)

    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == n_before else v
            for k, v in data.items()}, int(np.sum(mask))


def excess_kurtosis(arr):
    """Unbiased excess kurtosis (Fisher)."""
    return stats.kurtosis(arr, fisher=True, bias=False)


def kappa_bootstrap(theta, n_boot=N_BOOTSTRAP, seed=SEED):
    """Return (kappa, SE) from bootstrap resampling on theta_x."""
    rng = np.random.default_rng(seed)
    n = len(theta)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stats.kurtosis(theta[idx], fisher=True, bias=False)
    return float(np.mean(boots)), float(np.std(boots, ddof=1))


def load_raytrace_data():
    """Load ray-trace 1/N data for 20% infill stacked rectilinear."""
    with open(SWEEP0_DIR / "n_scaling_summary.json") as f:
        all_data = json.load(f)
    result = {}
    for entry in all_data:
        if (entry["geometry"] == "stacked_rectilinear"
                and entry["infill_target_pct"] == 20):
            result[entry["n_layers"]] = entry
    return result


def main():
    # ── 1. Get kappa_M from solid control ─────────────────────────────────
    print("=" * 70)
    print("Phase 3: Stacked-layer N-scaling analysis (geometric kurtosis)")
    print("=" * 70)

    if not SOLID_CONTROL.exists():
        print(f"ERROR: Solid control file not found: {SOLID_CONTROL}")
        sys.exit(1)

    solid_data = load_root_file(SOLID_CONTROL)
    solid_cut, solid_npass = apply_cuts(solid_data)
    kappa_M, kappa_M_se = kappa_bootstrap(solid_cut["theta_x"])
    print(f"  Solid control: n_pass={solid_npass}, "
          f"kappa_M={kappa_M:.4f} +/- {kappa_M_se:.4f}")

    # ── 2. Analyze Geant4 ROOT files ──────────────────────────────────────
    g4_results = {}
    for n in N_VALUES:
        fname = f"stacked_rect_20pct_{n}layer_4GeV.root"
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"WARNING: {fpath} not found, skipping N={n}")
            continue

        data = load_root_file(fpath)
        n_total = len(data["theta_x"])
        data_cut, n_pass = apply_cuts(data)
        theta_x = data_cut["theta_x"]

        kappa_total, kappa_total_se = kappa_bootstrap(theta_x)
        sigma_x = float(np.std(theta_x))

        # Extract geometric kurtosis:
        # kappa_geo = (kappa_total - kappa_M) / (1 + kappa_M/3)
        denom = 1.0 + kappa_M / 3.0
        kappa_geo = (kappa_total - kappa_M) / denom
        # Propagate errors (kappa_total and kappa_M independent)
        kappa_geo_se = np.sqrt(kappa_total_se**2 + kappa_M_se**2) / denom

        g4_results[n] = {
            "n_layers": n,
            "n_total": n_total,
            "n_pass": n_pass,
            "kappa_total": round(kappa_total, 4),
            "kappa_total_se": round(kappa_total_se, 4),
            "kappa_geo": round(kappa_geo, 4),
            "kappa_geo_se": round(kappa_geo_se, 4),
            "sigma_x_urad": round(sigma_x * 1e6, 2),
        }
        print(f"  N={n:2d}: n_pass={n_pass:6d}, "
              f"kappa_total={kappa_total:7.4f}, "
              f"kappa_geo={kappa_geo:7.4f} +/- {kappa_geo_se:.4f}, "
              f"sigma_x={sigma_x*1e6:.1f} urad")

    if not g4_results:
        print("ERROR: No ROOT files found!")
        sys.exit(1)

    # ── 3. Load ray-trace data ────────────────────────────────────────────
    rt_data = load_raytrace_data()

    # ── 4. Single-cell kappa_geo from N=1 ─────────────────────────────────
    kappa_geo_single = g4_results[1]["kappa_geo"]
    print(f"\nkappa_M (solid control) = {kappa_M:.4f}")
    print(f"kappa_geo_single (N=1)  = {kappa_geo_single:.4f}")

    # ── 5. Create two-panel figure ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Arrays for Geant4
    g4_ns = sorted(g4_results.keys())
    g4_kappas_geo = [g4_results[n]["kappa_geo"] for n in g4_ns]
    g4_errs_geo = [g4_results[n]["kappa_geo_se"] for n in g4_ns]

    # Arrays for ray-trace
    rt_ns = sorted(rt_data.keys())
    rt_kappas = [rt_data[n]["predicted_kurtosis"] for n in rt_ns]

    # Theory: kappa_geo_single / N
    n_theory = np.array(sorted(set(g4_ns + rt_ns)))
    theory_kappas = kappa_geo_single / n_theory

    # ── Left panel: log-log kappa_geo vs N ──
    ax1.errorbar(g4_ns, g4_kappas_geo, yerr=g4_errs_geo,
                 fmt='o', color='C0', markersize=7, capsize=3,
                 label=r'Geant4 $\kappa_{\mathrm{geo}}$', zorder=3)
    ax1.plot(rt_ns, rt_kappas, 'o', mfc='none', mec='C1', markersize=7,
             linestyle='--', color='C1', label='Ray-trace')
    ax1.plot(n_theory, theory_kappas, ':', color='gray', linewidth=1.5,
             label=r'$\kappa_{\mathrm{geo,1}}/N$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of layers $N$')
    ax1.set_ylabel(r'Geometric excess kurtosis $\kappa_{\mathrm{geo}}$')
    ax1.set_title(r'$\kappa_{\mathrm{geo}}$ vs $N$ (rectilinear 20\%, 4 GeV)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # ── Right panel: universality collapse ──
    # Geant4 collapse: kappa_geo * N / kappa_geo_single should → 1
    g4_collapse = [g4_results[n]["kappa_geo"] * n / kappa_geo_single
                   for n in g4_ns]
    ax2.plot(g4_ns, g4_collapse, 'o', color='C0', markersize=7,
             label=r'Geant4 $\kappa_{\mathrm{geo}}$')

    # Ray-trace collapse (use ray-trace's own N=1 kappa for self-consistency)
    rt_kappa_single = rt_data[1]["predicted_kurtosis"]
    rt_collapse = [rt_data[n]["predicted_kurtosis"] * n / rt_kappa_single
                   for n in rt_ns]
    ax2.plot(rt_ns, rt_collapse, 'o', mfc='none', mec='C1', markersize=7,
             label='Ray-trace')

    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of layers $N$')
    ax2.set_ylabel(r'$\kappa_{\mathrm{geo}} \cdot N / \kappa_{\mathrm{geo,1}}$')
    ax2.set_title('Universality collapse')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(0.0, 2.0)

    plt.tight_layout()
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(PAPER_DIR / "figure_6_n_scaling_with_geant4.png", dpi=200)
    fig.savefig(PAPER_DIR / "figure_6_n_scaling_with_geant4.pdf")
    plt.close()
    print(f"\nFigure saved to paper/figure_6_n_scaling_with_geant4.{{png,pdf}}")

    # ── 6. Print comparison table ─────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"{'N':>4s} | {'kappa_total':>12s} | {'kappa_geo':>12s} | "
          f"{'kappa_geo/N':>12s} | {'ray-trace':>12s} | {'geo*N/geo1':>11s}")
    print("-" * 100)

    all_ns = sorted(set(g4_ns + rt_ns))
    for n in all_ns:
        rt_k = f"{rt_data[n]['predicted_kurtosis']:.4f}" if n in rt_data else "—"
        if n in g4_results:
            kt = f"{g4_results[n]['kappa_total']:.4f}"
            kg = f"{g4_results[n]['kappa_geo']:.4f}"
            theory_k = f"{kappa_geo_single / n:.4f}"
            collapse = g4_results[n]["kappa_geo"] * n / kappa_geo_single
            collapse_s = f"{collapse:.4f}"
        else:
            kt = "—"
            kg = "—"
            theory_k = f"{kappa_geo_single / n:.4f}"
            collapse_s = "—"
        print(f"{n:4d} | {kt:>12s} | {kg:>12s} | "
              f"{theory_k:>12s} | {rt_k:>12s} | {collapse_s:>11s}")

    print("=" * 100)
    print(f"\nkappa_M = {kappa_M:.4f} (solid control baseline)")
    print(f"kappa_geo = (kappa_total - kappa_M) / (1 + kappa_M/3)")

    # ── 7. Save results to JSON ───────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Phase 3 stacked-layer N-scaling: geometric kurtosis extraction",
        "beam_energy_GeV": BEAM_ENERGY,
        "infill_pct": 20,
        "geometry": "stacked_rectilinear",
        "kappa_M": round(kappa_M, 4),
        "kappa_M_se": round(kappa_M_se, 4),
        "kappa_geo_single_N1": round(kappa_geo_single, 4),
        "n_bootstrap": N_BOOTSTRAP,
        "cuts": {
            "energy_fraction": ENERGY_CUT_FRAC,
            "angle_sigma": ANGLE_CUT_SIGMA,
            "fiducial_x_mm": FIDUCIAL_X,
            "fiducial_y_mm": FIDUCIAL_Y,
        },
        "geant4_results": [g4_results[n] for n in g4_ns],
        "raytrace_results": [
            {"n_layers": n,
             "kappa_analytic": rt_data[n]["kappa_analytic"],
             "predicted_kurtosis": rt_data[n]["predicted_kurtosis"]}
            for n in rt_ns
        ],
    }
    outpath = RESULTS_DIR / "phase3_stacked_results.json"
    with open(outpath, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {outpath}")


if __name__ == "__main__":
    main()
