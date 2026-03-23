#!/usr/bin/env python3
"""
Phase 3 stacked-layer analysis: compare Geant4 N-scaling with ray-trace predictions.

For each N in {1, 2, 4, 10, 20}, loads ROOT file, applies standard cuts,
computes excess kurtosis with bootstrap SE, and compares to ray-trace 1/N scaling.
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
    # ── 1. Analyze Geant4 ROOT files ──────────────────────────────────────
    print("=" * 70)
    print("Phase 3: Stacked-layer N-scaling analysis")
    print("=" * 70)

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

        kappa, kappa_se = kappa_bootstrap(theta_x)
        sigma_x = float(np.std(theta_x))

        g4_results[n] = {
            "n_layers": n,
            "n_total": n_total,
            "n_pass": n_pass,
            "kappa": round(kappa, 4),
            "kappa_se": round(kappa_se, 4),
            "sigma_x_urad": round(sigma_x * 1e6, 2),
        }
        print(f"  N={n:2d}: n_pass={n_pass:6d}, "
              f"kappa={kappa:7.3f} +/- {kappa_se:.3f}, "
              f"sigma_x={sigma_x*1e6:.1f} urad")

    if not g4_results:
        print("ERROR: No ROOT files found!")
        sys.exit(1)

    # ── 2. Load ray-trace data ────────────────────────────────────────────
    rt_data = load_raytrace_data()

    # ── 3. Single-cell kappa from N=1 Geant4 ─────────────────────────────
    kappa_single = g4_results[1]["kappa"]
    print(f"\nkappa_single (N=1 Geant4) = {kappa_single:.4f}")

    # ── 4. Create two-panel figure ────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Arrays for Geant4
    g4_ns = sorted(g4_results.keys())
    g4_kappas = [g4_results[n]["kappa"] for n in g4_ns]
    g4_errs = [g4_results[n]["kappa_se"] for n in g4_ns]

    # Arrays for ray-trace (only N values that exist in both)
    rt_ns = sorted(rt_data.keys())
    rt_kappas = [rt_data[n]["predicted_kurtosis"] for n in rt_ns]

    # Theory: kappa_single / N
    n_theory = np.array(sorted(set(g4_ns + rt_ns)))
    theory_kappas = kappa_single / n_theory

    # ── Left panel: log-log kappa vs N ──
    ax1.errorbar(g4_ns, g4_kappas, yerr=g4_errs,
                 fmt='o', color='C0', markersize=7, capsize=3,
                 label='Geant4', zorder=3)
    ax1.plot(rt_ns, rt_kappas, 'o', mfc='none', mec='C1', markersize=7,
             linestyle='--', color='C1', label='Ray-trace')
    ax1.plot(n_theory, theory_kappas, ':', color='gray', linewidth=1.5,
             label=r'$\kappa_{\mathrm{single}}/N$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of layers $N$')
    ax1.set_ylabel(r'Excess kurtosis $\kappa$')
    ax1.set_title(r'$\kappa$ vs $N$ (rectilinear 20\%, 4 GeV)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, which='both')

    # ── Right panel: universality collapse ──
    # Geant4 collapse
    g4_collapse = [g4_results[n]["kappa"] * n / kappa_single for n in g4_ns]
    ax2.plot(g4_ns, g4_collapse, 'o', color='C0', markersize=7,
             label='Geant4')

    # Ray-trace collapse (use ray-trace's own N=1 kappa for self-consistency)
    rt_kappa_single = rt_data[1]["predicted_kurtosis"]
    rt_collapse = [rt_data[n]["predicted_kurtosis"] * n / rt_kappa_single
                   for n in rt_ns]
    ax2.plot(rt_ns, rt_collapse, 'o', mfc='none', mec='C1', markersize=7,
             label='Ray-trace')

    ax2.axhline(1.0, color='gray', linestyle=':', linewidth=1.5)
    ax2.set_xscale('log')
    ax2.set_xlabel('Number of layers $N$')
    ax2.set_ylabel(r'$\kappa \cdot N / \kappa_{\mathrm{single}}$')
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

    # ── 5. Print comparison table ─────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'N':>4s} | {'kappa (ray-trace)':>18s} | {'kappa (Geant4)':>15s} | "
          f"{'kappa_single/N':>15s} | {'G4/theory':>10s}")
    print("-" * 90)

    all_ns = sorted(set(g4_ns + rt_ns))
    for n in all_ns:
        rt_k = f"{rt_data[n]['predicted_kurtosis']:.4f}" if n in rt_data else "—"
        g4_k = f"{g4_results[n]['kappa']:.4f}" if n in g4_results else "—"
        theory_k = f"{kappa_single / n:.4f}"
        if n in g4_results:
            ratio = g4_results[n]["kappa"] / (kappa_single / n)
            ratio_s = f"{ratio:.4f}"
        else:
            ratio_s = "—"
        print(f"{n:4d} | {rt_k:>18s} | {g4_k:>15s} | {theory_k:>15s} | {ratio_s:>10s}")

    print("=" * 90)

    # ── 6. Save results to JSON ───────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Phase 3 stacked-layer N-scaling: Geant4 vs ray-trace",
        "beam_energy_GeV": BEAM_ENERGY,
        "infill_pct": 20,
        "geometry": "stacked_rectilinear",
        "kappa_single_N1": kappa_single,
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
