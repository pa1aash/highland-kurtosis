#!/usr/bin/env python3
"""
Model comparison analysis: compare Geant4 EM physics options 0, 3, 4.

Key finding: kappa_geo = (kappa_total - kappa_M) / (1 + kappa_M/3)
should be the SAME across all three options (geometric, not physics-model
dependent), while kappa_M may differ.
"""

import numpy as np
from scipy import stats
from pathlib import Path
import json
import sys

try:
    import uproot
except ImportError:
    print("ERROR: uproot required. Install with: pip install uproot")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib required.")
    sys.exit(1)

# ── Constants ──────────────────────────────────────────────────────────────
PLA_X0_MM = 315.0
BEAM_ENERGY = 4.0  # GeV
SAMPLE_THICK = 10.0  # mm
N_BOOT = 1000
SEED = 42

OPTION_NAMES = {0: "Urban (opt0)", 3: "EmStandard_opt3", 4: "EmStandard_opt4 (EMZ)"}
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "phase04_model_comparison"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
PAPER_DIR = Path(__file__).resolve().parent.parent / "paper"


# ── Physics helpers ────────────────────────────────────────────────────────
def highland_sigma_rad(x_over_X0, p_gev):
    """Highland projected-angle sigma [rad]."""
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))


def excess_kurtosis(data):
    """Unbiased excess kurtosis (Fisher definition)."""
    return float(stats.kurtosis(data, fisher=True, bias=False))


def kappa_bootstrap(theta_x, theta_y, n_boot=N_BOOT, seed=SEED):
    """Return (kappa, SE) via bootstrap of averaged (tx, ty) kurtosis."""
    rng = np.random.default_rng(seed)
    n = len(theta_x)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = (excess_kurtosis(theta_x[idx]) + excess_kurtosis(theta_y[idx])) / 2.0
    kappa = (excess_kurtosis(theta_x) + excess_kurtosis(theta_y)) / 2.0
    return float(kappa), float(np.std(boots, ddof=1))


# ── I/O helpers ────────────────────────────────────────────────────────────
def load_root_file(filepath):
    """Load ROOT ntuple as dict of numpy arrays."""
    f = uproot.open(str(filepath))
    tree = None
    for name in ["scattering", "ntuple", "tree", "T"]:
        if name in f:
            tree = f[name]
            break
    if tree is None:
        for key in f.keys():
            obj = f[key]
            if hasattr(obj, "arrays"):
                tree = obj
                break
    if tree is None:
        raise ValueError(f"No TTree found in {filepath}")
    return tree.arrays(library="np")


def apply_cuts(data, beam_energy_gev=BEAM_ENERGY):
    """Apply standard cuts: energy, fiducial, angle."""
    n_before = len(data["theta_x"])
    mask = np.ones(n_before, dtype=bool)

    # Energy cut
    mask &= data["energy_out"] > 0.9 * beam_energy_gev

    # Fiducial cuts
    mask &= (np.abs(data["entry_x"]) < 5.0) & (np.abs(data["entry_y"]) < 10.0)

    # Angle cut: 10 * sigma_Highland(10 mm solid PLA)
    sigma_h = highland_sigma_rad(SAMPLE_THICK / PLA_X0_MM, beam_energy_gev)
    angle_max = 10.0 * sigma_h
    mask &= (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)

    n_after = int(np.sum(mask))
    print(f"    Cuts: {n_before} -> {n_after} events ({100*n_after/n_before:.1f}% kept)")

    return {k: v[mask] if isinstance(v, np.ndarray) and len(v) == n_before else v
            for k, v in data.items()}


# ── Main analysis ──────────────────────────────────────────────────────────
def analyze_option(opt):
    """Analyze one physics option, returning dict of results."""
    solid_path = DATA_DIR / f"model_opt{opt}_solid_4GeV.root"
    rect_path = DATA_DIR / f"model_opt{opt}_rect40_4GeV.root"

    results = {"option": opt, "name": OPTION_NAMES[opt]}

    for tag, path in [("solid", solid_path), ("rect40", rect_path)]:
        print(f"  Loading {tag}: {path.name}")
        data = load_root_file(path)
        data = apply_cuts(data)

        tx, ty = data["theta_x"], data["theta_y"]
        n = len(tx)

        kappa, se = kappa_bootstrap(tx, ty)
        sigma_x = float(np.std(tx, ddof=1))

        # Diagnostic: mean PLA path length
        mean_pla = float(np.mean(data["pla_path"])) if "pla_path" in data else float("nan")
        print(f"    n={n}, kappa={kappa:.2f}±{se:.2f}, "
              f"sigma_x={sigma_x*1e3:.3f} mrad, <pla_path>={mean_pla:.2f} mm")

        results[f"{tag}_n"] = n
        results[f"{tag}_kappa"] = round(kappa, 4)
        results[f"{tag}_se"] = round(se, 4)
        results[f"{tag}_sigma_x_mrad"] = round(sigma_x * 1e3, 4)
        results[f"{tag}_mean_pla_mm"] = round(mean_pla, 3)

    # Derived quantities
    results["kappa_M"] = results["solid_kappa"]
    results["kappa_M_se"] = results["solid_se"]
    results["kappa_total"] = results["rect40_kappa"]
    results["kappa_total_se"] = results["rect40_se"]
    # Correct extraction from the universal equation (Eq. 19):
    #   kappa_total = (3 + kappa_M) * (3 + kappa_geo) / 3 - 3
    # Solving for kappa_geo:
    #   kappa_geo = (kappa_total - kappa_M) / (1 + kappa_M / 3)
    # The simple subtraction kappa_total - kappa_M ignores the cross-term
    # and overestimates kappa_geo when kappa_M is non-negligible.
    kM = results["kappa_M"]
    kT = results["kappa_total"]
    results["kappa_geo"] = round((kT - kM) / (1.0 + kM / 3.0), 4)
    # Propagated SE for kappa_geo (assuming independence)
    # d(kappa_geo)/d(kT) = 1 / (1 + kM/3)
    # d(kappa_geo)/d(kM) = -(kT + 3) / (3 + kM)^2 * 3  (simplified)
    denom = 1.0 + kM / 3.0
    dg_dkT = 1.0 / denom
    dg_dkM = -(kT + 3.0) / (kM + 3.0)**2
    results["kappa_geo_se"] = round(
        np.sqrt((dg_dkT * results["kappa_total_se"])**2 +
                (dg_dkM * results["kappa_M_se"])**2), 4
    )

    return results


def make_figure(all_results):
    """Grouped bar chart: kappa_M and kappa_geo for each physics option."""
    opts = [r["option"] for r in all_results]
    names = [r["name"] for r in all_results]
    kappa_M = [r["kappa_M"] for r in all_results]
    kappa_M_err = [r["kappa_M_se"] for r in all_results]
    kappa_geo = [r["kappa_geo"] for r in all_results]
    kappa_geo_err = [r["kappa_geo_se"] for r in all_results]

    x = np.arange(len(opts))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, kappa_M, width, yerr=kappa_M_err,
                   label=r"$\kappa_M$ (solid)", color="#4C72B0", capsize=4)
    bars2 = ax.bar(x + width/2, kappa_geo, width, yerr=kappa_geo_err,
                   label=r"$\kappa_\mathrm{geo}$ (corrected extraction)", color="#DD8452", capsize=4)

    ax.set_xlabel("Geant4 EM Physics Option")
    ax.set_ylabel("Excess Kurtosis")
    ax.set_title("MCS Model Comparison: Molière vs Geometric Kurtosis")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.legend()
    ax.axhline(y=np.mean(kappa_geo), color="#DD8452", ls="--", alpha=0.5,
               label=f"mean $\\kappa_\\mathrm{{geo}}$ = {np.mean(kappa_geo):.2f}")
    ax.legend()

    # Value labels on bars
    for bar, val in zip(bars1, kappa_M):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, kappa_geo):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    PAPER_DIR.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        outpath = PAPER_DIR / f"figure_model_comparison.{ext}"
        fig.savefig(outpath, dpi=300)
        print(f"  Saved {outpath}")
    plt.close(fig)


def print_table(all_results):
    """Print comparison table to stdout."""
    # Reference: opt4
    ref = next(r for r in all_results if r["option"] == 4)

    print("\n" + "=" * 110)
    print(f"{'Option':<25s} {'kappa_M (solid)':<20s} {'kappa (rect40)':<20s} "
          f"{'kappa_geo':<15s} {'Delta_kappa_M from opt4'}")
    print("-" * 110)
    for r in all_results:
        delta = r["kappa_M"] - ref["kappa_M"]
        print(f"{r['name']:<25s} "
              f"{r['kappa_M']:>7.2f} ± {r['kappa_M_se']:<7.2f}   "
              f"{r['kappa_total']:>7.2f} ± {r['kappa_total_se']:<7.2f}   "
              f"{r['kappa_geo']:>7.2f} ± {r['kappa_geo_se']:<5.2f}   "
              f"{delta:>+7.2f}")
    print("=" * 110)

    # Diagnostic: check effective infill from pla_path
    print("\nDiagnostic: mean PLA path in rect40 files")
    for r in all_results:
        eff_infill = r["rect40_mean_pla_mm"] / SAMPLE_THICK * 100
        print(f"  {r['name']}: <pla_path> = {r['rect40_mean_pla_mm']:.2f} mm "
              f"(effective infill = {eff_infill:.1f}%, expected 40%)")

    # Key finding
    geos = [r["kappa_geo"] for r in all_results]
    print(f"\nkappa_geo values: {[f'{g:.2f}' for g in geos]}")
    print(f"  Spread: {max(geos) - min(geos):.2f}  (should be ~0 if geometry-only)")
    print(f"  Mean:   {np.mean(geos):.2f} ± {np.std(geos, ddof=1):.2f}")

    # Geometry validation
    avg_pla = np.mean([r["rect40_mean_pla_mm"] for r in all_results])
    eff_infill = avg_pla / SAMPLE_THICK * 100
    if abs(eff_infill - 40) > 5:
        print(f"\n  WARNING: rect40 files have unexpected effective infill (~{eff_infill:.0f}%"
              f" instead of 40%). Check geometry.")
    else:
        print(f"\n  Geometry OK: effective infill ~{eff_infill:.0f}% (expected 40%)")


def main():
    print("=" * 60)
    print("Model Comparison: Geant4 EM Physics Options 0, 3, 4")
    print("=" * 60)

    all_results = []
    for opt in [0, 3, 4]:
        print(f"\n── Option {opt}: {OPTION_NAMES[opt]} ──")
        r = analyze_option(opt)
        all_results.append(r)

    print_table(all_results)
    make_figure(all_results)

    # Save JSON
    RESULTS_DIR.mkdir(exist_ok=True)
    out_json = RESULTS_DIR / "model_comparison_results.json"
    output = {
        "options": all_results,
    }
    with open(out_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_json}")


if __name__ == "__main__":
    main()
