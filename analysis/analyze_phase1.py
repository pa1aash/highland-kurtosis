#!/usr/bin/env python3
"""
Phase 1 analysis: multi-material, muon, and thickness universality tests.

Sub-phases:
  1.1 — Silicon: universal kurtosis equation validation
  1.2 — Tungsten: same, with thick-target effects (10mm = 2.86 X0)
  1.3 — Muons: compare kappa_M(mu) vs kappa_M(e-), kappa_geo identical
  1.4 — Thickness variation: kappa_geo independent of thickness
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
BASE = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE / "results"
PAPER_DIR = BASE / "paper"

BEAM_ENERGY = 4.0  # GeV
N_BOOT = 1000
SEED = 42

# Material radiation lengths [mm]
X0 = {"PLA": 315.0, "Si": 93.7, "W": 3.5}

# Default sample thickness [mm]
DEFAULT_THICK = 10.0


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


def kappa_predicted(kappa_M, f):
    """Universal binary-lattice kurtosis: (3 + kappa_M) / f - 3."""
    return (3.0 + kappa_M) / f - 3.0


# ── I/O helpers ────────────────────────────────────────────────────────────
def load_root(filepath):
    """Load ROOT ntuple as dict of numpy arrays."""
    f = uproot.open(str(filepath))
    for name in ["scattering", "ntuple", "tree", "T"]:
        if name in f:
            return f[name].arrays(library="np")
    for key in f.keys():
        obj = f[key]
        if hasattr(obj, "arrays"):
            return obj.arrays(library="np")
    raise ValueError(f"No TTree found in {filepath}")


def apply_cuts(data, x0_mm, thickness_mm=DEFAULT_THICK,
               beam_energy=BEAM_ENERGY, energy_frac=0.9):
    """Apply standard cuts. Returns (cut_data, n_before, n_after)."""
    n_before = len(data["theta_x"])
    mask = np.ones(n_before, dtype=bool)

    # Energy cut
    mask &= data["energy_out"] > energy_frac * beam_energy

    # Fiducial
    mask &= (np.abs(data["entry_x"]) < 5.0) & (np.abs(data["entry_y"]) < 10.0)

    # Angle cut: 10 * sigma_Highland(thickness, material)
    sigma_h = highland_sigma_rad(thickness_mm / x0_mm, beam_energy)
    angle_max = 10.0 * sigma_h
    mask &= (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)

    n_after = int(np.sum(mask))
    cut = {k: v[mask] if isinstance(v, np.ndarray) and len(v) == n_before else v
           for k, v in data.items()}
    return cut, n_before, n_after


def analyze_file(filepath, x0_mm, thickness_mm=DEFAULT_THICK,
                 energy_frac=0.9, label=""):
    """Load, cut, compute kappa. Returns result dict."""
    data = load_root(filepath)
    cut, n_before, n_after = apply_cuts(data, x0_mm, thickness_mm,
                                        energy_frac=energy_frac)
    eff = n_after / n_before * 100

    tx, ty = cut["theta_x"], cut["theta_y"]
    kappa, se = kappa_bootstrap(tx, ty)
    sigma_x = float(np.std(tx, ddof=1))
    mean_pla = float(np.mean(cut["pla_path"])) if "pla_path" in cut else float("nan")

    if label:
        print(f"    {label}: N={n_before}->{n_after} ({eff:.1f}%), "
              f"kappa={kappa:.2f}+/-{se:.2f}, sigma_x={sigma_x*1e3:.3f} mrad, "
              f"<pla_path>={mean_pla:.2f} mm")

    return {
        "n_before": n_before, "n_after": n_after, "efficiency_pct": round(eff, 1),
        "kappa": round(kappa, 4), "se": round(se, 4),
        "sigma_x_mrad": round(sigma_x * 1e3, 4),
        "mean_pla_mm": round(mean_pla, 3),
    }


# ── Phase 1.1: Silicon ────────────────────────────────────────────────────
def phase1_silicon(pla_baseline):
    print("\n" + "=" * 72)
    print("  PHASE 1.1 — SILICON (X0 = 93.7 mm)")
    print("=" * 72)

    si_dir = BASE / "data" / "phase1_silicon"
    x0 = X0["Si"]

    # Solid
    solid = analyze_file(si_dir / "si_solid_4GeV.root", x0, label="Si solid")
    kappa_M_Si = solid["kappa"]

    # Rect at 40/60/80%
    infills = [40, 60, 80]
    rects = {}
    for inf in infills:
        r = analyze_file(si_dir / f"si_rect_{inf}pct_4GeV.root", x0,
                         label=f"Si rect {inf}%")
        rects[inf] = r

    # Prediction table
    print(f"\n  kappa_M(Si, solid) = {kappa_M_Si:.4f}")
    print(f"\n  {'Infill':>7} {'Measured':>10} {'Predicted':>10} {'Residual':>10} "
          f"{'kappa_geo(Si)':>14} {'kappa_geo(PLA)':>15}")
    print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*14} {'-'*15}")
    results = []
    for inf in infills:
        f = inf / 100.0
        pred = kappa_predicted(kappa_M_Si, f)
        meas = rects[inf]["kappa"]
        resid = meas - pred
        kg_si = meas - kappa_M_Si
        kg_pla = pla_baseline["rect40"]["kappa"] - pla_baseline["solid"]["kappa"] if inf == 40 else float("nan")
        print(f"  {inf:>6}% {meas:>10.2f} {pred:>10.2f} {resid:>+10.2f} "
              f"{kg_si:>14.2f} {kg_pla:>15.2f}")
        results.append({
            "infill_pct": inf, "measured": meas, "se": rects[inf]["se"],
            "predicted": round(pred, 4), "residual": round(resid, 4),
            "kappa_geo": round(kg_si, 4),
        })

    return {
        "material": "Si", "X0_mm": x0,
        "kappa_M_solid": solid,
        "rect_results": results,
    }


# ── Phase 1.2: Tungsten ───────────────────────────────────────────────────
def phase1_tungsten(pla_baseline):
    print("\n" + "=" * 72)
    print("  PHASE 1.2 — TUNGSTEN (X0 = 3.5 mm, 10mm = 2.86 X0)")
    print("=" * 72)

    w_dir = BASE / "data" / "phase1_tungsten"
    x0 = X0["W"]

    solid = analyze_file(w_dir / "w_solid_4GeV.root", x0, label="W solid")
    kappa_M_W = solid["kappa"]

    infills = [40, 60, 80]
    rects = {}
    for inf in infills:
        r = analyze_file(w_dir / f"w_rect_{inf}pct_4GeV.root", x0,
                         label=f"W rect {inf}%")
        rects[inf] = r

    print(f"\n  kappa_M(W, solid) = {kappa_M_W:.4f}")
    print(f"  Note: 10mm W = {DEFAULT_THICK/x0:.2f} X0 (thick target)")
    print(f"\n  Energy cut efficiency:")
    print(f"    solid: {solid['efficiency_pct']:.1f}%")
    for inf in infills:
        print(f"    rect {inf}%: {rects[inf]['efficiency_pct']:.1f}%")

    print(f"\n  {'Infill':>7} {'Measured':>10} {'Predicted':>10} {'Residual':>10} "
          f"{'kappa_geo(W)':>13} {'kappa_geo(PLA)':>15}")
    print(f"  {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*13} {'-'*15}")
    results = []
    for inf in infills:
        f = inf / 100.0
        pred = kappa_predicted(kappa_M_W, f)
        meas = rects[inf]["kappa"]
        resid = meas - pred
        kg_w = meas - kappa_M_W
        kg_pla = pla_baseline["rect40"]["kappa"] - pla_baseline["solid"]["kappa"] if inf == 40 else float("nan")
        print(f"  {inf:>6}% {meas:>10.2f} {pred:>10.2f} {resid:>+10.2f} "
              f"{kg_w:>13.2f} {kg_pla:>15.2f}")
        results.append({
            "infill_pct": inf, "measured": meas, "se": rects[inf]["se"],
            "predicted": round(pred, 4), "residual": round(resid, 4),
            "kappa_geo": round(kg_w, 4),
        })

    return {
        "material": "W", "X0_mm": x0,
        "kappa_M_solid": solid,
        "rect_results": results,
    }


# ── Phase 1.3: Muons ──────────────────────────────────────────────────────
def phase1_muons(pla_baseline):
    print("\n" + "=" * 72)
    print("  PHASE 1.3 — MUONS vs ELECTRONS (PLA, rect 40%)")
    print("=" * 72)

    mu_dir = BASE / "data" / "phase1_muons"
    x0 = X0["PLA"]

    # Muon with standard energy cut
    mu_solid = analyze_file(mu_dir / "muon_solid_4GeV.root", x0,
                            energy_frac=0.9, label="mu- solid (Ecut=0.9)")
    mu_rect = analyze_file(mu_dir / "muon_rect_40pct_4GeV.root", x0,
                           energy_frac=0.9, label="mu- rect40 (Ecut=0.9)")

    # Muon WITHOUT energy cut (energy_frac=0 effectively)
    mu_solid_nocut = analyze_file(mu_dir / "muon_solid_4GeV.root", x0,
                                  energy_frac=0.0, label="mu- solid (no Ecut)")
    mu_rect_nocut = analyze_file(mu_dir / "muon_rect_40pct_4GeV.root", x0,
                                 energy_frac=0.0, label="mu- rect40 (no Ecut)")

    # Electron baseline
    e_solid = pla_baseline["solid"]
    e_rect = pla_baseline["rect40"]

    kg_mu = mu_rect["kappa"] - mu_solid["kappa"]
    kg_e = e_rect["kappa"] - e_solid["kappa"]
    kg_mu_nocut = mu_rect_nocut["kappa"] - mu_solid_nocut["kappa"]

    print(f"\n  Comparison table:")
    print(f"  {'':>25} {'kappa_M':>10} {'kappa(rect40)':>14} {'kappa_geo':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*14} {'-'*10}")
    print(f"  {'e- (Ecut=0.9)':>25} {e_solid['kappa']:>10.2f} {e_rect['kappa']:>14.2f} {kg_e:>10.2f}")
    print(f"  {'mu- (Ecut=0.9)':>25} {mu_solid['kappa']:>10.2f} {mu_rect['kappa']:>14.2f} {kg_mu:>10.2f}")
    print(f"  {'mu- (no Ecut)':>25} {mu_solid_nocut['kappa']:>10.2f} {mu_rect_nocut['kappa']:>14.2f} {kg_mu_nocut:>10.2f}")

    print(f"\n  kappa_M: mu-={mu_solid['kappa']:.2f} vs e-={e_solid['kappa']:.2f} "
          f"(muon {'lower' if mu_solid['kappa'] < e_solid['kappa'] else 'higher'}, "
          f"{'expected' if mu_solid['kappa'] < e_solid['kappa'] else 'unexpected'}: no brems tail)")
    print(f"  kappa_geo: mu-={kg_mu:.2f} vs e-={kg_e:.2f} "
          f"(difference = {abs(kg_mu - kg_e):.2f})")
    print(f"  mu- Ecut sensitivity: kappa(0.9)={mu_rect['kappa']:.2f} vs "
          f"kappa(no cut)={mu_rect_nocut['kappa']:.2f} "
          f"(delta={abs(mu_rect['kappa'] - mu_rect_nocut['kappa']):.2f})")

    return {
        "muon_solid_ecut": mu_solid,
        "muon_rect40_ecut": mu_rect,
        "muon_solid_nocut": mu_solid_nocut,
        "muon_rect40_nocut": mu_rect_nocut,
        "electron_solid": e_solid,
        "electron_rect40": e_rect,
        "kappa_geo_muon": round(kg_mu, 4),
        "kappa_geo_electron": round(kg_e, 4),
        "kappa_geo_muon_nocut": round(kg_mu_nocut, 4),
    }


# ── Phase 1.4: Thickness ──────────────────────────────────────────────────
def phase1_thickness():
    print("\n" + "=" * 72)
    print("  PHASE 1.4 — THICKNESS VARIATION (PLA rect 40%)")
    print("=" * 72)

    thick_dir = BASE / "data" / "phase1_thickness"
    x0 = X0["PLA"]
    thicknesses = [5, 10, 20, 40]

    # We need solid controls at each thickness for kappa_M.
    # The thickness files are rect40 only.  For kappa_M we use the
    # standard 10mm solid as a reference (kappa_M depends on thickness
    # but we only have one solid file).  Actually the point is to show
    # kappa_geo is flat — we can compute it as kappa - kappa_M(thickness)
    # but we don't have solid files at each thickness.
    #
    # Strategy: use the 10mm solid PLA as kappa_M reference for all,
    # then note that kappa_geo = kappa_rect - kappa_M should be constant.
    # Since we don't have solid files at 5/20/40mm, we'll just report
    # total kappa and note the trend.

    # Load the 10mm solid for reference
    solid_10mm = analyze_file(
        BASE / "data" / "phase04_model_comparison" / "model_opt4_solid_4GeV.root",
        x0, thickness_mm=10.0, label="PLA solid 10mm (reference)")

    results = []
    print(f"\n  {'Thick(mm)':>10} {'kappa':>10} {'SE':>8} {'<pla_path>':>12} "
          f"{'kappa_geo*':>12} {'N_pass':>8}")
    print(f"  {'-'*10} {'-'*10} {'-'*8} {'-'*12} {'-'*12} {'-'*8}")

    for t in thicknesses:
        r = analyze_file(thick_dir / f"thick_{t}mm_rect_40pct_4GeV.root",
                         x0, thickness_mm=float(t),
                         label=f"rect40 {t}mm")
        # kappa_geo using 10mm solid reference (approximate)
        kg = r["kappa"] - solid_10mm["kappa"]
        print(f"  {t:>10} {r['kappa']:>10.2f} {r['se']:>8.2f} {r['mean_pla_mm']:>12.2f} "
              f"{kg:>12.2f} {r['n_after']:>8}")
        results.append({
            "thickness_mm": t, **r, "kappa_geo_approx": round(kg, 4),
        })

    print(f"\n  * kappa_geo computed using 10mm solid reference (kappa_M={solid_10mm['kappa']:.2f})")
    print(f"    kappa_geo should be ~constant if geometric contribution is thickness-independent")

    kgs = [r["kappa_geo_approx"] for r in results]
    print(f"    kappa_geo range: {min(kgs):.2f} to {max(kgs):.2f} (spread={max(kgs)-min(kgs):.2f})")

    return {
        "solid_10mm_reference": solid_10mm,
        "thickness_results": results,
    }


# ── Figures ────────────────────────────────────────────────────────────────
def figure_multi_material(si_results, w_results, pla_baseline):
    """kappa vs infill for PLA/Si/W with theory curves."""
    fig, ax = plt.subplots(figsize=(8, 6))

    infills_plot = np.linspace(0.15, 0.95, 200)

    materials = [
        ("PLA", X0["PLA"], pla_baseline, "#4C72B0"),
        ("Si", X0["Si"], si_results, "#DD8452"),
        ("W", X0["W"], w_results, "#55A868"),
    ]

    for mat_name, x0, res, color in materials:
        kappa_M = res["kappa_M_solid"]["kappa"]

        # Theory curve
        theory_k = [(3.0 + kappa_M) / f - 3.0 for f in infills_plot]
        ax.plot(infills_plot * 100, theory_k, '-', color=color, alpha=0.6,
                label=f"{mat_name} theory ($\\kappa_M$={kappa_M:.1f})")

        # Data points
        for r in res["rect_results"]:
            ax.errorbar(r["infill_pct"], r["measured"], yerr=r["se"],
                        fmt='o', color=color, markersize=8, capsize=4)

        # Solid point at f=1 (kappa_M)
        ax.errorbar(100, kappa_M, yerr=res["kappa_M_solid"]["se"],
                    fmt='s', color=color, markersize=8, capsize=4)

    ax.set_xlabel("Infill (%)", fontsize=12)
    ax.set_ylabel("Excess Kurtosis $\\kappa$", fontsize=12)
    ax.set_title("Universal Kurtosis Equation: Multi-Material Validation", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(15, 105)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PAPER_DIR / f"figure_multi_material.{ext}", dpi=300)
        print(f"  Saved paper/figure_multi_material.{ext}")
    plt.close(fig)


def figure_muon_comparison(muon_results):
    """Side-by-side bars of kappa_M and kappa_geo for e- vs mu-."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # kappa_M comparison
    labels = ["$e^-$", "$\\mu^-$\n(Ecut=0.9)", "$\\mu^-$\n(no Ecut)"]
    kM_vals = [
        muon_results["electron_solid"]["kappa"],
        muon_results["muon_solid_ecut"]["kappa"],
        muon_results["muon_solid_nocut"]["kappa"],
    ]
    kM_errs = [
        muon_results["electron_solid"]["se"],
        muon_results["muon_solid_ecut"]["se"],
        muon_results["muon_solid_nocut"]["se"],
    ]
    colors1 = ["#4C72B0", "#DD8452", "#DD8452"]
    bars1 = ax1.bar(range(3), kM_vals, yerr=kM_errs, color=colors1,
                    capsize=5, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(3))
    ax1.set_xticklabels(labels, fontsize=10)
    ax1.set_ylabel("$\\kappa_M$ (solid PLA)")
    ax1.set_title("Molière Kurtosis")
    for bar, val in zip(bars1, kM_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # kappa_geo comparison
    kg_vals = [
        muon_results["kappa_geo_electron"],
        muon_results["kappa_geo_muon"],
        muon_results["kappa_geo_muon_nocut"],
    ]
    # Propagated SE
    def prop_se(rect_r, solid_r):
        return np.sqrt(rect_r["se"]**2 + solid_r["se"]**2)
    kg_errs = [
        prop_se(muon_results["electron_rect40"], muon_results["electron_solid"]),
        prop_se(muon_results["muon_rect40_ecut"], muon_results["muon_solid_ecut"]),
        prop_se(muon_results["muon_rect40_nocut"], muon_results["muon_solid_nocut"]),
    ]
    colors2 = ["#4C72B0", "#DD8452", "#DD8452"]
    bars2 = ax2.bar(range(3), kg_vals, yerr=kg_errs, color=colors2,
                    capsize=5, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(range(3))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_ylabel("$\\kappa_{\\mathrm{geo}}$ (rect40 $-$ solid)")
    ax2.set_title("Geometric Kurtosis")
    for bar, val in zip(bars2, kg_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=9)

    # Horizontal line at mean kappa_geo
    mean_kg = np.mean(kg_vals)
    ax2.axhline(y=mean_kg, color="gray", ls="--", alpha=0.5)

    plt.suptitle("Electron vs Muon: PLA Rectilinear 40% at 4 GeV", fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PAPER_DIR / f"figure_muon_comparison.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"  Saved paper/figure_muon_comparison.{ext}")
    plt.close(fig)


def figure_thickness_independence(thick_results):
    """Two-panel: kappa vs thickness, kappa_geo vs thickness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    tvals = [r["thickness_mm"] for r in thick_results["thickness_results"]]
    kappas = [r["kappa"] for r in thick_results["thickness_results"]]
    kappa_ses = [r["se"] for r in thick_results["thickness_results"]]
    kgeos = [r["kappa_geo_approx"] for r in thick_results["thickness_results"]]

    # Panel 1: total kappa vs thickness
    ax1.errorbar(tvals, kappas, yerr=kappa_ses, fmt='o-', color="#4C72B0",
                 capsize=5, markersize=8)
    ax1.set_xlabel("Sample Thickness (mm)")
    ax1.set_ylabel("Excess Kurtosis $\\kappa$")
    ax1.set_title("Total Kurtosis vs Thickness")

    # Panel 2: kappa_geo vs thickness
    # SE for kappa_geo: propagated from rect SE and solid SE
    solid_se = thick_results["solid_10mm_reference"]["se"]
    kg_ses = [np.sqrt(se**2 + solid_se**2) for se in kappa_ses]
    ax2.errorbar(tvals, kgeos, yerr=kg_ses, fmt='s-', color="#DD8452",
                 capsize=5, markersize=8)
    ax2.axhline(y=np.mean(kgeos), color="gray", ls="--", alpha=0.5,
                label=f"mean = {np.mean(kgeos):.2f}")
    ax2.set_xlabel("Sample Thickness (mm)")
    ax2.set_ylabel("$\\kappa_{\\mathrm{geo}}$ (rect40 $-$ solid$_{10\\mathrm{mm}}$)")
    ax2.set_title("Geometric Kurtosis vs Thickness")
    ax2.legend()

    plt.suptitle("Thickness Independence: PLA Rectilinear 40% at 4 GeV", fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(PAPER_DIR / f"figure_thickness_independence.{ext}", dpi=300,
                    bbox_inches="tight")
        print(f"  Saved paper/figure_thickness_independence.{ext}")
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_DIR.mkdir(exist_ok=True)

    print("=" * 72)
    print("  PHASE 1: MULTI-MATERIAL, MUON, AND THICKNESS UNIVERSALITY")
    print("=" * 72)

    # PLA electron baseline (opt4)
    print("\n  Loading PLA electron baseline (opt4)...")
    mc_dir = BASE / "data" / "phase04_model_comparison"
    pla_solid = analyze_file(mc_dir / "model_opt4_solid_4GeV.root",
                             X0["PLA"], label="PLA e- solid")
    pla_rect = analyze_file(mc_dir / "model_opt4_rect40_4GeV.root",
                            X0["PLA"], label="PLA e- rect40")

    pla_baseline = {
        "solid": pla_solid, "rect40": pla_rect,
        "kappa_M_solid": pla_solid,
        "rect_results": [{
            "infill_pct": 40, "measured": pla_rect["kappa"], "se": pla_rect["se"],
            "predicted": round(kappa_predicted(pla_solid["kappa"], 0.4), 4),
            "residual": round(pla_rect["kappa"] - kappa_predicted(pla_solid["kappa"], 0.4), 4),
            "kappa_geo": round(pla_rect["kappa"] - pla_solid["kappa"], 4),
        }],
    }

    # Run all phases
    si_results = phase1_silicon(pla_baseline)
    w_results = phase1_tungsten(pla_baseline)
    muon_results = phase1_muons(pla_baseline)
    thick_results = phase1_thickness()

    # Summary
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    kg_pla = pla_rect["kappa"] - pla_solid["kappa"]
    kg_si_40 = next(r["kappa_geo"] for r in si_results["rect_results"]
                    if r["infill_pct"] == 40)
    kg_w_40 = next(r["kappa_geo"] for r in w_results["rect_results"]
                   if r["infill_pct"] == 40)
    print(f"\n  kappa_geo at 40% infill:")
    print(f"    PLA (e-):  {kg_pla:.2f}")
    print(f"    Si (e-):   {kg_si_40:.2f}")
    print(f"    W (e-):    {kg_w_40:.2f}")
    print(f"    PLA (mu-): {muon_results['kappa_geo_muon']:.2f}")
    print(f"    Spread: {max(kg_pla, kg_si_40, kg_w_40) - min(kg_pla, kg_si_40, kg_w_40):.2f}")

    # Figures
    print("\n  Generating figures...")
    figure_multi_material(si_results, w_results, pla_baseline)
    figure_muon_comparison(muon_results)
    figure_thickness_independence(thick_results)

    # Save JSON
    output = {
        "pla_baseline": {"solid": pla_solid, "rect40": pla_rect},
        "phase1_1_silicon": si_results,
        "phase1_2_tungsten": w_results,
        "phase1_3_muons": muon_results,
        "phase1_4_thickness": thick_results,
    }
    json_path = RESULTS_DIR / "phase1_results.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {json_path}")


if __name__ == "__main__":
    main()
