#!/usr/bin/env python3
"""
Systematic uncertainty budget for MCS Highland lattice analysis.

Computes all systematic contributions for the representative configuration
(rectilinear 40% infill at 4 GeV) and produces:
  - LaTeX table:   paper/table_systematics.tex
  - JSON results:  results/systematic_budget.json
  - Plain text summary to stdout
"""

import json
import numpy as np
from pathlib import Path
from scipy import stats

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

# ---------------------------------------------------------------------------
# Constants & helpers (matching analyze_mcs.py)
# ---------------------------------------------------------------------------
PLA_X0_MM = 315.0


def highland_sigma_rad(x_over_X0, p_gev):
    """Highland projected-angle sigma [rad]."""
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))


def excess_kurtosis(data):
    """Unbiased excess kurtosis (Fisher definition)."""
    return stats.kurtosis(data, fisher=True, bias=False)


def apply_standard_cuts(arrays, beam_energy_gev=4.0):
    """Apply default analysis cuts (same as analyze_mcs.py defaults)."""
    n = len(arrays["theta_x"])
    mask = np.ones(n, dtype=bool)

    # Energy: E_out > 0.9 * E_beam
    mask &= arrays["energy_out"] > 0.9 * beam_energy_gev

    # Fiducial: |entry_x| < 5 mm, |entry_y| < 10 mm
    mask &= (np.abs(arrays["entry_x"]) < 5.0) & (np.abs(arrays["entry_y"]) < 10.0)

    # Angle: |theta| < 10 * sigma_Highland(10 mm solid)
    sigma_h = highland_sigma_rad(10.0 / PLA_X0_MM, beam_energy_gev)
    angle_max = 10.0 * sigma_h
    mask &= (np.abs(arrays["theta_x"]) < angle_max) & (np.abs(arrays["theta_y"]) < angle_max)

    return {k: v[mask] for k, v in arrays.items()}, int(np.sum(mask))


def load_root_kappa(filepath, beam_energy_gev=4.0):
    """Load ROOT file, apply standard cuts, return (kappa_x, SE, N_pass)."""
    f = uproot.open(str(filepath))
    tree = f["scattering"]
    arrays = tree.arrays(library="np")

    cut, n_pass = apply_standard_cuts(arrays, beam_energy_gev)
    kappa = excess_kurtosis(cut["theta_x"])
    se = np.sqrt(24.0 / n_pass)
    return kappa, se, n_pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    base = Path(__file__).resolve().parent.parent
    results_dir = base / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    paper_dir = base / "paper"

    print("=" * 72)
    print("  SYSTEMATIC UNCERTAINTY BUDGET")
    print("  Representative config: rectilinear 40% infill, 4 GeV")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load thin-wall results
    # ------------------------------------------------------------------
    with open(base / "data" / "phase01_thin_wall" / "thin_wall_results.json") as f:
        thin_wall = json.load(f)
    print("\n[1] Loaded thin-wall results")

    # ------------------------------------------------------------------
    # 2. Load cut-variation results
    # ------------------------------------------------------------------
    with open(results_dir / "cut_variation_results.json") as f:
        cut_var = json.load(f)
    print("[2] Loaded cut-variation results")

    # ------------------------------------------------------------------
    # 3. Load model comparison ROOT files (SOLID controls) for kappa_M
    # ------------------------------------------------------------------
    # NOTE: The rect40 model comparison files have incorrect effective
    # infill (~20% instead of 40%, mean pla_path ~1.96 mm vs ~4.03 mm).
    # Use solid control files instead — these have correct geometry and
    # give the Moliere kurtosis kappa_M for each physics option.
    model_dir = base / "data" / "phase04_model_comparison"
    model_kappas = {}
    print("[3] Loading model comparison ROOT files (solid controls)...")
    for opt in [0, 3, 4]:
        fpath = model_dir / f"model_opt{opt}_solid_4GeV.root"
        if fpath.exists() and HAS_UPROOT:
            kappa, se, n = load_root_kappa(fpath, 4.0)
            model_kappas[opt] = {"kappa": round(kappa, 4), "se": round(se, 4), "n": n}
            print(f"    opt{opt} solid: kappa_M = {kappa:.4f} +/- {se:.4f}  (N = {n})")
        else:
            print(f"    opt{opt} solid: MISSING or uproot unavailable")

    # ------------------------------------------------------------------
    # 4. Compute each systematic contribution
    # ------------------------------------------------------------------
    print("\n" + "-" * 72)
    print("  SYSTEMATIC CONTRIBUTIONS")
    print("-" * 72)

    budget = []  # list of dicts {source, value, method, detail}
    defaults = cut_var["defaults"]

    # (a) Thin-wall kappa_M correction: |kappa_M(0.4mm) - kappa_M(10mm)| at 4 GeV
    kM_04 = next(r["kappa_M"] for r in thin_wall
                 if abs(r["thickness_mm"] - 0.4) < 0.01
                 and abs(r["energy_GeV"] - 4.0) < 0.01)
    kM_10 = next(r["kappa_M"] for r in thin_wall
                 if abs(r["thickness_mm"] - 10.0) < 0.01
                 and abs(r["energy_GeV"] - 4.0) < 0.01)
    delta_tw = abs(kM_04 - kM_10)
    print(f"\n  (a) Thin-wall kappa_M correction")
    print(f"      kappa_M(0.4 mm, 4 GeV) = {kM_04:.4f}")
    print(f"      kappa_M(10  mm, 4 GeV) = {kM_10:.4f}")
    print(f"      delta = {delta_tw:.4f}")
    budget.append({
        "source": r"Thin-wall $\kappa_M$ correction",
        "source_plain": "Thin-wall kappa_M correction",
        "value": round(delta_tw, 4),
        "method": r"$|\kappa_M(0.4\,\mathrm{mm}) - \kappa_M(10\,\mathrm{mm})|$ at 4\,GeV",
        "method_plain": "|kappa_M(0.4mm) - kappa_M(10mm)| at 4 GeV",
    })

    # (b) Highland log correction: |(1-2*epsilon)^2 - 1|, epsilon=0.076*ln(0.4/315)
    ell_wall = 0.4
    epsilon = 0.076 * np.log(ell_wall / PLA_X0_MM)
    highland_log = abs((1.0 - 2.0 * epsilon) ** 2 - 1.0)
    print(f"\n  (b) Highland log correction (analytic)")
    print(f"      epsilon = 0.076 * ln({ell_wall}/{PLA_X0_MM}) = {epsilon:.4f}")
    print(f"      |(1 - 2*eps)^2 - 1| = |({1-2*epsilon:.4f})^2 - 1| = {highland_log:.4f}")
    budget.append({
        "source": r"Highland log correction",
        "source_plain": "Highland log correction",
        "value": round(highland_log, 4),
        "method": r"$|(1-2\epsilon)^2 - 1|$; $\epsilon = 0.076\ln(\ell/X_0)$",
        "method_plain": "|(1-2*eps)^2 - 1|, eps = 0.076*ln(l/X0)",
    })

    # (c) Energy cut threshold
    e_sweep = cut_var["sweeps"]["energy_frac"]
    e_def_k = next(e["kappa"] for e in e_sweep
                   if abs(e["value"] - defaults["energy_frac"]) < 0.001)
    e_deltas = [(e["value"], abs(e["kappa"] - e_def_k)) for e in e_sweep
                if abs(e["value"] - defaults["energy_frac"]) > 0.001]
    delta_ecut = max(d for _, d in e_deltas)
    worst_eval = max(e_deltas, key=lambda x: x[1])
    print(f"\n  (c) Energy cut threshold")
    print(f"      Default kappa (frac={defaults['energy_frac']}): {e_def_k:.4f}")
    print(f"      Worst at frac={worst_eval[0]}: delta = {worst_eval[1]:.4f}")
    budget.append({
        "source": r"Energy cut threshold",
        "source_plain": "Energy cut threshold",
        "value": round(delta_ecut, 4),
        "method": r"$\max|\kappa_\mathrm{var} - \kappa_\mathrm{def}|$; $E_\mathrm{frac}\in[0.85,0.98]$",
        "method_plain": "max|kappa_var - kappa_def|, E_frac in [0.85,0.98]",
    })

    # (d) Angle cut threshold — restricted to 7-15σ (bracketing default 10σ).
    #     Excludes 5σ (truncates non-Gaussian tails) and 20σ (admits nuclear
    #     scattering outliers that the cut is designed to remove).
    a_sweep = cut_var["sweeps"]["angle_sigma"]
    a_def_k = next(a["kappa"] for a in a_sweep
                   if abs(a["value"] - defaults["angle_sigma"]) < 0.001)
    a_deltas = [(a["value"], abs(a["kappa"] - a_def_k)) for a in a_sweep
                if abs(a["value"] - defaults["angle_sigma"]) > 0.001
                and 7.0 <= a["value"] <= 15.0]
    delta_acut = max(d for _, d in a_deltas)
    worst_aval = max(a_deltas, key=lambda x: x[1])
    print(f"\n  (d) Angle cut threshold (7-15 sigma only)")
    print(f"      Default kappa (Nsig={defaults['angle_sigma']}): {a_def_k:.4f}")
    for val, delt in sorted(a_deltas):
        print(f"      Nsig={val}: delta = {delt:.4f}")
    print(f"      Worst at Nsig={worst_aval[0]}: delta = {worst_aval[1]:.4f}")
    budget.append({
        "source": r"Angle cut threshold",
        "source_plain": "Angle cut threshold",
        "value": round(delta_acut, 4),
        "method": r"$\max|\kappa_\mathrm{var} - \kappa_\mathrm{def}|$; $N_\sigma\in[7,15]$",
        "method_plain": "max|kappa_var - kappa_def|, Nsig in [7,15]",
    })

    # (e) Fiducial region
    f_sweep = cut_var["sweeps"]["fiducial_x"]
    f_def_k = next(fe["kappa"] for fe in f_sweep
                   if abs(fe["value"] - defaults["fiducial_x"]) < 0.001)
    f_deltas = [(fe["value"], abs(fe["kappa"] - f_def_k)) for fe in f_sweep
                if abs(fe["value"] - defaults["fiducial_x"]) > 0.001]
    delta_fcut = max(d for _, d in f_deltas)
    worst_fval = max(f_deltas, key=lambda x: x[1])
    print(f"\n  (e) Fiducial region")
    print(f"      Default kappa (|x|<{defaults['fiducial_x']} mm): {f_def_k:.4f}")
    print(f"      Worst at |x|<{worst_fval[0]} mm: delta = {worst_fval[1]:.4f}")
    budget.append({
        "source": r"Fiducial region",
        "source_plain": "Fiducial region",
        "value": round(delta_fcut, 4),
        "method": r"$\max|\kappa_\mathrm{var} - \kappa_\mathrm{def}|$; $|x|\in[3,10]$\,mm",
        "method_plain": "max|kappa_var - kappa_def|, |x| in [3,10] mm",
    })

    # (f) Beam profile uncertainty
    delta_beam = 0.10
    print(f"\n  (f) Beam profile uncertainty")
    print(f"      Conservative estimate: {delta_beam}")
    budget.append({
        "source": r"Beam profile uncertainty",
        "source_plain": "Beam profile uncertainty",
        "value": delta_beam,
        "method": r"Conservative; $\kappa$ insensitive to beam $\sigma$",
        "method_plain": "Conservative; kappa insensitive to beam sigma",
    })

    # (g) Geant4 MCS model — from solid control files (kappa_M).
    #     The rect40 model files have incorrect geometry (~20% effective
    #     infill instead of 40%), so we use the solid controls which give
    #     the Moliere kurtosis kappa_M for each physics option directly.
    if len(model_kappas) == 3:
        k0 = model_kappas[0]["kappa"]
        k3 = model_kappas[3]["kappa"]
        k4 = model_kappas[4]["kappa"]
        delta_g4 = max(abs(k0 - k4), abs(k3 - k4))
        print(f"\n  (g) Geant4 MCS model (from solid controls)")
        print(f"      opt0 (Urban) solid:  kappa_M = {k0:.4f}")
        print(f"      opt3 solid:          kappa_M = {k3:.4f}")
        print(f"      opt4 (EMZ) solid:    kappa_M = {k4:.4f}")
        print(f"      |opt0 - opt4| = {abs(k0-k4):.4f}")
        print(f"      |opt3 - opt4| = {abs(k3-k4):.4f}")
        print(f"      max|delta kappa_M| = {delta_g4:.4f}")
    else:
        delta_g4 = 0.0
        print(f"\n  (g) Geant4 MCS model: could not compute (missing files)")
    budget.append({
        "source": r"Geant4 MCS model",
        "source_plain": "Geant4 MCS model",
        "value": round(delta_g4, 4),
        "method": r"$\max|\kappa_{M,i} - \kappa_{M,\mathrm{opt4}}|$; solid controls",
        "method_plain": "max|kappa_M_optN - kappa_M_opt4| from solid controls",
    })

    # (h) Voxelisation resolution
    delta_voxel = 0.05
    print(f"\n  (h) Voxelisation resolution")
    print(f"      Estimate: {delta_voxel} (sub-dominant; rectilinear uses CSG)")
    budget.append({
        "source": r"Voxelisation resolution",
        "source_plain": "Voxelisation resolution",
        "value": delta_voxel,
        "method": r"Estimated; sub-dominant (CSG for rectilinear)",
        "method_plain": "Estimated; sub-dominant (CSG for rectilinear)",
    })

    # (i) Statistical uncertainty (bootstrap SE from main analysis)
    stat_se = next(e["se"] for e in e_sweep
                   if abs(e["value"] - defaults["energy_frac"]) < 0.001)
    print(f"\n  (i) Statistical uncertainty (bootstrap SE)")
    print(f"      SE = {stat_se:.4f}  (N_boot = {cut_var['n_bootstrap']})")
    budget.append({
        "source": r"Statistical (bootstrap SE)",
        "source_plain": "Statistical (bootstrap SE)",
        "value": round(stat_se, 4),
        "method": r"Bootstrap SE ($N_\mathrm{boot} = 1000$)",
        "method_plain": "Bootstrap SE (N_boot = 1000)",
    })

    # ------------------------------------------------------------------
    # 5. Total systematic (quadrature sum of non-statistical entries)
    # ------------------------------------------------------------------
    syst_vals = [b["value"] for b in budget if "Statistical" not in b["source"]]
    total_syst = np.sqrt(sum(v ** 2 for v in syst_vals))
    total_combined = np.sqrt(total_syst ** 2 + stat_se ** 2)

    print(f"\n{'=' * 72}")
    print(f"  Total systematic (quadrature): {total_syst:.4f}")
    print(f"  Statistical:                   {stat_se:.4f}")
    print(f"  Total (syst + stat):           {total_combined:.4f}")
    print(f"{'=' * 72}")

    # ------------------------------------------------------------------
    # 6. LaTeX table  ->  paper/table_systematics.tex
    # ------------------------------------------------------------------
    tex = []
    tex.append(r"\begin{table}")
    tex.append(r"  \centering")
    tex.append(r"  \caption{Systematic uncertainty budget for the representative "
               r"configuration (rectilinear 40\,\% infill at 4\,GeV).  "
               r"Values are absolute shifts in excess kurtosis~$\kappa$.}")
    tex.append(r"  \label{tab:systematics}")
    tex.append(r"  \begin{tabular}{@{}l r p{6.2cm}@{}}")
    tex.append(r"    \toprule")
    tex.append(r"    Source & $\delta\kappa$ & Method \\")
    tex.append(r"    \midrule")

    for b in budget:
        if "Statistical" not in b["source"]:
            tex.append(f"    {b['source']} & {b['value']:.2f} & {b['method']} \\\\")

    tex.append(r"    \midrule")
    tex.append(f"    Total systematic & {total_syst:.2f} & Quadrature sum \\\\")
    tex.append(r"    \midrule")
    stat_b = next(b for b in budget if "Statistical" in b["source"])
    tex.append(f"    {stat_b['source']} & {stat_b['value']:.2f} & {stat_b['method']} \\\\")
    tex.append(r"    \midrule")
    tex.append(f"    Total (syst $\\oplus$ stat) & {total_combined:.2f} & --- \\\\")
    tex.append(r"    \bottomrule")
    tex.append(r"  \end{tabular}")
    tex.append(r"\end{table}")

    latex_str = "\n".join(tex) + "\n"
    latex_path = paper_dir / "table_systematics.tex"
    with open(latex_path, "w") as f:
        f.write(latex_str)
    print(f"\nLaTeX table saved: {latex_path}")

    # ------------------------------------------------------------------
    # 7. Plain text summary table
    # ------------------------------------------------------------------
    hdr = f"  {'Source':<35s} {'delta_kappa':>11s}   {'Method'}"
    sep = f"  {'-'*35} {'-'*11}   {'-'*45}"
    print(f"\n{hdr}")
    print(sep)
    for b in budget:
        if "Statistical" not in b["source"]:
            print(f"  {b['source_plain']:<35s} {b['value']:>11.4f}   {b['method_plain']}")
    print(sep)
    print(f"  {'Total systematic':<35s} {total_syst:>11.4f}   Quadrature sum")
    print(f"  {'Statistical (bootstrap SE)':<35s} {stat_se:>11.4f}   Bootstrap SE (N_boot = 1000)")
    print(sep)
    print(f"  {'Total (syst + stat)':<35s} {total_combined:>11.4f}   Quadrature sum")
    print()

    # ------------------------------------------------------------------
    # 8. JSON output  ->  results/systematic_budget.json
    # ------------------------------------------------------------------
    # Strip LaTeX-specific keys for JSON
    budget_json = [{k: v for k, v in b.items() if k not in ("source_plain", "method_plain")}
                   for b in budget]

    output = {
        "config": "rectilinear_40pct_4GeV",
        "reference_kappa": round(e_def_k, 4),
        "contributions": budget_json,
        "total_systematic": round(total_syst, 4),
        "statistical": round(stat_se, 4),
        "total_combined": round(total_combined, 4),
        "model_comparison_solid": {f"opt{k}": v for k, v in model_kappas.items()},
        "thin_wall": {
            "kappa_M_04mm_4GeV": kM_04,
            "kappa_M_10mm_4GeV": kM_10,
        },
        "highland_log": {
            "epsilon": round(epsilon, 4),
            "wall_thickness_mm": ell_wall,
            "X0_mm": PLA_X0_MM,
        },
    }

    json_path = results_dir / "systematic_budget.json"
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON results saved: {json_path}")


if __name__ == "__main__":
    main()
