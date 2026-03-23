#!/usr/bin/env python3
"""Cut variation study: sweep energy, angle, and fiducial cuts to check kappa stability."""

import numpy as np
from scipy import stats
from pathlib import Path
import json
import argparse
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reuse utilities from the main analysis module
sys.path.insert(0, str(Path(__file__).parent))
from analyze_mcs import load_root_file, highland_sigma_rad, PLA_X0_MM


# ---------------------------------------------------------------------------
# Defaults (must match analyze_mcs.py)
# ---------------------------------------------------------------------------
DEFAULT_ENERGY_FRAC = 0.90
DEFAULT_ANGLE_SIGMA = 10.0
DEFAULT_FIDUCIAL_X  = 5.0
DEFAULT_FIDUCIAL_Y  = 10.0        # held constant throughout
BEAM_ENERGY_GEV     = 4.0
BOOTSTRAP_N         = 1000

# Sweep grids
ENERGY_FRACS = [0.85, 0.90, 0.92, 0.95, 0.98]
ANGLE_SIGMAS = [5.0, 7.0, 10.0, 15.0, 20.0]
FIDUCIAL_XS  = [3.0, 5.0, 7.0, 10.0]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def apply_cuts(data, energy_frac, angle_sigma, fid_x, fid_y=DEFAULT_FIDUCIAL_Y,
               beam_energy=BEAM_ENERGY_GEV):
    """Return boolean mask for events passing all cuts."""
    mask = np.ones(len(data["theta_x"]), dtype=bool)
    mask &= data["energy_out"] > energy_frac * beam_energy
    mask &= (np.abs(data["entry_x"]) < fid_x) & (np.abs(data["entry_y"]) < fid_y)
    sigma_h = highland_sigma_rad(10.0 / PLA_X0_MM, beam_energy)
    angle_max = angle_sigma * sigma_h
    mask &= (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)
    return mask


def kappa_bootstrap(theta, n_boot=BOOTSTRAP_N, rng=None):
    """Return (kappa, SE) from bootstrap resampling."""
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(theta)
    kappas = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        kappas[i] = stats.kurtosis(theta[idx], fisher=True, bias=False)
    return float(np.mean(kappas)), float(np.std(kappas, ddof=1))


# ---------------------------------------------------------------------------
# Sweep logic
# ---------------------------------------------------------------------------
def sweep_one_parameter(data, param_name, values, default_energy_frac,
                        default_angle_sigma, default_fid_x, n_boot=BOOTSTRAP_N):
    """Sweep *param_name* over *values*, holding others at default."""
    rows = []
    rng = np.random.default_rng(42)
    for val in values:
        if param_name == "energy_frac":
            mask = apply_cuts(data, val, default_angle_sigma, default_fid_x)
        elif param_name == "angle_sigma":
            mask = apply_cuts(data, default_energy_frac, val, default_fid_x)
        elif param_name == "fiducial_x":
            mask = apply_cuts(data, default_energy_frac, default_angle_sigma, val)
        else:
            raise ValueError(param_name)

        theta = data["theta_x"][mask]
        n_pass = int(np.sum(mask))
        if n_pass < 100:
            print(f"  WARNING: {param_name}={val} -> only {n_pass} events, skipping")
            continue
        kappa, se = kappa_bootstrap(theta, n_boot=n_boot, rng=rng)
        rows.append({
            "parameter": param_name,
            "value": val,
            "n_pass": n_pass,
            "kappa": round(kappa, 4),
            "se": round(se, 4),
        })
        print(f"  {param_name}={val:>6}  N={n_pass:>6}  kappa={kappa:+.4f} +/- {se:.4f}")
    return rows


# ---------------------------------------------------------------------------
# Saturation-point finder
# ---------------------------------------------------------------------------
def find_saturation(rows, default_value):
    """Return the first value where kappa deviates >2*SE from default-cut kappa."""
    ref = None
    for r in rows:
        if r["value"] == default_value:
            ref = r
            break
    if ref is None:
        return None

    kappa_ref = ref["kappa"]
    se_ref = ref["se"]

    for r in rows:
        if abs(r["kappa"] - kappa_ref) > 2.0 * se_ref:
            return r["value"]
    return None  # no deviation found


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def make_figure(results_by_param, output_stem):
    """3-panel figure: kappa vs threshold for each cut type."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panel_info = [
        ("energy_frac",  "Energy cut fraction $E_{\\rm out}/E_{\\rm beam}$",
         DEFAULT_ENERGY_FRAC),
        ("angle_sigma",  "Angle cut ($N \\times \\sigma_{\\rm Highland}$)",
         DEFAULT_ANGLE_SIGMA),
        ("fiducial_x",   "Fiducial $|x|$ cut (mm)",
         DEFAULT_FIDUCIAL_X),
    ]

    for ax, (pname, xlabel, default_val) in zip(axes, panel_info):
        rows = results_by_param[pname]
        vals   = [r["value"]  for r in rows]
        kappas = [r["kappa"]  for r in rows]
        ses    = [r["se"]     for r in rows]

        ax.errorbar(vals, kappas, yerr=ses, fmt='o-', capsize=4,
                    color='C0', markersize=7, linewidth=1.5)

        # shade +/- 2 SE band around default
        ref_row = next((r for r in rows if r["value"] == default_val), None)
        if ref_row:
            k0, s0 = ref_row["kappa"], ref_row["se"]
            ax.axhspan(k0 - 2*s0, k0 + 2*s0, color='C0', alpha=0.12,
                       label=f'default $\\pm 2\\,$SE')
            ax.axhline(k0, ls='--', color='C0', alpha=0.5)

        # mark saturation point
        sat = find_saturation(rows, default_val)
        if sat is not None:
            sat_row = next(r for r in rows if r["value"] == sat)
            ax.plot(sat, sat_row["kappa"], 'r*', markersize=14, zorder=5,
                    label=f'saturation at {sat}')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Excess kurtosis $\\kappa$', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle('Cut robustness: kappa stability under cut variation\n'
                 '(rect 40% infill, 4 GeV $e^-$)', fontsize=14, y=1.02)
    fig.tight_layout()

    for ext in ('.png', '.pdf'):
        fig.savefig(str(output_stem) + ext, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved: {output_stem}.png / .pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Cut variation study for kappa robustness")
    parser.add_argument("--input", type=str,
                        default=str(Path(__file__).parent.parent /
                                    "data/proposal/rect_40pct_4GeV.root"),
                        help="ROOT file to analyse")
    parser.add_argument("--n-bootstrap", type=int, default=BOOTSTRAP_N)
    args = parser.parse_args()

    n_bootstrap = args.n_bootstrap

    # Load data
    filepath = Path(args.input)
    print(f"Loading {filepath.name} ...")
    data = load_root_file(filepath)
    n_total = len(data["theta_x"])
    print(f"  Total events: {n_total}")

    # Run sweeps
    results_by_param = {}

    print("\n--- Energy cut fraction sweep ---")
    results_by_param["energy_frac"] = sweep_one_parameter(
        data, "energy_frac", ENERGY_FRACS,
        DEFAULT_ENERGY_FRAC, DEFAULT_ANGLE_SIGMA, DEFAULT_FIDUCIAL_X, n_bootstrap)

    print("\n--- Angle cut sigma sweep ---")
    results_by_param["angle_sigma"] = sweep_one_parameter(
        data, "angle_sigma", ANGLE_SIGMAS,
        DEFAULT_ENERGY_FRAC, DEFAULT_ANGLE_SIGMA, DEFAULT_FIDUCIAL_X, n_bootstrap)

    print("\n--- Fiducial x cut sweep ---")
    results_by_param["fiducial_x"] = sweep_one_parameter(
        data, "fiducial_x", FIDUCIAL_XS,
        DEFAULT_ENERGY_FRAC, DEFAULT_ANGLE_SIGMA, DEFAULT_FIDUCIAL_X, n_bootstrap)

    # Saturation points
    saturations = {
        "energy_frac": find_saturation(results_by_param["energy_frac"],
                                       DEFAULT_ENERGY_FRAC),
        "angle_sigma": find_saturation(results_by_param["angle_sigma"],
                                       DEFAULT_ANGLE_SIGMA),
        "fiducial_x":  find_saturation(results_by_param["fiducial_x"],
                                       DEFAULT_FIDUCIAL_X),
    }

    # Print summary table
    print("\n" + "="*72)
    print("CUT VARIATION SUMMARY")
    print("="*72)
    for pname in ("energy_frac", "angle_sigma", "fiducial_x"):
        rows = results_by_param[pname]
        print(f"\n  {pname}:")
        print(f"  {'Value':>8}  {'N_pass':>8}  {'kappa':>10}  {'SE':>8}")
        for r in rows:
            marker = " <-- default" if (
                (pname == "energy_frac" and r["value"] == DEFAULT_ENERGY_FRAC) or
                (pname == "angle_sigma" and r["value"] == DEFAULT_ANGLE_SIGMA) or
                (pname == "fiducial_x"  and r["value"] == DEFAULT_FIDUCIAL_X)
            ) else ""
            print(f"  {r['value']:>8}  {r['n_pass']:>8}  {r['kappa']:>+10.4f}  "
                  f"{r['se']:>8.4f}{marker}")
        sat = saturations[pname]
        if sat is not None:
            print(f"  -> Saturation at {pname} = {sat}")
        else:
            print(f"  -> No saturation detected (kappa stable within 2*SE)")

    # Save results JSON
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / "cut_variation_results.json"

    output = {
        "input_file": str(filepath),
        "n_total_events": n_total,
        "n_bootstrap": n_bootstrap,
        "defaults": {
            "energy_frac": DEFAULT_ENERGY_FRAC,
            "angle_sigma": DEFAULT_ANGLE_SIGMA,
            "fiducial_x": DEFAULT_FIDUCIAL_X,
            "fiducial_y": DEFAULT_FIDUCIAL_Y,
        },
        "sweeps": {k: v for k, v in results_by_param.items()},
        "saturation_points": saturations,
    }
    with open(out_json, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_json}")

    # Make figure
    paper_dir = Path(__file__).parent.parent / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    make_figure(results_by_param, paper_dir / "figure_cut_robustness")

    print("\nDone.")


if __name__ == "__main__":
    main()
