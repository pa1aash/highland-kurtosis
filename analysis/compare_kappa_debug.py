#!/usr/bin/env python3
"""Compare kappa values between proposal and phase04 ROOT files.

Key question: why does the proposal give kappa~8.65 (paper) vs phase04 giving ~13.65?

Two analysis scripts exist with DIFFERENT cut logic:
  - analyze_mcs.py: uses Highland sigma from NOMINAL 10mm/X0 for angle cut
  - proposal_analysis.py: uses Highland sigma from ACTUAL mean(pla_path)/X0 after energy cut
"""

import numpy as np
from scipy import stats
import uproot

PLA_X0_MM = 315.0
SAMPLE_T_MM = 10.0

def highland_sigma_rad(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))

def load_root(filepath):
    f = uproot.open(filepath)
    for name in ["scattering", "ntuple", "tree", "T"]:
        if name in f:
            return f[name].arrays(library="np")
    raise ValueError(f"No TTree found in {filepath}")

def apply_cuts_analyze_mcs(data, beam_energy=4.0, energy_frac=0.9,
                            angle_sigma=10.0, fid_x=5.0, fid_y=10.0):
    """Cuts as in analyze_mcs.py: Highland sigma from NOMINAL 10mm/X0."""
    n_before = len(data["theta_x"])
    mask = data["energy_out"] > energy_frac * beam_energy
    mask &= (np.abs(data["entry_x"]) < fid_x) & (np.abs(data["entry_y"]) < fid_y)
    sigma_H = highland_sigma_rad(SAMPLE_T_MM / PLA_X0_MM, beam_energy)
    angle_max = angle_sigma * sigma_H
    mask &= (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)
    n_after = np.sum(mask)
    print(f"    [analyze_mcs cuts] sigma_H(10mm) = {sigma_H*1e3:.4f} mrad, "
          f"angle_max = {angle_max*1e3:.2f} mrad")
    print(f"    {n_before} -> {n_after} ({100*n_after/n_before:.1f}% kept)")
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == n_before}

def apply_cuts_proposal(data, beam_energy=4.0, energy_frac=0.9,
                         angle_sigma=10.0, fiducial=True,
                         fid_x=5.0, fid_y=10.0):
    """Cuts as in proposal_analysis.py: Highland sigma from ACTUAL mean(pla_path)/X0."""
    n_before = len(data["theta_x"])

    # Energy cut (+ optional fiducial)
    mask_energy = data["energy_out"] > energy_frac * beam_energy
    if fiducial:
        mask_energy &= (np.abs(data["entry_x"]) < fid_x) & (np.abs(data["entry_y"]) < fid_y)

    # Compute Highland sigma from ACTUAL mean pla_path (after energy+fid cut)
    mean_pla = np.mean(data["pla_path"][mask_energy])
    xX0_actual = mean_pla / PLA_X0_MM
    sigma_H = highland_sigma_rad(xX0_actual, beam_energy)

    if sigma_H > 0:
        angle_max = angle_sigma * sigma_H
    else:
        angle_max = 0.02

    angle_mask = (np.abs(data["theta_x"]) < angle_max) & (np.abs(data["theta_y"]) < angle_max)
    mask = mask_energy & angle_mask

    n_after = np.sum(mask)
    print(f"    [proposal cuts, fiducial={fiducial}] mean_pla = {mean_pla:.3f} mm, "
          f"x/X0 = {xX0_actual:.5f}")
    print(f"    sigma_H(actual) = {sigma_H*1e3:.4f} mrad, "
          f"angle_max = {angle_max*1e3:.2f} mrad")
    print(f"    {n_before} -> {n_after} ({100*n_after/n_before:.1f}% kept)")
    return {k: v[mask] for k, v in data.items() if isinstance(v, np.ndarray) and len(v) == n_before}

def compute_kappa(data, label):
    tx = data["theta_x"]
    ty = data["theta_y"]
    n = len(tx)
    kx = stats.kurtosis(tx, fisher=True, bias=False)
    ky = stats.kurtosis(ty, fisher=True, bias=False)
    kavg = (kx + ky) / 2.0
    se = np.sqrt(24.0 / n)
    sx = np.std(tx, ddof=1) * 1e3  # mrad
    mean_pla = np.mean(data.get("pla_path", np.array([0])))
    print(f"  {label}:")
    print(f"    N={n}, sigma_x={sx:.4f} mrad, mean_pla={mean_pla:.3f} mm")
    print(f"    kappa_x={kx:.4f}, kappa_y={ky:.4f}, kappa_avg={kavg:.4f} +/- {se:.4f}")
    return kx, ky, kavg, se, n


# =====================================================================
print("=" * 75)
print("PART 1: analyze_mcs.py cuts (Highland sigma from nominal 10mm/X0)")
print("=" * 75)

files = {
    "proposal rect_40pct": "data/proposal/rect_40pct_4GeV.root",
    "phase04 opt4 rect40": "data/phase04_model_comparison/model_opt4_rect40_4GeV.root",
    "proposal solid_ctrl": "data/proposal/control_solid_4GeV.root",
    "phase04 opt4 solid":  "data/phase04_model_comparison/model_opt4_solid_4GeV.root",
}

results_mcs = {}
for label, fpath in files.items():
    print(f"\n--- {label} ---")
    raw = load_root(fpath)
    print(f"  Raw events: {len(raw['theta_x'])}")
    cut = apply_cuts_analyze_mcs(raw)
    results_mcs[label] = compute_kappa(cut, f"{label} [analyze_mcs]")


# =====================================================================
print("\n\n" + "=" * 75)
print("PART 2: proposal_analysis.py cuts (Highland sigma from ACTUAL pla_path)")
print("  Using fiducial=True (primary analysis for paper Table 1)")
print("=" * 75)

results_prop_fid = {}
for label, fpath in files.items():
    print(f"\n--- {label} ---")
    raw = load_root(fpath)
    cut = apply_cuts_proposal(raw, fiducial=True)
    results_prop_fid[label] = compute_kappa(cut, f"{label} [proposal, fid=True]")

# =====================================================================
print("\n\n" + "=" * 75)
print("PART 3: proposal_analysis.py cuts with fiducial=False")
print("=" * 75)

results_prop_nofid = {}
for label, fpath in files.items():
    print(f"\n--- {label} ---")
    raw = load_root(fpath)
    cut = apply_cuts_proposal(raw, fiducial=False)
    results_prop_nofid[label] = compute_kappa(cut, f"{label} [proposal, fid=False]")


# =====================================================================
print("\n\n" + "=" * 75)
print("SUMMARY TABLE")
print("=" * 75)

print(f"\n{'File':<25s} | {'analyze_mcs':>12s} | {'prop fid=T':>12s} | {'prop fid=F':>12s} | Paper")
print("-" * 85)
paper_vals = {
    "proposal rect_40pct": "8.65",
    "proposal solid_ctrl": "3.62",
    "phase04 opt4 rect40": "---",
    "phase04 opt4 solid":  "---",
}
for label in files:
    k_mcs = results_mcs[label][2]       # kappa_avg
    k_fid = results_prop_fid[label][2]
    k_nofid = results_prop_nofid[label][2]
    pv = paper_vals[label]
    n_mcs = results_mcs[label][4]
    n_fid = results_prop_fid[label][4]
    n_nofid = results_prop_nofid[label][4]
    print(f"  {label:<23s} | {k_mcs:>6.2f} (N={n_mcs:>5d}) | "
          f"{k_fid:>6.2f} (N={n_fid:>5d}) | {k_nofid:>6.2f} (N={n_nofid:>5d}) | {pv:>5s}")

print(f"\nPaper Table 1 values:")
print(f"  Solid 4GeV:       kappa = 3.62 +/- 0.26  (N_pass = 46,102)")
print(f"  Rect 40% 4GeV:    kappa = 8.65 +/- 0.32  (N_pass = 40,587)")


# =====================================================================
print("\n\n" + "=" * 75)
print("KEY DIAGNOSTIC: Why the angle cut matters so much for lattice")
print("=" * 75)

raw_rect = load_root("data/proposal/rect_40pct_4GeV.root")
pla = raw_rect["pla_path"]
print(f"\nRect 40% pla_path distribution:")
print(f"  mean   = {np.mean(pla):.3f} mm")
print(f"  median = {np.median(pla):.3f} mm")
print(f"  std    = {np.std(pla):.3f} mm")
print(f"  frac(pla=0)        = {np.mean(pla < 0.01)*100:.1f}%  (air-only)")
print(f"  frac(pla=10mm)     = {np.mean(pla > 9.9)*100:.1f}%  (solid)")
print(f"  frac(0 < pla < 10) = {np.mean((pla > 0.01) & (pla < 9.9))*100:.1f}%  (partial)")

# The angle cut based on actual mean pla (~4mm) is TIGHTER than based on 10mm.
# This preferentially removes events that passed through more material
# (bigger scattering angles), biasing toward air-traversing particles.
sigma_nom = highland_sigma_rad(10.0 / PLA_X0_MM, 4.0)
mean_pla_rect = np.mean(pla)
sigma_actual = highland_sigma_rad(mean_pla_rect / PLA_X0_MM, 4.0)
print(f"\n  Highland sigma (10mm nominal):     {sigma_nom*1e3:.4f} mrad -> 10*sigma = {10*sigma_nom*1e3:.2f} mrad")
print(f"  Highland sigma ({mean_pla_rect:.1f}mm actual): {sigma_actual*1e3:.4f} mrad -> 10*sigma = {10*sigma_actual*1e3:.2f} mrad")
print(f"\n  The tighter angle cut removes MORE events with large scattering,")
print(f"  which are events that traversed more PLA (near solid).")
print(f"  This asymmetrically clips the heavy tails, REDUCING kappa.")


# =====================================================================
print("\n\n" + "=" * 75)
print("EXTRA: Check solid control with kappa_x only vs kappa_avg")
print("=" * 75)

raw_solid = load_root("data/proposal/control_solid_4GeV.root")
for fid in [True, False]:
    d = apply_cuts_proposal(raw_solid, fiducial=fid)
    tx = d["theta_x"]
    ty = d["theta_y"]
    n = len(tx)
    kx = stats.kurtosis(tx, fisher=True, bias=False)
    ky = stats.kurtosis(ty, fisher=True, bias=False)
    kavg = (kx + ky) / 2
    se = np.sqrt(24.0 / n)
    print(f"  fiducial={fid}: N={n}, kappa_x={kx:.4f}, kappa_y={ky:.4f}, "
          f"kappa_avg={kavg:.4f} +/- {se:.4f}")
    print(f"    Paper: 3.62 +/- 0.26 -> matches kappa_avg={kavg:.2f} "
          f"({'YES' if abs(kavg - 3.62) < 0.5 else 'NO'})")
