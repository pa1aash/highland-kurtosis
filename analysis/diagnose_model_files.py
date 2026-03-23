#!/usr/bin/env python3
"""Diagnostic: compare model comparison ROOT files vs proposal data."""

import numpy as np
from pathlib import Path
from scipy import stats
import uproot

PLA_X0_MM = 315.0

def highland_sigma_rad(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))

def diagnose(filepath, beam_energy=4.0):
    filepath = Path(filepath)
    print(f"\n{'='*72}")
    print(f"  FILE: {filepath.name}")
    print(f"{'='*72}")

    f = uproot.open(str(filepath))
    tree = f["scattering"]
    arrays = tree.arrays(library="np")

    n_total = len(arrays["theta_x"])
    print(f"  Total events: {n_total}")

    for key in ["theta_x", "theta_y", "energy_out", "entry_x", "entry_y", "pla_path"]:
        arr = arrays[key]
        print(f"  {key:12s}:  mean={np.mean(arr):12.6f}  std={np.std(arr):12.6f}  "
              f"min={np.min(arr):12.6f}  max={np.max(arr):12.6f}")

    # Kappa with NO cuts
    kappa_nocut = stats.kurtosis(arrays["theta_x"], fisher=True, bias=False)
    print(f"\n  Kappa(theta_x) NO cuts:  {kappa_nocut:.4f}  (N={n_total})")

    # Step-by-step cuts
    mask = np.ones(n_total, dtype=bool)

    # Energy cut
    emask = arrays["energy_out"] > 0.9 * beam_energy
    n_energy = np.sum(mask & ~emask)
    mask &= emask
    print(f"\n  After energy cut (E > {0.9*beam_energy:.1f} GeV):")
    print(f"    removed {n_energy}, remaining {np.sum(mask)}")
    kappa_e = stats.kurtosis(arrays["theta_x"][mask], fisher=True, bias=False)
    print(f"    kappa = {kappa_e:.4f}")

    # Fiducial cut
    fmask = (np.abs(arrays["entry_x"]) < 5.0) & (np.abs(arrays["entry_y"]) < 10.0)
    n_fid = np.sum(mask & ~fmask)
    mask &= fmask
    print(f"\n  After fiducial cut (|x|<5, |y|<10 mm):")
    print(f"    removed {n_fid}, remaining {np.sum(mask)}")
    kappa_ef = stats.kurtosis(arrays["theta_x"][mask], fisher=True, bias=False)
    print(f"    kappa = {kappa_ef:.4f}")

    # Angle cut
    sigma_h = highland_sigma_rad(10.0 / PLA_X0_MM, beam_energy)
    angle_max = 10.0 * sigma_h
    amask = (np.abs(arrays["theta_x"]) < angle_max) & (np.abs(arrays["theta_y"]) < angle_max)
    n_angle = np.sum(mask & ~amask)
    mask &= amask
    print(f"\n  After angle cut (10*sigma_H = {angle_max*1e3:.4f} mrad):")
    print(f"    sigma_Highland(10mm solid, {beam_energy} GeV) = {sigma_h*1e6:.2f} urad")
    print(f"    angle_max = {angle_max*1e3:.4f} mrad = {angle_max*1e6:.2f} urad")
    print(f"    removed {n_angle}, remaining {np.sum(mask)}")
    if np.sum(mask) > 0:
        kappa_efa = stats.kurtosis(arrays["theta_x"][mask], fisher=True, bias=False)
        print(f"    kappa = {kappa_efa:.4f}")
    else:
        print(f"    NO EVENTS REMAINING")

    # Check: what fraction of events have |theta_x| > angle_max?
    n_outliers = np.sum(np.abs(arrays["theta_x"]) > angle_max)
    print(f"\n  Events with |theta_x| > angle_max: {n_outliers} ({100*n_outliers/n_total:.2f}%)")

    # Check theta_x distribution percentiles
    pcts = [1, 5, 25, 50, 75, 95, 99]
    vals = np.percentile(arrays["theta_x"], pcts)
    print(f"\n  theta_x percentiles (mrad):")
    for p, v in zip(pcts, vals):
        print(f"    {p:3d}%: {v*1e3:10.4f} mrad")


def main():
    base = Path(__file__).resolve().parent.parent

    # Model comparison files
    model_dir = base / "data" / "phase04_model_comparison"
    for f in sorted(model_dir.glob("*.root")):
        diagnose(f)

    # Proposal reference
    proposal_file = base / "data" / "proposal" / "rect_40pct_4GeV.root"
    if proposal_file.exists():
        diagnose(proposal_file)


if __name__ == "__main__":
    main()
