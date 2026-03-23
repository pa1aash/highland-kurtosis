#!/usr/bin/env python3
"""Diagnose stacked-layer issue: check pla_path distributions and layer independence."""

import numpy as np
import uproot
from pathlib import Path
from scipy import stats

BASE = Path(__file__).resolve().parent.parent
DATA_DIR = BASE / "data" / "phase3_stacked"

N_VALUES = [1, 2, 4, 10, 20]

print("=" * 80)
print("DIAGNOSTIC: pla_path statistics for stacked-layer ROOT files")
print("=" * 80)
print(f"{'N':>4s} | {'mean_pla':>10s} | {'std_pla':>10s} | {'var_pla':>10s} | "
      f"{'min':>8s} | {'max':>8s} | {'frac_0':>8s} | {'kurtosis':>10s}")
print("-" * 80)

for n in N_VALUES:
    fname = f"stacked_rect_20pct_{n}layer_4GeV.root"
    fpath = DATA_DIR / fname
    if not fpath.exists():
        print(f"  N={n}: FILE NOT FOUND")
        continue

    f = uproot.open(str(fpath))
    tree = f["scattering"]
    data = tree.arrays(library="np")

    pla = data["pla_path"]
    theta_x = data["theta_x"]

    frac_zero = np.sum(pla < 0.01) / len(pla)

    print(f"{n:4d} | {np.mean(pla):10.4f} | {np.std(pla):10.4f} | "
          f"{np.var(pla):10.4f} | {np.min(pla):8.4f} | {np.max(pla):8.4f} | "
          f"{frac_zero:8.4f} | {stats.kurtosis(theta_x, fisher=True, bias=False):10.4f}")

print("\n" + "=" * 80)
print("DIAGNOSTIC: pla_path histogram for N=1 and N=20")
print("=" * 80)

for n in [1, 20]:
    fname = f"stacked_rect_20pct_{n}layer_4GeV.root"
    fpath = DATA_DIR / fname
    f = uproot.open(str(fpath))
    tree = f["scattering"]
    pla = tree.arrays(library="np")["pla_path"]

    # Histogram of pla_path
    hist, edges = np.histogram(pla, bins=20, range=(0, 12))
    print(f"\nN={n} pla_path histogram (total={len(pla)}):")
    for i, count in enumerate(hist):
        bar = "#" * (count * 50 // max(hist))
        print(f"  [{edges[i]:5.1f}, {edges[i+1]:5.1f}) {count:6d} {bar}")

print("\n" + "=" * 80)
print("DIAGNOSTIC: expected vs actual effective infill")
print("=" * 80)
wall = 0.4  # mm
cell_in_macro = 1.78  # mm (what the macros use)
cell_for_20pct = 3.79  # mm (correct for 20%)

f_actual = 1 - (1 - wall / cell_in_macro) ** 2
f_intended = 1 - (1 - wall / cell_for_20pct) ** 2
print(f"  cellSize in macros: {cell_in_macro} mm → effective infill = {f_actual*100:.1f}%")
print(f"  cellSize for 20%:   {cell_for_20pct} mm → effective infill = {f_intended*100:.1f}%")
print(f"  BUG: macros specify 20% infill but cellSize gives ~{f_actual*100:.0f}% infill!")

# Check if var(pla_path) scales as 1/N
print("\n" + "=" * 80)
print("DIAGNOSTIC: Does var(pla_path) scale as 1/N?")
print("=" * 80)
vars_by_n = {}
for n in N_VALUES:
    fname = f"stacked_rect_20pct_{n}layer_4GeV.root"
    fpath = DATA_DIR / fname
    f = uproot.open(str(fpath))
    tree = f["scattering"]
    pla = tree.arrays(library="np")["pla_path"]
    vars_by_n[n] = np.var(pla)

var_1 = vars_by_n[1]
print(f"{'N':>4s} | {'var(pla)':>12s} | {'var(1)/N':>12s} | {'ratio':>8s}")
print("-" * 50)
for n in N_VALUES:
    ratio = vars_by_n[n] / (var_1 / n)
    print(f"{n:4d} | {vars_by_n[n]:12.4f} | {var_1/n:12.4f} | {ratio:8.4f}")
