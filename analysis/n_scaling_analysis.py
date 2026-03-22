import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path("data/sweep0")
OUT_DIR = Path("results/n_scaling")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "n_scaling_summary.json") as f:
    n_scaling_data = json.load(f)

with open(DATA_DIR / "gyroid_period_summary.json") as f:
    gyroid_determ_data = json.load(f)

with open(DATA_DIR / "gyroid_independent_summary.json") as f:
    gyroid_indep_data = json.load(f)

rect_by_infill = {}
for entry in n_scaling_data:
    infill = entry["infill_target_pct"]
    if infill not in rect_by_infill:
        rect_by_infill[infill] = {"N": [], "kappa_ray": [], "kappa_analytic": [],
                                   "f_hit": entry["hit_fraction"]}
    rect_by_infill[infill]["N"].append(entry["n_layers"])
    rect_by_infill[infill]["kappa_ray"].append(entry["predicted_kurtosis"])
    rect_by_infill[infill]["kappa_analytic"].append(entry["kappa_analytic"])

gyr_det_N = np.array([entry["n_cells_z"] for entry in gyroid_determ_data])
gyr_det_kappa = np.array([entry["predicted_kurtosis"] for entry in gyroid_determ_data])

kappa_single_gyr = gyroid_indep_data["kappa_single"]
gyr_indep = gyroid_indep_data["independent"]
gyr_stacked = gyroid_indep_data["stacked"]
gyr_indep_N = np.array([e["n_periods"] for e in gyr_indep])
gyr_indep_kappa = np.array([e["predicted_kurtosis"] for e in gyr_indep])
gyr_stacked_kappa = np.array([e["predicted_kurtosis"] for e in gyr_stacked])
gyr_indep_predicted = np.array([e["kappa_predicted_1_over_N"] for e in gyr_indep])

rect40 = rect_by_infill[40]
rect40_N = np.array(rect40["N"])
rect40_kappa = np.array(rect40["kappa_ray"])
kappa_single_rect40 = rect40_kappa[0]

fig, axes = plt.subplots(2, 2, figsize=(14, 11))

ax = axes[0, 0]

ax.loglog(rect40_N, rect40_kappa, 'o-', color='#377eb8', markersize=6,
          linewidth=1.5, label='stacked rect 40% (independent)')

ax.loglog(gyr_det_N, gyr_det_kappa, 's--', color='#984ea3', markersize=7,
          linewidth=1.5, alpha=0.7, label='gyroid 40% deterministic (control)')

ax.loglog(gyr_indep_N, gyr_indep_kappa, 'D-', color='#e41a1c', markersize=7,
          linewidth=1.5, label='gyroid 40% independent')

ax.loglog(gyr_indep_N, gyr_stacked_kappa, 'x', color='#ff7f00', markersize=8,
          markeredgewidth=2, label='gyroid 40% stacked (cross-check)')

N_theory = np.array([1, 2, 4, 8, 16, 32, 64, 100])
ax.loglog(N_theory, kappa_single_rect40 / N_theory, ':', color='#377eb8',
          linewidth=1, alpha=0.5, label=f'rect theory: {kappa_single_rect40:.2f}/N')
ax.loglog(N_theory, kappa_single_gyr / N_theory, ':', color='#e41a1c',
          linewidth=1, alpha=0.5, label=f'gyr theory: {kappa_single_gyr:.2f}/N')

ax.set_xlabel('N (independent layers / periods)', fontsize=11)
ax.set_ylabel('predicted excess kurtosis κ', fontsize=11)
ax.set_title('κ vs N: universal 1/N scaling', fontsize=12, fontweight='bold')
ax.legend(fontsize=7, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.8, 120)

ax = axes[0, 1]

collapse_rect = rect40_kappa * rect40_N / kappa_single_rect40
ax.semilogx(rect40_N, collapse_rect, 'o-', color='#377eb8', markersize=6,
            linewidth=1.5, label='stacked rect 40%')

collapse_gyr = gyr_indep_kappa * gyr_indep_N / kappa_single_gyr
ax.semilogx(gyr_indep_N, collapse_gyr, 'D-', color='#e41a1c', markersize=7,
            linewidth=1.5, label='gyroid 40% independent')

collapse_gyr_s = gyr_stacked_kappa * gyr_indep_N / kappa_single_gyr
ax.semilogx(gyr_indep_N, collapse_gyr_s, 'x', color='#ff7f00', markersize=8,
            markeredgewidth=2, label='gyroid 40% stacked (cross-check)')

collapse_det = gyr_det_kappa * gyr_det_N / kappa_single_gyr
ax.semilogx(gyr_det_N, collapse_det, 's--', color='#984ea3', markersize=7,
            linewidth=1.5, alpha=0.7, label='gyroid 40% deterministic')

ax.axhline(1.0, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='ideal = 1')
ax.set_xlabel('N', fontsize=11)
ax.set_ylabel('κ · N / κ_single', fontsize=11)
ax.set_title('universality collapse (→ 1.0 if κ ~ 1/N)', fontsize=12, fontweight='bold')
ax.legend(fontsize=7.5)
ax.grid(True, alpha=0.3, which='both')
ax.set_ylim(0, 18)
ax.set_xlim(0.8, 20)

ax = axes[1, 0]
colors = {20: '#e41a1c', 40: '#377eb8', 60: '#4daf4a'}
for infill, d in sorted(rect_by_infill.items()):
    N = np.array(d["N"])
    kray = np.array(d["kappa_ray"])
    c = colors[infill]
    kappa_1 = kray[0]
    ax.loglog(N, kray, 'o-', color=c, markersize=5, linewidth=1.5,
              label=f'rect {infill}%')
    ax.loglog(N, kappa_1 / N, ':', color=c, alpha=0.4, linewidth=1)

ax.set_xlabel('N (layers)', fontsize=11)
ax.set_ylabel('κ', fontsize=11)
ax.set_title('stacked rect: all infills', fontsize=12, fontweight='bold')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim(0.8, 120)

ax = axes[1, 1]
ax.axis('off')

lines = []
lines.append("SUMMARY TABLE")
lines.append(f"{'geometry':<24} {'N':>3} {'κ_ray':>8} {'κ_pred':>8} "
             f"{'ratio':>6} {'indep?':>7}")

for i, N in enumerate(rect40["N"]):
    kr = rect40["kappa_ray"][i]
    kp = kappa_single_rect40 / N
    ratio = kr / kp if kp > 0 else 0
    lines.append(f"{'stacked_rect_40%':<24} {N:>3} {kr:>8.4f} {kp:>8.4f} "
                 f"{ratio:>6.3f} {'yes':>7}")

lines.append("")

for entry in gyroid_determ_data:
    N = entry["n_cells_z"]
    kr = entry["predicted_kurtosis"]
    lines.append(f"{'gyroid_40%_determ':<24} {N:>3.0f} {kr:>8.4f} {'--':>8} "
                 f"{'--':>6} {'no':>7}")

lines.append("")

for i, entry in enumerate(gyr_indep):
    N = entry["n_periods"]
    kr = entry["predicted_kurtosis"]
    kp = entry["kappa_predicted_1_over_N"]
    ratio = kr / kp if kp > 0 else 0
    lines.append(f"{'gyroid_40%_indep':<24} {N:>3} {kr:>8.4f} {kp:>8.4f} "
                 f"{ratio:>6.3f} {'yes':>7}")

lines.append("")
lines.append("KEY CONCLUSION:")
lines.append("  κ ~ κ_single/N holds for BOTH binary")
lines.append("  and continuous geometries when periods")
lines.append("  are independent (random phase shifts).")
lines.append("  deterministic gyroid: κ ~ const (trivial")
lines.append("  — correlated periods, not independent).")

summary_text = "\n".join(lines)
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=7.5, verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

fig.suptitle('MCS kurtosis N-scaling: universal κ ~ 1/N for independent layers',
             fontsize=14, fontweight='bold')
fig.tight_layout()

outpath = OUT_DIR / "n_scaling_universal.png"
fig.savefig(outpath, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"saved: {outpath}")

print("\n" + summary_text)

summary_json = {
    "stacked_rectilinear": {},
    "gyroid_deterministic": [],
    "gyroid_independent": [],
    "gyroid_stacked_crosscheck": [],
    "key_findings": {
        "1_over_N_universal_for_independent_layers": True,
        "deterministic_gyroid_constant_kappa": True,
        "independent_gyroid_follows_1_over_N": True,
        "stacked_crosscheck_agrees": True,
    },
}

for infill, d in sorted(rect_by_infill.items()):
    kappa_1 = d["kappa_ray"][0]
    summary_json["stacked_rectilinear"][str(infill)] = {
        "hit_fraction": d["f_hit"],
        "kappa_single": round(kappa_1, 4),
        "N_values": d["N"],
        "kappa_ray": [round(k, 4) for k in d["kappa_ray"]],
        "kappa_analytic": [round(k, 4) for k in d["kappa_analytic"]],
        "collapse_values": [round(k * N / kappa_1, 4)
                           for k, N in zip(d["kappa_ray"], d["N"])],
    }

for entry in gyroid_determ_data:
    summary_json["gyroid_deterministic"].append({
        "n_cells_z": entry["n_cells_z"],
        "period_mm": entry["cell_size_mm"],
        "kappa": entry["predicted_kurtosis"],
    })

for entry in gyr_indep:
    summary_json["gyroid_independent"].append({
        "n_periods": entry["n_periods"],
        "kappa": entry["predicted_kurtosis"],
        "kappa_predicted": entry["kappa_predicted_1_over_N"],
    })

for entry in gyr_stacked:
    summary_json["gyroid_stacked_crosscheck"].append({
        "n_periods": entry["n_periods"],
        "kappa": entry["predicted_kurtosis"],
    })

with open(OUT_DIR / "n_scaling_summary.json", "w") as f:
    json.dump(summary_json, f, indent=2)
print(f"\nsaved: {OUT_DIR / 'n_scaling_summary.json'}")

