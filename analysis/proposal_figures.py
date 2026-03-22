import json
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT_DIR = Path("results/proposal_figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("results/proposal/proposal_summary.json") as f:
    proposal = json.load(f)
fid = proposal["fiducial_cut"]

with open("data/sweep0/sweep0_summary.json") as f:
    sweep0 = json.load(f)

kappa_solid = {2: fid["control_solid_2GeV"]["kappa_avg"],
               4: fid["control_solid_4GeV"]["kappa_avg"],
               6: fid["control_solid_6GeV"]["kappa_avg"]}

ray = {}
for entry in sweep0:
    key = (entry["geometry"], entry["infill_target_pct"])
    ray[key] = entry["predicted_kurtosis"]

def get_dk(label, energy_gev):
    if label not in fid:
        return None, None
    d = fid[label]
    dk = d["kappa_avg"] - kappa_solid[energy_gev]
    err = d["kappa_err_boot"]
    return dk, err

def make_rect_mask(cell_size, res=0.1):
    halfW = 10.0
    nx = int(20.0 / res)
    x = np.linspace(-halfW + res/2, halfW - res/2, nx)
    XX, YY = np.meshgrid(x, x, indexing='ij')
    n_cells = max(1, int(20.0 / cell_size))
    mask = np.zeros((nx, nx), dtype=bool)
    for i in range(n_cells + 1):
        pos = -halfW + i * cell_size
        pos = max(-halfW + 0.2, min(pos, halfW - 0.2))
        mask |= (np.abs(XX - pos) < 0.2)
        mask |= (np.abs(YY - pos) < 0.2)
    return mask, x

def make_hc_mask(d, res=0.1):
    halfW = 10.0
    a = d / np.sqrt(3.0)
    halfA = a / 2.0
    cos60, sin60 = 0.5, np.sqrt(3.0) / 2.0
    cos120, sin120 = -0.5, np.sqrt(3.0) / 2.0
    colSp, rowSp = 1.5 * a, d
    nx = int(20.0 / res)
    x = np.linspace(-halfW + res/2, halfW - res/2, nx)
    XX, YY = np.meshgrid(x, x, indexing='ij')
    mask = np.zeros((nx, nx), dtype=bool)
    nc = int(20.0 / colSp) + 4
    nr = int(20.0 / rowSp) + 4
    for c in range(-nc, nc + 1):
        cx = c * colSp
        yOff = rowSp * 0.5 if (abs(c) % 2) else 0.0
        for r in range(-nr, nr + 1):
            cy = r * rowSp + yOff
            dx1, dy1 = XX - cx, YY - (cy + d * 0.5)
            h1 = (np.abs(dx1) < halfA) & (np.abs(dy1) < 0.2)
            dx2, dy2 = XX - (cx + 0.75*a), YY - (cy + d*0.25)
            lx2 = cos120*dx2 + sin120*dy2
            ly2 = -sin120*dx2 + cos120*dy2
            h2 = (np.abs(lx2) < halfA) & (np.abs(ly2) < 0.2)
            dx3, dy3 = XX - (cx + 0.75*a), YY - (cy - d*0.25)
            lx3 = cos60*dx3 + sin60*dy3
            ly3 = -sin60*dx3 + cos60*dy3
            h3 = (np.abs(lx3) < halfA) & (np.abs(ly3) < 0.2)
            mask |= h1 | h2 | h3
    mask &= (np.abs(XX) <= halfW) & (np.abs(YY) <= halfW)
    return mask, x

def make_gyr_xy(cell_size, threshold, z_val=0.0, res=0.1):
    halfW = 10.0
    nx = int(20.0 / res)
    x = np.linspace(-halfW + res/2, halfW - res/2, nx)
    k = 2.0 * np.pi / cell_size
    XX, YY = np.meshgrid(x, x, indexing='ij')
    F = (np.sin(k*XX)*np.cos(k*YY) + np.sin(k*YY)*np.cos(k*z_val) +
         np.sin(k*z_val)*np.cos(k*XX))
    return np.abs(F) < threshold, x

def fig1_geometries():
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    zoom = (-4, 4, -4, 4)
    cmap = 'YlOrBr'

    configs = [
        ("(a) Rectilinear 20%", lambda: make_rect_mask(3.794)),
        ("(b) Rectilinear 40%", lambda: make_rect_mask(1.778)),
        ("(c) Honeycomb 40%", lambda: make_hc_mask(1.76914)),
        ("(d) Gyroid 20%  (z = 0)", lambda: make_gyr_xy(7.0, 0.305398)),
        ("(e) Gyroid 40%  (z = 0)", lambda: make_gyr_xy(4.5, 0.61701)),
        ("(f) Gyroid 40%  (z = 2.25 mm)", lambda: make_gyr_xy(4.5, 0.61701, 2.25)),
    ]

    for idx, (title, gen_fn) in enumerate(configs):
        r, c = divmod(idx, 3)
        ax = axes[r, c]
        mask, x = gen_fn()
        ext = [x[0] - 0.05, x[-1] + 0.05, x[0] - 0.05, x[-1] + 0.05]
        ax.imshow(mask.T, origin='lower', cmap=cmap, extent=ext,
                  aspect='equal', interpolation='nearest')
        ax.set_xlim(zoom[0], zoom[1])
        ax.set_ylim(zoom[2], zoom[3])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('x (mm)', fontsize=9)
        ax.set_ylabel('y (mm)', fontsize=9)
        ax.tick_params(labelsize=8)

    fig.suptitle('Figure 1: Sample geometries (beam enters page, XY cross-section)',
                 fontsize=13, fontweight='bold', y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = OUT_DIR / "fig1_geometries.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")

def fig2_kappa_vs_infill():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    infills = [20, 40, 60]

    geoms = {
        "Rectilinear": ("rectilinear", '#377eb8', 'o', '-'),
        "Honeycomb":   ("honeycomb",   '#e41a1c', 's', '-'),
        "Gyroid":      ("gyroid",      '#4daf4a', 'D', '-'),
    }

    for label, (geom, color, marker, ls) in geoms.items():
        kvals = [ray.get((geom, inf), 0) for inf in infills]
        ax1.plot(infills, kvals, marker=marker, linestyle=ls, color=color,
                 markersize=8, linewidth=2, label=label)

    f_arr = np.linspace(0.15, 0.65, 100)
    ax1.plot(f_arr * 100, 3 * (1 - f_arr) / f_arr, '--', color='gray',
             linewidth=1.5, alpha=0.7, label=r'Analytic: $3(1-f)/f$')

    ax1.set_xlabel('Infill (%)', fontsize=12)
    ax1.set_ylabel(r'Predicted excess kurtosis $\kappa$', fontsize=12)
    ax1.set_title(r'(a) Ray-trace prediction (geometry only)', fontsize=12,
                  fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(15, 65)
    ax1.set_ylim(-0.5, 14)

    g4_configs = {
        "Rectilinear": [("rect_20pct_4GeV", 20), ("rect_40pct_4GeV", 40),
                        ("rect_60pct_4GeV", 60)],
        "Honeycomb":   [("hc_20pct_4GeV", 20), ("hc_40pct_4GeV", 40),
                        ("hc_60pct_4GeV", 60)],
        "Gyroid":      [("gyr_20pct_4GeV", 20), ("gyr_40pct_4GeV", 40),
                        ("gyr_60pct_4GeV", 60)],
    }

    for label, (geom, color, marker, ls) in geoms.items():
        configs = g4_configs[label]
        infs, dks, errs = [], [], []
        for cfg, inf in configs:
            dk, err = get_dk(cfg, 4)
            if dk is not None:
                infs.append(inf)
                dks.append(dk)
                errs.append(err)
        ax2.errorbar(infs, dks, yerr=errs, fmt=marker + ls, color=color,
                     markersize=8, linewidth=2, capsize=4, label=label)

    ax2.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)
    ax2.set_xlabel('Infill (%)', fontsize=12)
    ax2.set_ylabel(r'$\Delta\kappa = \kappa_{\rm lattice} - \kappa_{\rm solid}$',
                   fontsize=12)
    ax2.set_title(r'(b) Geant4 simulation, 4 GeV, fiducial cut', fontsize=12,
                  fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(15, 65)
    ax2.set_ylim(-1, 14)

    fig.suptitle(r'Figure 2: Excess kurtosis $\kappa$ vs infill percentage',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig2_kappa_vs_infill.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")

def fig3_energy_dependence():
    fig, ax = plt.subplots(figsize=(8, 5.5))

    energies = [2, 4, 6]

    rect_labels = ["rect_40pct_2GeV", "rect_40pct_4GeV", "rect_40pct_6GeV"]
    rect_dk, rect_err = [], []
    for lbl, e in zip(rect_labels, energies):
        dk, err = get_dk(lbl, e)
        rect_dk.append(dk)
        rect_err.append(err)

    ax.errorbar(energies, rect_dk, yerr=rect_err, fmt='o-', color='#377eb8',
                markersize=10, linewidth=2.5, capsize=5, label='Rectilinear 40% (binary)')

    gyr_labels = ["gyr_40pct_2GeV", "gyr_40pct_4GeV", "gyr_40pct_6GeV"]
    gyr_dk, gyr_err = [], []
    for lbl, e in zip(gyr_labels, energies):
        dk, err = get_dk(lbl, e)
        gyr_dk.append(dk)
        gyr_err.append(err)

    ax.errorbar(energies, gyr_dk, yerr=gyr_err, fmt='D-', color='#4daf4a',
                markersize=10, linewidth=2.5, capsize=5, label='Gyroid 40% (continuous)')

    ax.axhline(ray[("rectilinear", 40)], color='#377eb8', linestyle='--',
               linewidth=1.5, alpha=0.5, label=f'Ray-trace rect: {ray[("rectilinear", 40)]:.1f}')
    ax.axhline(ray[("gyroid", 40)], color='#4daf4a', linestyle='--',
               linewidth=1.5, alpha=0.5, label=f'Ray-trace gyroid: {ray[("gyroid", 40)]:.2f}')

    ax.axhline(0, color='black', linewidth=0.8, linestyle='-', alpha=0.3)

    ax.annotate(r'Gyroid $\Delta\kappa \approx 0$ at 2 GeV' + '\n(Moliere tails mask lattice signal)',
                xy=(2, gyr_dk[0]), xytext=(3.2, -1.5),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='#4daf4a', lw=1.5),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    ax.set_xlabel('Beam energy (GeV)', fontsize=12)
    ax.set_ylabel(r'$\Delta\kappa = \kappa_{\rm lattice} - \kappa_{\rm solid}$', fontsize=12)
    ax.set_title('Figure 3: Energy dependence of excess kurtosis', fontsize=13,
                 fontweight='bold')
    ax.legend(fontsize=9.5, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1.5, 6.5)
    ax.set_ylim(-2.5, 7)
    ax.set_xticks([2, 3, 4, 5, 6])

    fig.tight_layout()
    path = OUT_DIR / "fig3_energy_dependence.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")

def fig4_setup_and_predictions():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5),
                                    gridspec_kw={'width_ratios': [1.2, 1]})

    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 6)
    ax1.set_aspect('equal')
    ax1.axis('off')

    ax1.annotate('', xy=(1.5, 3), xytext=(0.3, 3),
                 arrowprops=dict(arrowstyle='->', color='#e41a1c', lw=3))
    ax1.text(0.1, 3.8, 'e$^-$ beam\n1-6 GeV', fontsize=9, ha='left',
             fontweight='bold', color='#e41a1c')

    for z in [2.0, 2.4, 2.8]:
        ax1.add_patch(plt.Rectangle((z, 1.5), 0.08, 3, facecolor='#377eb8',
                                     edgecolor='black', linewidth=0.5))
    ax1.text(2.4, 0.8, 'Upstream\ntelescope', fontsize=8, ha='center',
             color='#377eb8', fontweight='bold')

    ax1.add_patch(plt.Rectangle((4.0, 1.8), 1.0, 2.4, facecolor='#ff7f00',
                                 edgecolor='black', linewidth=1.5, alpha=0.7))
    ax1.text(4.5, 4.6, '3D-printed\nPLA sample\n20x20x10 mm',
             fontsize=8, ha='center', fontweight='bold', color='#ff7f00')

    for z in [6.2, 6.6, 7.0]:
        ax1.add_patch(plt.Rectangle((z, 1.5), 0.08, 3, facecolor='#377eb8',
                                     edgecolor='black', linewidth=0.5))
    ax1.text(6.6, 0.8, 'Downstream\ntelescope', fontsize=8, ha='center',
             color='#377eb8', fontweight='bold')

    ax1.annotate('', xy=(8.5, 4.2), xytext=(5.2, 3),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                 linestyle='--'))
    ax1.annotate('', xy=(8.5, 1.8), xytext=(5.2, 3),
                 arrowprops=dict(arrowstyle='->', color='gray', lw=1.5,
                                 linestyle='--'))
    ax1.text(8.7, 3.0, r'$\theta_x, \theta_y$' + '\nMCS angles',
             fontsize=9, ha='left', color='gray')

    ax1.text(1.0, 5.7, 'MIMOSA26 pixel telescope (6 planes)',
             fontsize=9, ha='left', style='italic', color='#377eb8')
    ax1.set_title('(a) Experimental setup at DESY', fontsize=12,
                  fontweight='bold')

    ax2.axis('off')

    headers = ['Sample', 'Infill', r'$\kappa_{\rm ray}$',
               r'$\Delta\kappa_{\rm G4}$', r'$\sigma$']
    rows = [
        ['Solid PLA', '100%', '0.00', '0.0 (baseline)', '--'],
        ['Rectilinear', '20%', '12.3', '+12.4 +/- 0.5', '25'],
        ['Rectilinear', '40%', '4.7', '+5.0 +/- 0.4', '13'],
        ['Rectilinear', '60%', '2.2', '+3.3 +/- 0.5', '7'],
        ['Honeycomb', '40%', '4.5', '+5.0 +/- 0.4', '13'],
        ['Gyroid', '20%', '0.93', '+2.2 +/- 0.4', '5'],
        ['Gyroid', '40%', '0.32', '+1.6 +/- 0.4', '4'],
        ['Gyroid', '60%', '0.10', '+1.5 +/- 0.4', '4'],
        ['Slicer rect', '40%', '~0.1', '~0 (N-scaling)', '--'],
    ]

    table = ax2.table(cellText=rows, colLabels=headers, loc='center',
                       cellLoc='center', colColours=['#d4e6f1'] * 5)
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for i in range(1, 4):
        for j in range(5):
            table[i + 1, j].set_facecolor('#dbeaf7')
    for i in range(5, 8):
        for j in range(5):
            table[i + 1, j].set_facecolor('#d5f5d5')

    ax2.set_title('(b) Predicted measurements at 4 GeV', fontsize=12,
                  fontweight='bold')

    fig.suptitle('Figure 4: Experimental concept and predictions',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "fig4_setup_predictions.png"
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"saved: {path}")

def main():
    print("generating proposal figures...\n")
    fig1_geometries()
    fig2_kappa_vs_infill()
    fig3_energy_dependence()
    fig4_setup_and_predictions()
    print(f"\nall figures saved to {OUT_DIR}/")

if __name__ == "__main__":
    main()

