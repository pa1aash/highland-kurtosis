import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['STIX Two Text', 'Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'text.usetex': False,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'axes.titlepad': 12,
    'legend.fontsize': 10,
    'legend.labelspacing': 0.45,
    'legend.handlelength': 2.0,
    'legend.handletextpad': 0.5,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 400,
    'axes.linewidth': 1.1,
    'xtick.major.size': 6,
    'ytick.major.size': 6,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 0.9,
    'ytick.major.width': 0.9,
    'xtick.minor.width': 0.5,
    'ytick.minor.width': 0.5,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'lines.linewidth': 1.6,
    'lines.markersize': 7,
    'errorbar.capsize': 3,
    'axes.prop_cycle': plt.cycler('color', [
        '#c44e52', '#4c72b0', '#55a868', '#dd8452', '#8172b3',
        '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd'
    ]),
})

FIG_W, FIG_H = 7.0, 5.0

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "proposal_figures"
OUT.mkdir(exist_ok=True)

C_RECT  = '#c44e52'
C_HC    = '#dd8452'
C_GYR   = '#4c72b0'
C_CUB   = '#55a868'
C_VOR   = '#8172b3'
C_SOLID = '#2d2d2d'
C_THEORY = '#666666'

def figure_4():
    with open(BASE / "data/sweep0/n_scaling_summary.json") as f:
        rect_data = json.load(f)
    with open(BASE / "data/sweep0/gyroid_independent_summary.json") as f:
        gyr_data = json.load(f)

    rect_20 = [d for d in rect_data
               if d["geometry"] == "stacked_rectilinear" and d["infill_target_pct"] == 20]
    rect_N = np.array([d["n_layers"] for d in rect_20])
    rect_k = np.array([d["predicted_kurtosis"] for d in rect_20])

    gyr_ind = gyr_data["independent"]
    gyr_N = np.array([d["n_periods"] for d in gyr_ind])
    gyr_k = np.array([d["predicted_kurtosis"] for d in gyr_ind])

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    N_ref = np.logspace(0, 2.2, 300)
    ax.plot(N_ref, rect_k[0] / N_ref, '--', color=C_RECT, linewidth=1.0,
            alpha=0.35, zorder=0)
    ax.plot(N_ref, gyr_k[0] / N_ref, '--', color=C_GYR, linewidth=1.0,
            alpha=0.35, zorder=0)
    ax.plot([], [], '--', color='#999999', linewidth=1.0,
            label=r'$\kappa_1 / N$ reference')

    ax.plot(rect_N, rect_k, 'o-', color=C_RECT, markersize=8,
            markeredgecolor='white', markeredgewidth=1.0, linewidth=1.4,
            zorder=4, label='Rectilinear 20%')

    ax.plot(gyr_N, gyr_k, 's-', color=C_GYR, markersize=7.5,
            markeredgecolor='white', markeredgewidth=1.0, linewidth=1.4,
            zorder=4, label='Gyroid 40% (indep.)')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'Number of independent cells  $N$')
    ax.set_ylabel(r'Geometric excess kurtosis  $\kappa_{\mathrm{geo}}$')
    ax.set_title(r'$\mathbf{1/}$$\boldsymbol{N}$' + '  Scaling Law',
                 fontweight='bold')

    ax.set_xlim(0.7, 150)
    ax.set_ylim(0.008, 25)
    ax.xaxis.set_major_locator(ticker.FixedLocator([1, 2, 5, 10, 20, 50, 100]))
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.legend(loc='lower left', frameon=True, edgecolor='#dddddd',
              facecolor='white', framealpha=0.95, borderpad=0.6)

    plt.tight_layout()
    fig.savefig(OUT / "figure_4.png", bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "figure_4.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Figure 4 saved")

def figure_5():
    import uproot

    data_dir = BASE / "data/proposal"
    configs = [
        ('Solid PLA',        'control_solid_4GeV.root', C_SOLID, 2.2),
        ('Rectilinear 40%',  'rect_40pct_4GeV.root',    C_RECT,  1.6),
        ('Gyroid 40%',       'gyr_40pct_4GeV.root',     C_GYR,   1.6),
    ]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for label, fname, color, lw in configs:
        tree = uproot.open(str(data_dir / fname))["scattering"]
        theta_x = tree["theta_x"].array(library="np")
        energy = tree["energy_out"].array(library="np")
        entry_x = tree["entry_x"].array(library="np")
        entry_y = tree["entry_y"].array(library="np")

        mask = (np.abs(entry_x) < 5.0) & (np.abs(entry_y) < 10.0) & (energy > 3.6)
        theta = theta_x[mask] * 1000

        bins = np.linspace(-2.8, 2.8, 141)
        counts, edges = np.histogram(theta, bins=bins, density=True)
        centers = 0.5 * (edges[:-1] + edges[1:])

        log_counts = np.where(counts > 0, np.log10(counts), -4)
        log_smooth = gaussian_filter1d(log_counts, sigma=1.2)
        smooth = 10**log_smooth
        core = np.abs(centers) < 1.0
        smooth[core] = counts[core]

        ax.plot(centers, smooth, '-', color=color, linewidth=lw,
                label=label, zorder=2, alpha=0.92)

    ax.set_yscale('log')
    ax.set_xlabel(r'Projected scattering angle  $\theta_x$  (mrad)')
    ax.set_ylabel(r'Probability density  (mrad$^{-1}$)')
    ax.set_title('Angular Distributions at 4 GeV (Geant4)', fontweight='bold')

    ax.set_xlim(-2.8, 2.8)
    ax.set_ylim(8e-4, 20)

    ax.legend(loc='upper right', frameon=True, edgecolor='#dddddd',
              facecolor='white', framealpha=0.95, borderpad=0.6)

    plt.tight_layout()
    fig.savefig(OUT / "figure_5.png", bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "figure_5.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Figure 5 saved")

def figure_6():
    with open(BASE / "results/proposal/proposal_summary.json") as f:
        data = json.load(f)

    fid = data["fiducial_cut"]
    k_solid = fid["control_solid_4GeV"]["kappa_avg"]
    k_solid_err = fid["control_solid_4GeV"]["kappa_err_boot"]

    geom_list = [
        ('rect', 'o', C_RECT,  'Rectilinear', 1.0),
        ('hc',   'D', C_HC,    'Honeycomb',    0.9),
        ('gyr',  's', C_GYR,   'Gyroid',       0.9),
        ('cub',  '^', C_CUB,   '3D Grid',      0.9),
        ('vor',  'v', C_VOR,   'Voronoi',      0.9),
    ]
    infills = [20, 40, 60, 80]

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    f_arr = np.linspace(0.12, 0.92, 300)
    theory = 3 * (1 - f_arr) / f_arr
    ax.fill_between(f_arr * 100, theory * 0.88, theory * 1.12,
                    color=C_THEORY, alpha=0.10, zorder=0)
    ax.plot(f_arr * 100, theory, '-', color=C_THEORY, linewidth=1.4,
            zorder=1, label=r'$3(1 - f) / f$')

    for prefix, marker, color, label, ms_scale in geom_list:
        x, y, ye = [], [], []
        for pct in infills:
            key = f"{prefix}_{pct}pct_4GeV"
            if key not in fid:
                continue
            dk = fid[key]["kappa_avg"] - k_solid
            dk_err = np.sqrt(fid[key]["kappa_err_boot"]**2 + k_solid_err**2)
            x.append(pct)
            y.append(dk)
            ye.append(dk_err)
        if x:
            ax.errorbar(x, y, yerr=ye, fmt=marker, color=color,
                        markersize=8 * ms_scale,
                        markeredgecolor='white', markeredgewidth=0.8,
                        linewidth=0, elinewidth=1.0, capsize=3, capthick=0.8,
                        label=label, zorder=3)
            ax.plot(x, y, '-', color=color, linewidth=0.9, alpha=0.45, zorder=2)

    ax.axhline(0, color='k', linewidth=0.6, alpha=0.2)

    ax.set_xlabel(r'Infill fraction  $f$  (%)')
    ax.set_ylabel(
        r'Excess kurtosis  $\Delta\kappa = \kappa_{\mathrm{lattice}}'
        r' - \kappa_{\mathrm{solid}}$')
    ax.set_title('Excess Kurtosis vs. Infill at 4 GeV', fontweight='bold')

    ax.set_xlim(12, 88)
    ax.set_ylim(-0.5, 15)
    ax.set_xticks([20, 40, 60, 80])
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.legend(loc='upper right', frameon=True, edgecolor='#dddddd',
              facecolor='white', framealpha=0.95, ncol=2,
              columnspacing=1.2, borderpad=0.6)

    plt.tight_layout()
    fig.savefig(OUT / "figure_6.png", bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "figure_6.pdf", bbox_inches='tight', facecolor='white')
    plt.close()
    print("  Figure 6 saved")

if __name__ == "__main__":
    print("Generating proposal figures...\n")
    figure_4()
    figure_5()
    figure_6()
    print(f"\nAll figures saved to {OUT}/")

