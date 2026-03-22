import numpy as np
import matplotlib.pyplot as plt
import uproot
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
})

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "proposal_figures"
OUT.mkdir(exist_ok=True)

def highland_sigma(ell_mm, p_gev=4.0, X0_mm=315.0):
    if ell_mm <= 0:
        return 0.0
    x_over_X0 = ell_mm / X0_mm
    theta0 = (13.6 / (p_gev * 1000)) * np.sqrt(x_over_X0) * \
             (1 + 0.038 * np.log(x_over_X0))
    return theta0 * 1000

def gaussian(x, sigma):
    if sigma <= 0:
        return np.zeros_like(x)
    return np.exp(-0.5 * (x / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)

def moliere_2gauss(x, sigma, w=0.02, r=3.2):
    if sigma <= 0:
        return np.zeros_like(x)
    sigma_core = sigma / np.sqrt((1 - w) + w * r**2)
    sigma_tail = r * sigma_core
    return (1 - w) * gaussian(x, sigma_core) + w * gaussian(x, sigma_tail)

def main():
    f = 0.40
    T = 10.0
    p = 4.0

    sigma_PLA = highland_sigma(T, p)
    sigma_avg = highland_sigma(f * T, p)
    sigma_air = 0.015

    kappa_geo = 3 * (1 - f) / f

    print(f"Highland parameters at {p} GeV:")
    print(f"  σ_PLA (ℓ={T}mm)    = {sigma_PLA:.4f} mrad")
    print(f"  σ_avg (ℓ={f*T}mm)  = {sigma_avg:.4f} mrad")
    print(f"  κ_geo              = {kappa_geo:.1f}")

    root_file = BASE / "data/proposal/rect_40pct_4GeV.root"
    tree = uproot.open(str(root_file))["scattering"]
    theta_x = tree["theta_x"].array(library="np")
    energy = tree["energy_out"].array(library="np")
    entry_x = tree["entry_x"].array(library="np")
    entry_y = tree["entry_y"].array(library="np")

    mask = (np.abs(entry_x) < 5.0) & (np.abs(entry_y) < 10.0) & (energy > 3.6)
    theta_g4 = theta_x[mask] * 1000
    print(f"  Geant4 events (after cuts): {np.sum(mask)}")

    bins = np.linspace(-2.5, 2.5, 201)
    counts_g4, edges = np.histogram(theta_g4, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])

    log_g4 = np.where(counts_g4 > 0, np.log10(counts_g4), -4)
    log_smooth = gaussian_filter1d(log_g4, sigma=1.5)
    g4_smooth = 10**log_smooth
    core = np.abs(centers) < 0.8
    g4_smooth[core] = counts_g4[core]

    theta = np.linspace(-2.5, 2.5, 2001)

    P_highland = gaussian(theta, sigma_avg)

    P_air = gaussian(theta, sigma_air)
    P_PLA = gaussian(theta, sigma_PLA)
    P_gauss_mix = (1 - f) * P_air + f * P_PLA

    M_air = moliere_2gauss(theta, sigma_air)
    M_PLA = moliere_2gauss(theta, sigma_PLA)
    P_moliere_mix = (1 - f) * M_air + f * M_PLA

    C_HIGHLAND = '#888888'
    C_OURS     = '#c44e52'
    C_G4       = '#2d2d2d'

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    ax.plot(centers, g4_smooth, '-', color=C_G4, linewidth=2.4,
            label='Geant4 simulation', zorder=5)

    ax.plot(theta, P_gauss_mix, '-', color=C_OURS, linewidth=2.0,
            label=r'Our prediction: $P = (1{-}f)\,G_{\mathrm{air}} + f\,G_{\mathrm{PLA}}$',
            zorder=4)

    ax.plot(theta, P_highland, '--', color=C_HIGHLAND, linewidth=2.0,
            label=r'Highland (same-mass): single Gaussian',
            zorder=2)

    ax.set_yscale('log')
    ax.set_xlabel(r'Projected scattering angle  $\theta_x$  (mrad)')
    ax.set_ylabel(r'Probability density  (mrad$^{-1}$)')
    ax.set_title('Rectilinear 40% at 4 GeV: Three Models Compared',
                 fontweight='bold')

    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(5e-4, 30)

    ax.legend(loc='upper left', frameon=True, edgecolor='#dddddd',
              facecolor='white', framealpha=0.95, borderpad=0.5,
              fontsize=9)

    ax.annotate('Highland misses\nthe heavy tails',
                xy=(1.3, gaussian(1.3, sigma_avg)),
                xytext=(1.7, 0.003),
                fontsize=9.5, color=C_HIGHLAND, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_HIGHLAND, lw=1.2))

    ax.annotate(r'$\kappa_{\mathrm{geo}} = 3(1{-}f)/f = 4.5$',
                xy=(0.95, P_gauss_mix[np.argmin(np.abs(theta - 0.95))]),
                xytext=(1.65, 0.12),
                fontsize=10, color=C_OURS, ha='center',
                arrowprops=dict(arrowstyle='->', color=C_OURS, lw=1.2))

    plt.tight_layout()
    fig.savefig(OUT / "theory_same_mass_comparison.png",
                bbox_inches='tight', facecolor='white')
    fig.savefig(OUT / "theory_same_mass_comparison.pdf",
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Saved to {OUT / 'theory_same_mass_comparison.pdf'}")

if __name__ == "__main__":
    main()

