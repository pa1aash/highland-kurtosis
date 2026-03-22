import numpy as np
from scipy import stats, optimize
from pathlib import Path
import json
import argparse
import warnings

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("WARNING: matplotlib not available — skipping plots")

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False
    print("WARNING: uproot not available — trying ROOT")
    try:
        import ROOT
        HAS_ROOT = True
    except ImportError:
        HAS_ROOT = False
        print("ERROR: Neither uproot nor ROOT available. Install: pip install uproot")

PLA_X0_MM = 315.0
PLA_X0_CM = 31.5

def highland_sigma_rad(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1.0 + 0.038 * np.log(x_over_X0))

def excess_kurtosis(data):
    return stats.kurtosis(data, fisher=True, bias=False)

def rms_width(data):
    return np.std(data, ddof=1)

def core_gaussian_width(data, central_fraction=0.98):
    lo = np.percentile(data, (1 - central_fraction) / 2 * 100)
    hi = np.percentile(data, (1 + central_fraction) / 2 * 100)
    core = data[(data >= lo) & (data <= hi)]
    return np.std(core, ddof=1)

def tail_fraction(data, n_sigma=3):
    sigma = np.std(data, ddof=1)
    mean = np.mean(data)
    n_tail = np.sum(np.abs(data - mean) > n_sigma * sigma)
    return n_tail / len(data)

def two_gaussian_mixture_pdf(x, mu, sigma1, sigma2, w):
    g1 = stats.norm.pdf(x, mu, sigma1)
    g2 = stats.norm.pdf(x, mu, sigma2)
    return (1 - w) * g1 + w * g2

def fit_two_gaussian(data, n_bins=200):
    hist, bin_edges = np.histogram(data, bins=n_bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    mu = np.mean(data)
    sigma_est = np.std(data)

    def single_gauss(x, s):
        return stats.norm.pdf(x, mu, s)

    try:
        popt_single, _ = optimize.curve_fit(single_gauss, bin_centers, hist,
                                             p0=[sigma_est], maxfev=5000)
        single_pred = single_gauss(bin_centers, *popt_single)
        single_chi2 = np.sum((hist - single_pred)**2 / (single_pred + 1e-20))
        single_ndf = n_bins - 1
        single_chi2_ndf = single_chi2 / single_ndf
    except Exception:
        single_chi2_ndf = np.inf

    def mix_gauss(x, s1, s2, w):
        return two_gaussian_mixture_pdf(x, mu, s1, s2, w)

    try:
        popt, pcov = optimize.curve_fit(
            mix_gauss, bin_centers, hist,
            p0=[sigma_est * 0.8, sigma_est * 2.0, 0.1],
            bounds=([sigma_est * 0.1, sigma_est * 0.5, 0.001],
                    [sigma_est * 2.0, sigma_est * 10.0, 0.5]),
            maxfev=10000)

        sigma_core, sigma_tail, weight = popt
        mix_pred = mix_gauss(bin_centers, *popt)
        mix_chi2 = np.sum((hist - mix_pred)**2 / (mix_pred + 1e-20))
        mix_ndf = n_bins - 3
        mix_chi2_ndf = mix_chi2 / mix_ndf

        if sigma_core > sigma_tail:
            sigma_core, sigma_tail = sigma_tail, sigma_core
            weight = 1.0 - weight

        return sigma_core, sigma_tail, weight, mix_chi2_ndf, single_chi2_ndf

    except Exception as e:
        warnings.warn(f"Two-Gaussian fit failed: {e}")
        return sigma_est, sigma_est * 2, 0.0, np.inf, single_chi2_ndf

def load_root_file(filepath):
    filepath = Path(filepath)

    if HAS_UPROOT:
        f = uproot.open(str(filepath))
        tree = None
        for name in ["scattering", "ntuple", "tree", "T"]:
            if name in f:
                tree = f[name]
                break
        if tree is None:
            for key in f.keys():
                obj = f[key]
                if hasattr(obj, 'arrays'):
                    tree = obj
                    break

        if tree is None:
            raise ValueError(f"No TTree found in {filepath}")

        arrays = tree.arrays(library="np")
        return arrays

    elif HAS_ROOT:
        f = ROOT.TFile.Open(str(filepath))
        tree = f.Get("scattering")
        if not tree:
            tree = f.Get("ntuple")

        data = {
            "theta_x": [], "theta_y": [], "theta_space": [],
            "energy_out": [], "entry_x": [], "entry_y": [], "pla_path": []
        }
        for event in tree:
            data["theta_x"].append(event.theta_x)
            data["theta_y"].append(event.theta_y)
            data["theta_space"].append(event.theta_space)
            data["energy_out"].append(event.energy_out)
            data["entry_x"].append(event.entry_x)
            data["entry_y"].append(event.entry_y)
            data["pla_path"].append(event.pla_path)

        return {k: np.array(v) for k, v in data.items()}

    else:
        raise RuntimeError("No ROOT file reader available")

def load_csv_file(filepath):
    data = np.loadtxt(filepath, delimiter=',', skiprows=1)
    return {
        "theta_x": data[:, 0],
        "theta_y": data[:, 1],
        "theta_space": data[:, 2],
        "energy_out": data[:, 3],
        "entry_x": data[:, 4],
        "entry_y": data[:, 5],
        "pla_path": data[:, 6],
    }

def analyze_config(data, config_name="", beam_energy_gev=4.0, sample_thick_mm=10.0):
    theta_x = data["theta_x"]
    theta_y = data["theta_y"]
    n_events = len(theta_x)

    if n_events == 0:
        return {"error": "No events", "config": config_name}

    sigma_x = rms_width(theta_x)
    sigma_y = rms_width(theta_y)

    kappa_x = excess_kurtosis(theta_x)
    kappa_y = excess_kurtosis(theta_y)

    kappa_err = np.sqrt(24.0 / n_events)

    sigma_core_x = core_gaussian_width(theta_x)

    f_tail_x = tail_fraction(theta_x, 3)
    gaussian_tail = 2 * stats.norm.sf(3)

    s_core, s_tail, w_tail, chi2_mix, chi2_single = fit_two_gaussian(theta_x)

    if "pla_path" in data and len(data["pla_path"]) > 0:
        mean_pla_path = np.mean(data["pla_path"])
        mean_xX0 = mean_pla_path / PLA_X0_MM
    else:
        mean_xX0 = sample_thick_mm / PLA_X0_MM

    highland_pred = highland_sigma_rad(mean_xX0, beam_energy_gev)

    kappa_significance = abs(kappa_x) / kappa_err if kappa_err > 0 else 0

    result = {
        "config": config_name,
        "n_events": n_events,
        "beam_energy_GeV": beam_energy_gev,
        "sigma_x_urad": round(sigma_x * 1e6, 2),
        "sigma_y_urad": round(sigma_y * 1e6, 2),
        "sigma_core_urad": round(sigma_core_x * 1e6, 2),
        "highland_pred_urad": round(highland_pred * 1e6, 2),
        "sigma_over_highland": round(sigma_x / highland_pred, 4) if highland_pred > 0 else None,
        "kurtosis_x": round(kappa_x, 4),
        "kurtosis_y": round(kappa_y, 4),
        "kurtosis_err": round(kappa_err, 4),
        "kurtosis_significance_sigma": round(kappa_significance, 1),
        "tail_fraction_3sigma": round(f_tail_x, 5),
        "gaussian_tail_3sigma": round(gaussian_tail, 5),
        "excess_tail_fraction": round(f_tail_x - gaussian_tail, 5),
        "fit_sigma_core_urad": round(s_core * 1e6, 2),
        "fit_sigma_tail_urad": round(s_tail * 1e6, 2),
        "fit_tail_weight": round(w_tail, 4),
        "chi2_ndf_single_gauss": round(chi2_single, 2),
        "chi2_ndf_two_gauss": round(chi2_mix, 2),
        "mean_pla_path_mm": round(np.mean(data.get("pla_path", [0])), 3),
        "mean_x_over_X0": round(mean_xX0, 5),
    }

    return result

def position_resolved_analysis(data, bin_size_mm=1.0):
    entry_x = data["entry_x"]
    entry_y = data["entry_y"]
    theta_x = data["theta_x"]

    x_min, x_max = np.percentile(entry_x, [1, 99])
    y_min, y_max = np.percentile(entry_y, [1, 99])
    x_bins = np.arange(x_min, x_max + bin_size_mm, bin_size_mm)
    y_bins = np.arange(y_min, y_max + bin_size_mm, bin_size_mm)

    sigma_map = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)
    kappa_map = np.full((len(x_bins)-1, len(y_bins)-1), np.nan)

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            mask = ((entry_x >= x_bins[i]) & (entry_x < x_bins[i+1]) &
                    (entry_y >= y_bins[j]) & (entry_y < y_bins[j+1]))
            if np.sum(mask) >= 100:
                sigma_map[i, j] = np.std(theta_x[mask]) * 1e6
                if np.sum(mask) >= 1000:
                    kappa_map[i, j] = stats.kurtosis(theta_x[mask], fisher=True)

    return x_bins, y_bins, sigma_map, kappa_map

def plot_figure1_kurtosis_vs_infill(results, output_path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    geometries = {}
    for r in results:
        geom = r.get("config", "").split("_")[0] if "_" in r.get("config", "") else "unknown"
        parts = r.get("config", "").split("_")
        infill = None
        for p in parts:
            if p.endswith("pct"):
                try:
                    infill = int(p.replace("pct", ""))
                except ValueError:
                    pass
        if infill is None:
            continue

        if geom not in geometries:
            geometries[geom] = {"infill": [], "kurtosis": [], "error": []}
        geometries[geom]["infill"].append(infill)
        geometries[geom]["kurtosis"].append(r["kurtosis_x"])
        geometries[geom]["error"].append(r["kurtosis_err"])

    colors = {"rectilinear": "C0", "honeycomb": "C1", "gyroid": "C2",
              "cubic": "C3", "voronoi": "C4"}
    markers = {"rectilinear": "s", "honeycomb": "^", "gyroid": "o",
               "cubic": "D", "voronoi": "v"}

    for geom, data in sorted(geometries.items()):
        idx = np.argsort(data["infill"])
        x = np.array(data["infill"])[idx]
        y = np.array(data["kurtosis"])[idx]
        e = np.array(data["error"])[idx]
        ax.errorbar(x, y, yerr=e, fmt=markers.get(geom, 'o') + '-',
                    color=colors.get(geom, 'gray'), label=geom.capitalize(),
                    capsize=3, markersize=8)

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5,
               label='Gaussian (Highland)')
    ax.set_xlabel('Infill Percentage (%)', fontsize=14)
    ax.set_ylabel('Excess Kurtosis κ', fontsize=14)
    ax.set_title('Departure from Highland Gaussian Prediction\n'
                 'in 3D-Printed PLA Lattices', fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 105)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Figure 1 saved: {output_path}")

def plot_figure2_universality(results, sweep0_path, output_path):
    if not HAS_MPL:
        return

    try:
        with open(sweep0_path) as f:
            sweep0 = json.load(f)
    except FileNotFoundError:
        print("  Sweep 0 data not found — skipping Figure 2")
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for r in results:
        config = r.get("config", "")
        geom = config.split("_")[0] if "_" in config else "unknown"

        var_xX0 = None
        for s in sweep0:
            if s.get("geometry") == geom:
                infill_match = False
                for p in config.split("_"):
                    if p.endswith("pct"):
                        try:
                            if int(p.replace("pct", "")) == s.get("infill_target_pct"):
                                infill_match = True
                        except ValueError:
                            pass
                if infill_match:
                    var_xX0 = s.get("var_x_over_X0", None)
                    break

        if var_xX0 is not None and var_xX0 > 0:
            colors = {"rectilinear": "C0", "honeycomb": "C1", "gyroid": "C2",
                      "cubic": "C3", "voronoi": "C4"}
            ax.errorbar(var_xX0, r["kurtosis_x"], yerr=r["kurtosis_err"],
                        fmt='o', color=colors.get(geom, 'gray'),
                        label=geom.capitalize(), markersize=10, capsize=3)

    ax.set_xlabel('Column Density Variance, Var(x/X₀)', fontsize=14)
    ax.set_ylabel('Excess Kurtosis κ', fontsize=14)
    ax.set_title('Universality Test: κ vs Column-Density Variance', fontsize=16)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=12)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Figure 2 saved: {output_path}")

def plot_figure6_distribution_shapes(results_by_infill, output_path):
    if not HAS_MPL:
        return

    fig, ax = plt.subplots(figsize=(10, 7))

    for infill, data in sorted(results_by_infill.items()):
        theta = data["theta_x"]
        sigma = np.std(theta)
        if sigma == 0:
            continue

        theta_norm = theta / sigma

        hist, edges = np.histogram(theta_norm, bins=200, range=(-5, 5), density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        ax.semilogy(centers, hist, label=f'{infill}% infill', linewidth=1.5)

    x_ref = np.linspace(-5, 5, 500)
    ax.semilogy(x_ref, stats.norm.pdf(x_ref), 'k--', linewidth=2,
                label='Gaussian', alpha=0.7)

    ax.set_xlabel('θ / σ_θ (normalized)', fontsize=14)
    ax.set_ylabel('Probability Density', fontsize=14)
    ax.set_title('Angular Distribution Shape vs Infill\n'
                 '(normalized to same RMS width)', fontsize=16)
    ax.legend(fontsize=11)
    ax.set_xlim(-5, 5)
    ax.set_ylim(1e-5, 1)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"  Figure 6 saved: {output_path}")

def plot_position_map(x_bins, y_bins, sigma_map, kappa_map, config_name, output_path):
    if not HAS_MPL:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    im1 = ax1.pcolormesh(x_bins, y_bins, sigma_map.T, cmap='viridis')
    ax1.set_xlabel('Entry X (mm)')
    ax1.set_ylabel('Entry Y (mm)')
    ax1.set_title(f'σ_θ map (μrad)\n{config_name}')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='σ_θ (μrad)')

    im2 = ax2.pcolormesh(x_bins, y_bins, kappa_map.T, cmap='RdBu_r',
                          vmin=-0.5, vmax=2.0)
    ax2.set_xlabel('Entry X (mm)')
    ax2.set_ylabel('Entry Y (mm)')
    ax2.set_title(f'Excess kurtosis κ map\n{config_name}')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='κ')

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)

def parse_config_name(filepath):
    name = Path(filepath).stem
    parts = name.split("_")

    info = {"name": name, "geometry": "", "infill": 0, "energy": 0}

    for i, p in enumerate(parts):
        if p in ["rectilinear", "honeycomb", "gyroid", "cubic", "voronoi", "solid", "air"]:
            info["geometry"] = p
        if p.endswith("pct"):
            try:
                info["infill"] = int(p.replace("pct", ""))
            except ValueError:
                pass
        if p.endswith("GeV"):
            try:
                info["energy"] = float(p.replace("GeV", ""))
            except ValueError:
                pass

    return info

def main():
    parser = argparse.ArgumentParser(description="MCS Highland Analysis Pipeline")
    parser.add_argument("--input-dir", type=str, help="Directory with ROOT files")
    parser.add_argument("--single-file", type=str, help="Single ROOT file to analyze")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sweep0", type=str, default=None,
                        help="Path to sweep0_summary.json")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    if args.single_file:
        files = [Path(args.single_file)]
    elif args.input_dir:
        input_dir = Path(args.input_dir)
        files = sorted(input_dir.glob("*.root"))
        if not files:
            files = sorted(input_dir.glob("*.csv"))
    else:
        print("ERROR: Specify --input-dir or --single-file")
        return

    print(f"Found {len(files)} data files")

    for filepath in files:
        print(f"\nAnalyzing: {filepath.name}")
        config = parse_config_name(filepath)

        try:
            if filepath.suffix == ".root":
                data = load_root_file(filepath)
            else:
                data = load_csv_file(filepath)
        except Exception as e:
            print(f"  ERROR loading: {e}")
            continue

        result = analyze_config(
            data,
            config_name=config["name"],
            beam_energy_gev=config["energy"] if config["energy"] > 0 else 4.0,
        )
        all_results.append(result)

        print(f"  N = {result['n_events']}")
        print(f"  σ_x = {result['sigma_x_urad']:.1f} μrad "
              f"(Highland: {result['highland_pred_urad']:.1f} μrad, "
              f"ratio: {result.get('sigma_over_highland', 'N/A')})")
        print(f"  κ = {result['kurtosis_x']:.4f} ± {result['kurtosis_err']:.4f} "
              f"({result['kurtosis_significance_sigma']:.1f}σ)")
        print(f"  Tail fraction (3σ): {result['tail_fraction_3sigma']*100:.3f}% "
              f"(Gaussian: {result['gaussian_tail_3sigma']*100:.3f}%)")
        print(f"  2-Gauss fit: σ_core={result['fit_sigma_core_urad']:.1f}, "
              f"σ_tail={result['fit_sigma_tail_urad']:.1f}, "
              f"w={result['fit_tail_weight']:.3f}")
        print(f"  χ²/ndf: single={result['chi2_ndf_single_gauss']:.1f}, "
              f"mixture={result['chi2_ndf_two_gauss']:.1f}")

        if result['n_events'] > 10000 and "entry_x" in data:
            try:
                xb, yb, sm, km = position_resolved_analysis(data)
                map_path = output_dir / f"map_{config['name']}.png"
                plot_position_map(xb, yb, sm, km, config['name'], map_path)
            except Exception as e:
                print(f"  Position map failed: {e}")

    results_file = output_dir / "analysis_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    if len(all_results) >= 3 and HAS_MPL:
        print("\nGenerating publication figures...")

        plot_figure1_kurtosis_vs_infill(
            all_results, output_dir / "fig1_kurtosis_vs_infill.png")

        sweep0_path = args.sweep0 or (Path(__file__).parent.parent / "data" / "sweep0" / "sweep0_summary.json")
        plot_figure2_universality(
            all_results, sweep0_path, output_dir / "fig2_universality.png")

    print(f"{'Config':<30} {'N':>8} {'σ_x(μrad)':>10} {'Highland':>10} "
          f"{'Ratio':>8} {'κ':>8} {'κ_sig':>6} {'Tail%':>8}")
    for r in all_results:
        print(f"{r['config']:<30} {r['n_events']:>8} "
              f"{r['sigma_x_urad']:>10.1f} {r['highland_pred_urad']:>10.1f} "
              f"{r.get('sigma_over_highland', 0):>8.3f} "
              f"{r['kurtosis_x']:>8.4f} {r['kurtosis_significance_sigma']:>6.1f} "
              f"{r['tail_fraction_3sigma']*100:>7.3f}%")

if __name__ == "__main__":
    main()

