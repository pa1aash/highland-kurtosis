import numpy as np
from scipy import stats, optimize
from pathlib import Path
import json
import sys
import argparse

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import uproot
    HAS_UPROOT = True
except ImportError:
    HAS_UPROOT = False

PLA_X0_MM = 315.0
SAMPLE_T_MM = 10.0
n_cut_sigma_DEFAULT = 10
FIDUCIAL_X_MM_DEFAULT = 5.0
FIDUCIAL_Y_MM_DEFAULT = 10.0
ENERGY_CUT_FRACTION_DEFAULT = 0.9

SIGMA_POS_UM = 2.0
LEVER_ARM_MM = 20.0
N_PLANES_UP = 3
SIGMA_TELESCOPE_MRAD = SIGMA_POS_UM * 1e-3 * np.sqrt(12.0 / (N_PLANES_UP * (N_PLANES_UP**2 - 1))) / LEVER_ARM_MM * 1e3

DISPLAY_NAMES = {
    "rect": "Rectilinear",
    "hc":   "Honeycomb",
    "gyr":  "Gyroid",
    "cub":  "3D Grid",
    "vor":  "Voronoi",
}

ENERGY_SCAN = [2, 4, 6]

def highland_sigma(x_over_X0, p_gev):
    if x_over_X0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_X0) * (1 + 0.038 * np.log(x_over_X0))

def analyze_file(filepath, p_gev=4.0, label="", fiducial=False,
                 energy_cut_fraction=ENERGY_CUT_FRACTION_DEFAULT,
                 n_cut_sigma=n_cut_sigma_DEFAULT,
                 fiducial_x=FIDUCIAL_X_MM_DEFAULT,
                 fiducial_y=FIDUCIAL_Y_MM_DEFAULT):
    f = uproot.open(filepath)
    tree = f["scattering"]
    data = tree.arrays(library="np")

    theta_x = data["theta_x"]
    theta_y = data["theta_y"]
    energy_out = data["energy_out"]
    pla_path = data["pla_path"]
    entry_x = data["entry_x"]
    entry_y = data["entry_y"]
    N_raw = len(theta_x)

    energy_cut = energy_out > (p_gev * energy_cut_fraction)

    if fiducial:
        fid_cut = (np.abs(entry_x) < fiducial_x) & (np.abs(entry_y) < fiducial_y)
        energy_cut = energy_cut & fid_cut

    mean_pla_all = np.mean(pla_path[energy_cut])
    xX0_all = mean_pla_all / PLA_X0_MM
    sigma_highland = highland_sigma(xX0_all, p_gev)

    if sigma_highland > 0:
        angle_max = n_cut_sigma * sigma_highland
    else:
        angle_max = 0.02

    angle_cut = (np.abs(theta_x) < angle_max) & (np.abs(theta_y) < angle_max)
    mask = energy_cut & angle_cut

    n_after_energy = np.sum(energy_cut)
    n_after_all = np.sum(mask)

    theta_x = theta_x[mask]
    theta_y = theta_y[mask]
    pla_path_cut = pla_path[mask]

    f_hit_data = np.mean(pla_path_cut > 0.1)
    f_intermediate = np.mean((pla_path_cut > 0.1) & (pla_path_cut < SAMPLE_T_MM - 0.5))
    N = len(theta_x)
    n_cut = N_raw - N

    print(f"    {label}: {N_raw} -> {n_after_energy} (energy"
          f"{'+fiducial' if fiducial else ''}) -> {N} (angle) = {n_cut} removed")

    if N < 100:
        return None

    sigma_x = np.std(theta_x, ddof=1)
    sigma_y = np.std(theta_y, ddof=1)
    kappa_x = stats.kurtosis(theta_x, fisher=True, bias=False)
    kappa_y = stats.kurtosis(theta_y, fisher=True, bias=False)
    kappa_avg = (kappa_x + kappa_y) / 2
    kappa_err = np.sqrt(24.0 / N)

    n_boot = 1000
    rng = np.random.default_rng(42)
    boot_kappas = np.zeros(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        bk_x = stats.kurtosis(theta_x[idx], fisher=True, bias=False)
        bk_y = stats.kurtosis(theta_y[idx], fisher=True, bias=False)
        boot_kappas[b] = (bk_x + bk_y) / 2
    kappa_err_boot = np.std(boot_kappas, ddof=1)
    kappa_ci_lo = np.percentile(boot_kappas, 2.5)
    kappa_ci_hi = np.percentile(boot_kappas, 97.5)

    mean_pla = np.mean(pla_path_cut)
    xX0 = mean_pla / PLA_X0_MM

    f_tail = np.mean(np.abs(theta_x - np.mean(theta_x)) > 3 * sigma_x)
    f_gauss = 2 * stats.norm.sf(3)

    s1, s2, w = sigma_x, sigma_x, 0
    try:
        hist, edges = np.histogram(theta_x, bins=200, density=True)
        centers = (edges[:-1] + edges[1:]) / 2
        mu = np.mean(theta_x)

        def mix(x, s1_, s2_, w_):
            return (1-w_)*stats.norm.pdf(x, mu, s1_) + w_*stats.norm.pdf(x, mu, s2_)

        popt, _ = optimize.curve_fit(mix, centers, hist,
                                      p0=[sigma_x*0.8, sigma_x*2, 0.1],
                                      bounds=([0.01*sigma_x, 0.5*sigma_x, 0.001],
                                              [2*sigma_x, 10*sigma_x, 0.5]),
                                      maxfev=10000)
        s1, s2, w = popt
        if s1 > s2:
            s1, s2 = s2, s1
            w = 1 - w
    except Exception:
        pass

    return {
        "label": label,
        "N_raw": N_raw,
        "N": N,
        "n_cut": n_cut,
        "angle_cut_mrad": angle_max * 1e3,
        "sigma_x_urad": sigma_x * 1e6,
        "sigma_y_urad": sigma_y * 1e6,
        "sigma_highland_urad": sigma_highland * 1e6,
        "ratio_sigma_highland": sigma_x / sigma_highland if sigma_highland > 0 else 0,
        "mean_pla_mm": mean_pla,
        "xX0": xX0,
        "kappa_x": kappa_x,
        "kappa_y": kappa_y,
        "kappa_avg": kappa_avg,
        "kappa_err": kappa_err,
        "kappa_err_boot": kappa_err_boot,
        "kappa_ci_lo": kappa_ci_lo,
        "kappa_ci_hi": kappa_ci_hi,
        "significance": abs(kappa_avg) / kappa_err,
        "significance_boot": abs(kappa_avg) / kappa_err_boot if kappa_err_boot > 0 else 0,
        "tail_frac_pct": f_tail * 100,
        "tail_excess_pct": (f_tail - f_gauss) * 100,
        "sigma_core_urad": s1 * 1e6,
        "sigma_tail_urad": s2 * 1e6,
        "w_tail": w,
        "p_gev": p_gev,
        "f_hit_data": f_hit_data,
        "f_intermediate": f_intermediate,
        "theta_x": theta_x,
        "theta_y": theta_y,
    }

def load_all_results(data_dir, fiducial=False, energy_cut_fraction=ENERGY_CUT_FRACTION_DEFAULT,
                     n_cut_sigma=n_cut_sigma_DEFAULT,
                     fiducial_x=FIDUCIAL_X_MM_DEFAULT,
                     fiducial_y=FIDUCIAL_Y_MM_DEFAULT):
    results = {}
    tag = "fiducial" if fiducial else "full"
    for f in sorted(data_dir.glob("*.root")):
        label = f.stem
        if "2GeV" in label:
            energy = 2.0
        elif "6GeV" in label:
            energy = 6.0
        else:
            energy = 4.0

        r = analyze_file(f, p_gev=energy, label=label, fiducial=fiducial,
                         energy_cut_fraction=energy_cut_fraction,
                         n_cut_sigma=n_cut_sigma,
                         fiducial_x=fiducial_x, fiducial_y=fiducial_y)
        if r:
            results[label] = r
            print(f"  [{tag}] {label} (N={r['N']}, kappa={r['kappa_avg']:.2f})")
    return results

def get_solid_kappa(results):
    solid_kappa = {}
    solid_boot_se = {}
    for energy in ENERGY_SCAN:
        key = f"control_solid_{energy}GeV"
        r = results.get(key)
        if r:
            solid_kappa[energy] = r['kappa_avg']
            solid_boot_se[energy] = r['kappa_err_boot']
        else:
            r4 = results.get("control_solid_4GeV")
            if r4:
                solid_kappa[energy] = r4['kappa_avg']
                solid_boot_se[energy] = r4['kappa_err_boot']
            else:
                solid_kappa[energy] = 0
                solid_boot_se[energy] = 0
    return solid_kappa, solid_boot_se

def print_tables(results, tag="", n_cut_sigma=n_cut_sigma_DEFAULT):
    prefix_tag = f" [{tag}]" if tag else ""

    solid_kappa, solid_boot_se = get_solid_kappa(results)

    print(f"TABLE 1{prefix_tag}: Control Configurations")
    print(f"  (Angle cut: {n_cut_sigma}x sigma_Highland per config)")
    print(f"  (SE: formula=sqrt(24/N), boot=bootstrap 1000 resamples)")
    print(f"{'Config':<25} {'N':>7} {'sigma_x (urad)':>14} {'Highland':>10} "
          f"{'s/H':>6} {'kappa_avg':>9} {'+-form':>7} {'+-boot':>7}")

    for energy in ENERGY_SCAN:
        key = f"control_solid_{energy}GeV"
        r = results.get(key)
        if r:
            print(f"{f'Solid PLA ({energy} GeV)':<25} {r['N']:>7} "
                  f"{r['sigma_x_urad']:>14.1f} {r['sigma_highland_urad']:>10.1f} "
                  f"{r['ratio_sigma_highland']:>6.3f} {r['kappa_avg']:>9.2f} "
                  f"{r['kappa_err']:>7.3f} {r['kappa_err_boot']:>7.3f}")

    air = results.get("control_air_4GeV")
    if air:
        print(f"{'Air only (4 GeV)':<25} {air['N']:>7} "
              f"{air['sigma_x_urad']:>14.1f} {air['sigma_highland_urad']:>10.1f} "
              f"{air['ratio_sigma_highland']:>6.3f} {air['kappa_avg']:>9.2f} "
              f"{air['kappa_err']:>7.3f} {air['kappa_err_boot']:>7.3f}")
    else:
        print(f"\n  NOTE: Air control (control_air_4GeV) produced no data.")
        print(f"  With no target volume, SteppingAction never detects entry/exit,")
        print(f"  so EventAction writes 0 rows. Geant4 auto-deletes the empty ROOT file.")
        print(f"  This is expected -- the air control cannot be analyzed with the standard pipeline.")

    kappa_base_4 = solid_kappa.get(4, 0)
    base_se_4 = solid_boot_se.get(4, 0)

    print(f"TABLE 2{prefix_tag}: Infill Scan at 4 GeV -- Excess Kurtosis")
    print(f"  (Baseline solid PLA: kappa = {kappa_base_4:.2f})")
    print(f"  (Observable: Dk = kappa_lattice - kappa_solid)")
    print(f"{'Geometry':<14} {'Infill':>6} {'N':>7} {'sigma_x':>8} {'sigma_H':>8} "
          f"{'s/H':>6} {'kappa':>8} {'Dk':>8} {'Dk/SE_b':>8} {'Tail%':>7}")

    geometries = {
        "rect": ("Rectilinear", [20, 40, 60, 80]),
        "hc":   ("Honeycomb",   [20, 40, 60, 80]),
        "gyr":  ("Gyroid",      [20, 40, 60, 80]),
        "cub":  ("3D Grid",     [20, 40, 60, 80]),
        "vor":  ("Voronoi",     [20, 40, 60, 80]),
    }

    infill_data = {}
    for prefix, (name, infills) in geometries.items():
        geom_infills = []
        geom_kappas = []
        geom_deltas = []
        for infill in infills:
            key = f"{prefix}_{infill}pct_4GeV"
            r = results.get(key)
            if r:
                dk = r['kappa_avg'] - kappa_base_4
                dk_se = np.sqrt(r['kappa_err_boot']**2 + base_se_4**2)
                dk_sig = dk / dk_se if dk_se > 0 else 0
                print(f"{name:<14} {infill:>5}% {r['N']:>7} "
                      f"{r['sigma_x_urad']:>7.1f} {r['sigma_highland_urad']:>7.1f} "
                      f"{r['ratio_sigma_highland']:>6.3f} {r['kappa_avg']:>8.2f} "
                      f"{dk:>+8.2f} {dk_sig:>7.1f}s {r['tail_frac_pct']:>6.2f}%")
                geom_infills.append(infill)
                geom_kappas.append(r['kappa_avg'])
                geom_deltas.append(dk)
        infill_data[name] = (geom_infills, geom_kappas, geom_deltas)

    if kappa_base_4 > 0:
        for name in infill_data:
            infills_l, kappas_l, deltas_l = infill_data[name]
            infills_l.append(100)
            kappas_l.append(kappa_base_4)
            deltas_l.append(0)
            infill_data[name] = (infills_l, kappas_l, deltas_l)

    print(f"TABLE 3{prefix_tag}: Energy Dependence (40% infill)")
    print(f"  Baselines: " + ", ".join(
        f"solid_{e}GeV kappa={solid_kappa.get(e, 0):.2f}" for e in ENERGY_SCAN))
    print(f"{'Geometry':<14} {'E (GeV)':>8} {'N':>7} {'sigma_x':>8} {'sigma_H':>8} "
          f"{'s/H':>6} {'kappa':>8} {'k_solid':>8} {'Dk':>8} {'cut(mrad)':>10}")

    energy_configs = [
        ("Rectilinear", "rect_40pct_2GeV", 2),
        ("Rectilinear", "rect_40pct_4GeV", 4),
        ("Rectilinear", "rect_40pct_6GeV", 6),
        ("Gyroid",      "gyr_40pct_2GeV",  2),
        ("Gyroid",      "gyr_40pct_4GeV",  4),
        ("Gyroid",      "gyr_40pct_6GeV",  6),
    ]
    energy_dk = {}
    for name, key, energy in energy_configs:
        r = results.get(key)
        if r:
            k_solid = solid_kappa.get(energy, solid_kappa.get(4, 0))
            dk = r['kappa_avg'] - k_solid
            print(f"{name:<14} {energy:>7.0f} {r['N']:>7} "
                  f"{r['sigma_x_urad']:>7.1f} {r['sigma_highland_urad']:>7.1f} "
                  f"{r['ratio_sigma_highland']:>6.3f} {r['kappa_avg']:>8.2f} "
                  f"{k_solid:>8.2f} {dk:>+8.2f} {r['angle_cut_mrad']:>9.1f}")
            energy_dk.setdefault(name, []).append((energy, dk))

    print(f"TABLE 4{prefix_tag}: Two-Gaussian Mixture Fits")
    print(f"{'Config':<22} {'sigma_core':>10} {'sigma_tail':>10} {'w_tail':>8} {'ratio':>14}")
    fit_keys = ["control_solid_4GeV",
                "rect_20pct_4GeV", "rect_40pct_4GeV", "rect_60pct_4GeV", "rect_80pct_4GeV",
                "hc_40pct_4GeV", "gyr_40pct_4GeV", "cub_40pct_4GeV", "vor_40pct_4GeV"]
    for key in fit_keys:
        r = results.get(key)
        if r and r['w_tail'] > 0.001:
            ratio = r['sigma_tail_urad'] / r['sigma_core_urad']
            print(f"{key:<22} {r['sigma_core_urad']:>9.1f} {r['sigma_tail_urad']:>9.1f} "
                  f"{r['w_tail']:>8.4f} {ratio:>14.2f}")

    sweep0_file = Path("data/sweep0/sweep0_summary.json")
    if sweep0_file.exists():
        with open(sweep0_file) as sf:
            sweep0 = json.load(sf)

        BINARY_GEOMS = {"rectilinear", "honeycomb"}

        print(f"TABLE 5{prefix_tag}: Formula Verification -- Dk = k_geo * (1 + k_M/3)")
        print(f"  k_geo (ray): 3 Var(s)/<s>^2 from ray-trace path distribution")
        print(f"  k_M = kappa_solid = {kappa_base_4:.2f} (4 GeV, {n_cut_sigma}sigma cut)")
        print(f"  Moliere 4th moment diverges => k_M depends on cut => cross-term suppressed")
        print(f"  => Dk ≈ k_geo (not k_geo*(1+k_M/3)) for normalized-cut measurement")
        geom_display = {"rectilinear": "Rectilinear", "honeycomb": "Honeycomb",
                        "gyroid": "Gyroid", "cubic": "3D Grid", "voronoi": "Voronoi"}
        print(f"{'Geometry':<14} {'Infill':>6} {'f_ray':>6} {'f_data':>6} "
              f"{'k_geo':>7} {'Dk_G4':>7} {'Dk/kgeo':>7}")

        for item in sweep0:
            geom = item["geometry"]
            infill = item.get("infill_pct", item.get("infill_target_pct", 0))
            k_ray = item["predicted_kurtosis"]
            f_hit_ray = item.get("hit_fraction", None)

            prefix_map = {"rectilinear": "rect", "honeycomb": "hc", "gyroid": "gyr",
                          "cubic": "cub", "voronoi": "vor"}
            prefix = prefix_map.get(geom, geom)
            key = f"{prefix}_{infill}pct_4GeV"
            r = results.get(key)
            if r and infill < 100:
                dk_g4 = r['kappa_avg'] - kappa_base_4
                f_data = r.get('f_hit_data', 0)
                dname = geom_display.get(geom, geom)

                ratio_simple = dk_g4 / k_ray if k_ray > 0.01 else float('nan')
                f_ray_str = f"{f_hit_ray:.3f}" if f_hit_ray else "--"
                f_data_str = f"{f_data:.3f}" if f_data > 0 else "--"
                print(f"{dname:<14} {infill:>5}% {f_ray_str:>6} {f_data_str:>6} "
                      f"{k_ray:>7.3f} {dk_g4:>+7.2f} {ratio_simple:>7.2f}")

    print(f"TABLE 5b{prefix_tag}: Binary Path Verification (Geant4 pla_path distribution)")
    print(f"  For binary geometries, ALL PLA paths should be ~{SAMPLE_T_MM:.0f} mm or 0 mm")
    print(f"  f_data = fraction with pla_path > 0.1 mm (from Geant4)")
    print(f"  'intermediate' = fraction with 0.1 < pla_path < {SAMPLE_T_MM - 0.5:.1f} mm")
    print(f"{'Config':<25} {'f_data':>7} {'f_ray':>7} {'f_ratio':>7} {'intermediate%':>13}")

    for item in sweep0:
        geom = item["geometry"]
        infill = item.get("infill_pct", item.get("infill_target_pct", 0))
        f_hit_ray = item.get("hit_fraction", None)
        prefix = {"rectilinear": "rect", "honeycomb": "hc", "gyroid": "gyr",
                  "cubic": "cub", "voronoi": "vor"}.get(geom, geom)
        key = f"{prefix}_{infill}pct_4GeV"
        r = results.get(key)
        if r and infill < 100:
            f_data = r.get('f_hit_data', 0)
            f_inter = r.get('f_intermediate', 0)
            f_ratio = f_data / f_hit_ray if f_hit_ray and f_hit_ray > 0 else 0
            f_ray_str = f"{f_hit_ray:.3f}" if f_hit_ray else "--"
            print(f"{key:<25} {f_data:>7.3f} {f_ray_str:>7} {f_ratio:>7.3f} {f_inter*100:>12.2f}%")

    print(f"TABLE 6{prefix_tag}: Angular Resolution Ratio (sigma_H / sigma_telescope)")
    print(f"  sigma_telescope = {SIGMA_TELESCOPE_MRAD:.3f} mrad "
          f"(MIMOSA26, {SIGMA_POS_UM:.0f}um res, {LEVER_ARM_MM:.0f}mm spacing, "
          f"{N_PLANES_UP} planes)")
    print(f"  Configs with R < 5 are FLAGGED as marginal")
    print(f"{'Config':<25} {'E':>4} {'sigma_H(mrad)':>13} {'sigma_T(mrad)':>13} "
          f"{'R=sH/sT':>8} {'Status':>10}")

    all_configs = []
    for key, r in sorted(results.items()):
        sigma_h_mrad = r['sigma_highland_urad'] / 1e3
        R = sigma_h_mrad / SIGMA_TELESCOPE_MRAD
        status = "OK" if R >= 5 else "MARGINAL"
        energy = r['p_gev']
        print(f"{key:<25} {energy:>3.0f} {sigma_h_mrad:>13.3f} "
              f"{SIGMA_TELESCOPE_MRAD:>13.3f} {R:>8.1f} {status:>10}")
        all_configs.append((key, energy, sigma_h_mrad, R, status))

    marginal = [c for c in all_configs if c[4] == "MARGINAL"]
    if marginal:
        print(f"\n  WARNING: {len(marginal)} config(s) have R < 5 "
              f"(telescope resolution contaminates kurtosis):")
        for key, energy, sh, R, _ in marginal:
            print(f"    {key}: R={R:.1f} -- measurement would require "
                  f"lower beam energy or thicker samples")
    else:
        print(f"\n  All configs have R >= 5. Telescope resolution is subdominant.")

    print(f"TABLE 7{prefix_tag}: Recommended Energy Range per Geometry x Infill")
    print(f"  Criterion: R = sigma_Highland / sigma_telescope >= 5")
    print(f"  sigma_telescope = {SIGMA_TELESCOPE_MRAD:.3f} mrad")
    print(f"  Highland sigma scales as 1/p, so lower energy -> larger sigma -> easier measurement")
    print(f"  DESY: 1-6 GeV, CERN PS: 0.5-4 GeV")

    test_energies = np.arange(0.5, 7.0, 0.5)
    threshold_R = 5.0

    header = f"{'Geometry':<14}"
    for infill in [20, 40, 60, 80]:
        header += f" {infill:>5}%"
    print(header)

    for prefix, name in DISPLAY_NAMES.items():
        row = f"{name:<14}"
        for infill in [20, 40, 60, 80]:
            key = f"{prefix}_{infill}pct_4GeV"
            r = results.get(key)
            if r:
                xX0 = r['xX0']
                max_e = None
                for e in test_energies:
                    sh = highland_sigma(xX0, e) * 1e3
                    R = sh / SIGMA_TELESCOPE_MRAD
                    if R >= threshold_R:
                        max_e = e
                if max_e is None:
                    row += f" {'<0.5':>6}"
                elif max_e >= 6.5:
                    row += f" {'any':>6}"
                else:
                    row += f" {f'<={max_e:.0f}':>6}"
            else:
                row += f" {'--':>6}"
        print(row)

    print(f"\n  Reading: '<= N' means R>=5 only at beam energies up to N GeV.")
    print(f"  'any' means R>=5 even at 6 GeV. '<0.5' means even 0.5 GeV is insufficient.")

    print(f"PROPOSAL SUMMARY{prefix_tag}")

    solid = results.get("control_solid_4GeV")
    if solid:
        print(f"\n1. Highland Formula Validation (Solid PLA, 4 GeV):")
        print(f"   Measured sigma_x = {solid['sigma_x_urad']:.1f} urad")
        print(f"   Highland prediction = {solid['sigma_highland_urad']:.1f} urad")
        print(f"   Ratio = {solid['ratio_sigma_highland']:.3f}")
        print(f"   Status: {'PASS' if 0.85 < solid['ratio_sigma_highland'] < 1.15 else 'CHECK'} "
              f"(within {abs(solid['ratio_sigma_highland']-1)*100:.1f}%)")

    print(f"\n2. Moliere Baseline (solid PLA kurtosis):")
    for energy in ENERGY_SCAN:
        r = results.get(f"control_solid_{energy}GeV")
        if r:
            print(f"   {energy} GeV: kappa_solid = {solid_kappa[energy]:.2f} "
                  f"+- {solid_boot_se[energy]:.3f} (boot)")
    print(f"   NOTE: Non-zero due to Moliere single-scattering tails.")
    print(f"   The geometric observable is Dk = kappa_lattice - kappa_solid(E).")

    print(f"\n3. Kurtosis Detection (Dk significance, bootstrap SE):")
    min_sig = float('inf')
    max_sig = 0
    min_key = max_key = ""
    for key, r in results.items():
        if "control" in key:
            continue
        energy = int(r['p_gev'])
        dk = r['kappa_avg'] - solid_kappa.get(energy, solid_kappa.get(4, 0))
        dk_se = np.sqrt(r['kappa_err_boot']**2 + solid_boot_se.get(energy, 0)**2)
        sig = abs(dk) / dk_se if dk_se > 0 else 0
        if sig < min_sig:
            min_sig = sig
            min_key = key
        if sig > max_sig:
            max_sig = sig
            max_key = key
    print(f"   Minimum |Dk|: {min_key} at {min_sig:.1f}s")
    print(f"   Maximum |Dk|: {max_key} at {max_sig:.1f}s")
    print(f"   All lattice configs detected at >5s: "
          f"{'YES' if min_sig > 5 else 'NO (some below 5s)'}")

    print(f"\n4. Geometry Ranking (Dk at 40% infill, 4 GeV):")
    ranking = []
    for prefix, name in DISPLAY_NAMES.items():
        r = results.get(f"{prefix}_40pct_4GeV")
        if r:
            dk = r['kappa_avg'] - kappa_base_4
            ranking.append((name, dk, r['kappa_avg']))
    ranking.sort(key=lambda x: -x[1])
    for name, dk, k in ranking:
        print(f"   {name:<14}: kappa = {k:.2f}, Dk = {dk:+.2f}")

    print(f"\n5. Energy Scaling (energy-matched baselines, {n_cut_sigma}sigma cut):")
    for prefix, name in [("rect", "Rectilinear"), ("gyr", "Gyroid")]:
        deltas_e = []
        for energy in ENERGY_SCAN:
            r = results.get(f"{prefix}_40pct_{energy}GeV")
            if r:
                dk = r['kappa_avg'] - solid_kappa.get(energy, solid_kappa.get(4, 0))
                deltas_e.append((energy, r['kappa_avg'], dk))
        if deltas_e:
            dk_vals = [dk for _, _, dk in deltas_e]
            if np.mean(np.abs(dk_vals)) > 0:
                spread = (max(dk_vals) - min(dk_vals)) / np.mean(np.abs(dk_vals)) * 100
            else:
                spread = 0
            print(f"   {name} 40%: Dk = {', '.join(f'{dk:+.2f} ({e}GeV)' for e, _, dk in deltas_e)}")
            print(f"   Spread: {spread:.1f}%")

    print(f"\n6. Monotonicity Check (Dk should decrease with infill):")
    for prefix, name in DISPLAY_NAMES.items():
        dks = []
        for infill in [20, 40, 60, 80]:
            r = results.get(f"{prefix}_{infill}pct_4GeV")
            if r:
                dks.append(r['kappa_avg'] - kappa_base_4)
        if len(dks) >= 2:
            monotonic = all(dks[i] >= dks[i+1] for i in range(len(dks)-1))
            trend = " > ".join(f"{dk:+.2f}" for dk in dks)
            status = "MONOTONIC" if monotonic else "NON-MONOTONIC"
            print(f"   {name:<14}: Dk = {trend}  [{status}]")

    print("TABLE 8: Slicer Validation -- Simulation vs Real 3D Printing")

    print("""
RECTILINEAR:
  This geometry CAN be produced by PrusaSlicer ("Rectilinear" infill).
  The simulation DOES NOT exactly match the slicer's implementation because:
  - Real FDM prints alternate layer direction: odd layers print walls in X,
    even layers print walls in Y. Each wall is one layer height (~0.2 mm).
  - The simulation places continuous X and Y wall slabs spanning the full
    10 mm Z height, so every particle sees both wall families simultaneously.
  - In reality, a particle at (x,y) traversing 50 layers sees ~25 X-wall
    layers and ~25 Y-wall layers. If it hits only one wall family, it gets
    5 mm of PLA, not 10 mm. This creates a ternary distribution {0, 5, 10 mm}
    instead of the simulation's binary {0, 10 mm}.
  - The simulation's binary distribution gives MAXIMUM kurtosis for a given
    infill -- real rectilinear would have lower Dk. The infill fraction is
    unchanged (same total volume of PLA).
  IMPACT: Simulation provides an upper bound on Dk. The qualitative physics
  (binary path -> high kurtosis, monotonic decrease with infill) is correct.

HONEYCOMB:
  This geometry CAN be produced by PrusaSlicer ("Honeycomb" infill).
  The simulation APPROXIMATELY matches the slicer's implementation because:
  - PrusaSlicer's honeycomb uses three families of zigzag lines at 0/60/120 deg,
    rotating through the three orientations across consecutive layers.
  - The simulation uses voxelized finite-length hex edge segments creating
    proper hexagonal cells with 3-way Y-junctions at vertices.
  - Same layer-alternation caveat as rectilinear: real path distribution
    may differ from continuous-wall simulation.
  IMPACT: Moderate. The hex topology matches, but continuous vs layer-by-layer
  wall structure could affect the path distribution and hence Dk magnitude.

GYROID:
  This geometry CAN be produced by PrusaSlicer ("Gyroid" infill).
  The simulation DOES match the slicer's implementation because:
  - PrusaSlicer uses the same trigonometric TPMS level-set equation:
    sin(kx)cos(ky) + sin(ky)cos(kz) + sin(kz)cos(kx) = threshold
  - The slicer analytically solves for 2D cross-sections at each z-level.
  - The simulation voxelizes the same equation and integrates through z.
  IMPACT: Best match of all geometries.

3D GRID (labeled "cubic" in macro commands):
  This geometry DOES NOT match PrusaSlicer's "Cubic" infill pattern.
  - PrusaSlicer's "Cubic" orients cubes standing on a vertex (~45 deg rotation),
    creating cross-sections that vary layer-by-layer.
  - The simulation uses orthogonal walls in X, Y, Z planes (a simple 3D grid).
  - Results are valid as "orthogonal 3D grid" and labeled "3D Grid" throughout
    to avoid confusion with the slicer's tilted-cube pattern.
  IMPACT: Significant mismatch with slicer. Labeled "3D Grid" in all output.

VORONOI:
  This geometry CANNOT be produced by standard slicer software.
  - Voronoi/CVT tessellation is not a built-in infill pattern in any slicer.
  - Physical samples would require custom STL generation.
  IMPACT: Valid as simulation predictions; requires custom fabrication.
""")

    return infill_data, solid_kappa, energy_dk

def main():
    parser = argparse.ArgumentParser(description="Proposal Analysis Pipeline")
    parser.add_argument("--data-dir", type=str, default="data/proposal",
                        help="Directory with ROOT files (default: data/proposal)")
    parser.add_argument("--output-dir", type=str, default="results/proposal",
                        help="Output directory (default: results/proposal)")
    parser.add_argument("--energy-cut-fraction", type=float, default=ENERGY_CUT_FRACTION_DEFAULT,
                        help=f"Keep events with energy_out > fraction * beam_energy (default {ENERGY_CUT_FRACTION_DEFAULT})")
    parser.add_argument("--angle-cut-sigma", type=float, default=n_cut_sigma_DEFAULT,
                        help=f"Angle cut in multiples of sigma_Highland (default {n_cut_sigma_DEFAULT})")
    parser.add_argument("--fiducial-x", type=float, default=FIDUCIAL_X_MM_DEFAULT,
                        help=f"Fiducial cut on |entry_x| in mm (default {FIDUCIAL_X_MM_DEFAULT})")
    parser.add_argument("--fiducial-y", type=float, default=FIDUCIAL_Y_MM_DEFAULT,
                        help=f"Fiducial cut on |entry_y| in mm (default {FIDUCIAL_Y_MM_DEFAULT})")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cut_kwargs = dict(
        energy_cut_fraction=args.energy_cut_fraction,
        n_cut_sigma=args.angle_cut_sigma,
        fiducial_x=args.fiducial_x,
        fiducial_y=args.fiducial_y,
    )

    print("Cut configuration:")
    print(f"  Energy cut: energy_out > {args.energy_cut_fraction} * beam_energy")
    print(f"  Angle cut: |theta| < {args.angle_cut_sigma} * sigma_Highland_solid")
    print(f"  Fiducial: |entry_x| < {args.fiducial_x} mm, |entry_y| < {args.fiducial_y} mm")

    if not HAS_UPROOT:
        print("ERROR: uproot not installed. pip install uproot")
        sys.exit(1)

    print(f"\nANALYSIS PASS 1 (PRIMARY): Telescope fiducial cut "
          f"(|x|<{args.fiducial_x}mm, |y|<{args.fiducial_y}mm)")
    results_fid = load_all_results(data_dir, fiducial=True, **cut_kwargs)

    if not results_fid:
        print("No results found!")
        sys.exit(1)

    infill_data, solid_kappa_fid, energy_dk_fid = print_tables(
        results_fid, tag="Fiducial (primary)", n_cut_sigma=args.angle_cut_sigma)

    print("\nANALYSIS PASS 2 (SECONDARY): Full acceptance (no fiducial cut)")
    results_full = load_all_results(data_dir, fiducial=False, **cut_kwargs)

    infill_data_full = None
    solid_kappa_full = {}
    if results_full:
        infill_data_full, solid_kappa_full, _ = print_tables(
            results_full, tag="Full acceptance", n_cut_sigma=args.angle_cut_sigma)

    if results_full and results_fid:
        solid_fid = results_fid.get("control_solid_4GeV")
        solid_full = results_full.get("control_solid_4GeV")
        kappa_base_fid = solid_fid['kappa_avg'] if solid_fid else 0
        kappa_base_full = solid_full['kappa_avg'] if solid_full else 0
        baseline_boot_fid = solid_fid['kappa_err_boot'] if solid_fid else 0

        print("TABLE 9: Fiducial vs Full Acceptance Comparison")
        print(f"  Fiducial: |x|<{args.fiducial_x}mm, |y|<{args.fiducial_y}mm "
              f"(MIMOSA26 active area)")
        print(f"  Threshold: |Dk_diff| > 1 bootstrap SE ({baseline_boot_fid:.2f})")
        print(f"{'Config':<25} {'N_fid':>7} {'N_full':>7} "
              f"{'k_fid':>8} {'k_full':>8} {'Dk_fid':>8} {'Dk_full':>8} "
              f"{'diff':>7} {'Flag':>8}")

        any_flagged = False
        for key in sorted(set(list(results_fid.keys()) + list(results_full.keys()))):
            rfid = results_fid.get(key)
            rf = results_full.get(key)
            if rfid and rf:
                energy = int(rfid['p_gev'])
                if "control" in key:
                    dk_fid = 0
                    dk_full = 0
                else:
                    dk_fid = rfid['kappa_avg'] - solid_kappa_fid.get(energy, kappa_base_fid)
                    dk_full = rf['kappa_avg'] - solid_kappa_full.get(energy, kappa_base_full)
                diff = abs(rfid['kappa_avg'] - rf['kappa_avg'])
                flag = "DIFFERS" if diff > baseline_boot_fid else ""
                if flag:
                    any_flagged = True
                print(f"{key:<25} {rfid['N']:>7} {rf['N']:>7} "
                      f"{rfid['kappa_avg']:>8.2f} {rf['kappa_avg']:>8.2f} "
                      f"{dk_fid:>+8.2f} {dk_full:>+8.2f} "
                      f"{diff:>7.2f} {flag:>8}")

        if any_flagged:
            print("\n  WARNING: Some configs differ by > 1 bootstrap SE between "
                  "fiducial and full acceptance.")
            print("  Fiducial cut is the primary result (what the telescope measures).")
        else:
            print("\n  All configs agree within 1 bootstrap SE.")

    if HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        ax = axes[0, 0]
        markers = {'Rectilinear': 'o', 'Honeycomb': 's', 'Gyroid': '^',
                   '3D Grid': 'D', 'Voronoi': 'v'}
        colors = {'Rectilinear': '#2196F3', 'Honeycomb': '#FF9800', 'Gyroid': '#4CAF50',
                  '3D Grid': '#9C27B0', 'Voronoi': '#F44336'}
        for name, (infills_p, kappas_p, deltas_p) in infill_data.items():
            if not infills_p:
                continue
            idx = np.argsort(infills_p)
            inf_sorted = [infills_p[i] for i in idx]
            dk_sorted = [deltas_p[i] for i in idx]
            ax.plot(inf_sorted, dk_sorted, f'-{markers.get(name, "o")}',
                    color=colors.get(name, 'gray'), label=name, markersize=8, linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Infill (%)', fontsize=12)
        ax.set_ylabel('Dk = kappa_lattice - kappa_solid', fontsize=12)
        ax.set_title('Kurtosis excess (fiducial cut)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        for infill, color in [(20, '#D32F2F'), (40, '#FF9800'),
                              (60, '#4CAF50'), (80, '#2196F3')]:
            key = f"rect_{infill}pct_4GeV"
            r = results_fid.get(key)
            if r:
                theta = r['theta_x']
                sigma = np.std(theta)
                ax.hist(theta / sigma, bins=200, density=True, alpha=0.4,
                        color=color, label=f'{infill}%')
        x_g = np.linspace(-5, 5, 500)
        ax.semilogy(x_g, stats.norm.pdf(x_g), 'k--', linewidth=2, label='Gaussian')
        ax.set_xlabel('theta_x / sigma (normalized)', fontsize=12)
        ax.set_ylabel('Probability density (log)', fontsize=12)
        ax.set_title('Tail structure: Rectilinear (fiducial)', fontsize=13, fontweight='bold')
        ax.set_ylim(1e-5, 1)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        for name in ['Rectilinear', 'Honeycomb', 'Gyroid', '3D Grid', 'Voronoi']:
            prefix_map = {'Rectilinear': 'rect', 'Honeycomb': 'hc', 'Gyroid': 'gyr',
                         '3D Grid': 'cub', 'Voronoi': 'vor'}
            prefix = prefix_map[name]
            infills_r = []
            ratios_r = []
            for infill in [20, 40, 60, 80]:
                r = results_fid.get(f"{prefix}_{infill}pct_4GeV")
                if r:
                    infills_r.append(infill)
                    ratios_r.append(r['ratio_sigma_highland'])
            if infills_r:
                ax.plot(infills_r, ratios_r, f'-{markers.get(name, "o")}',
                        color=colors.get(name, 'gray'), label=name, markersize=8, linewidth=2)
        solid_fid = results_fid.get("control_solid_4GeV")
        if solid_fid:
            ax.axhline(y=solid_fid['ratio_sigma_highland'], color='gray',
                       linestyle='--', linewidth=1,
                       label=f'Solid ({solid_fid["ratio_sigma_highland"]:.3f})')
        ax.axhline(y=1.0, color='red', linestyle=':', linewidth=1, label='Highland = 1.0')
        ax.set_xlabel('Infill (%)', fontsize=12)
        ax.set_ylabel('sigma_measured / sigma_Highland', fontsize=12)
        ax.set_title('Highland formula accuracy (fiducial)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        for prefix, name, color in [("rect", "Rectilinear 40%", '#2196F3'),
                                     ("gyr", "Gyroid 40%", '#4CAF50')]:
            energies = []
            deltas_e = []
            for energy in ENERGY_SCAN:
                r = results_fid.get(f"{prefix}_40pct_{energy}GeV")
                if r:
                    dk = r['kappa_avg'] - solid_kappa_fid.get(energy, solid_kappa_fid.get(4, 0))
                    energies.append(energy)
                    deltas_e.append(dk)

            if energies:
                ax.plot(energies, deltas_e, '-o', color=color, label=f'Dk {name}',
                        markersize=8, linewidth=2)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Beam energy (GeV)', fontsize=12)
        ax.set_ylabel('Dk = kappa_lattice - kappa_solid(E)', fontsize=12)
        ax.set_title('Energy dependence (energy-matched baselines)', fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.suptitle(f'MCS Highland Validation -- Proposal Results (fiducial, {args.angle_cut_sigma}sigma_H cut)',
                     fontsize=15, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = results_dir / "proposal_results.png"
        fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\nPlot saved: {plot_path}")

    summary = {"fiducial_cut": {}, "full_acceptance": {}}
    for key, r in results_fid.items():
        s = {k: v for k, v in r.items() if k not in ('theta_x', 'theta_y')}
        summary["fiducial_cut"][key] = s
    for key, r in results_full.items():
        s = {k: v for k, v in r.items() if k not in ('theta_x', 'theta_y')}
        summary["full_acceptance"][key] = s

    summary["solid_baselines"] = {}
    for energy in ENERGY_SCAN:
        summary["solid_baselines"][f"{energy}GeV"] = {
            "kappa_fiducial": solid_kappa_fid.get(energy),
            "kappa_full": solid_kappa_full.get(energy),
        }

    summary["cut_parameters"] = {
        "energy_cut_fraction": args.energy_cut_fraction,
        "angle_cut_sigma": args.angle_cut_sigma,
        "fiducial_x_mm": args.fiducial_x,
        "fiducial_y_mm": args.fiducial_y,
    }
    summary["telescope_info"] = {
        "sigma_telescope_mrad": SIGMA_TELESCOPE_MRAD,
        "sigma_pos_um": SIGMA_POS_UM,
        "lever_arm_mm": LEVER_ARM_MM,
        "n_planes_upstream": N_PLANES_UP,
        "fiducial_x_mm": args.fiducial_x,
        "fiducial_y_mm": args.fiducial_y,
    }

    json_path = results_dir / "proposal_summary.json"
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2, default=str)
    print(f"JSON saved: {json_path}")

if __name__ == "__main__":
    main()

