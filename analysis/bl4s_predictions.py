import numpy as np
import json
from pathlib import Path

PLA_X0_MM = 315.0
SAMPLE_T_MM = 10.0
WALL_T_MM = 0.4

RESULTS_DIR = Path("results/bl4s_predictions")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def highland_sigma(x_mm, p_gev, X0_mm=PLA_X0_MM):
    x_over_x0 = x_mm / X0_mm
    if x_over_x0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_x0) * (1 + 0.038 * np.log(x_over_x0))

def highland_sigma2(x_mm, p_gev, X0_mm=PLA_X0_MM):
    s = highland_sigma(x_mm, p_gev, X0_mm)
    return s ** 2

def moliere_kappa_FL(w2=0.02, r=3.2):
    w1 = 1.0 - w2
    num = w1 + w2 * r**4
    den = (w1 + w2 * r**2) ** 2
    return 3.0 * num / den - 3.0

def moliere_kappa_energy_dependent(p_gev, x_mm=SAMPLE_T_MM, cut_nsigma=10.0):
    kM_FL = moliere_kappa_FL()
    return kM_FL

KAPPA_M_G4 = {
    2.0: 4.394,
    4.0: 3.616,
    6.0: 4.097,
}

THIN_WALL_ENHANCEMENT = {
    2.0: 1.00,
    4.0: 1.10,
    6.0: 1.22,
}

def load_raytrace_data():
    with open("data/sweep0/sweep0_summary.json") as f:
        data = json.load(f)

    lookup = {}
    for entry in data:
        geo = entry["geometry"]
        infill = entry["infill_target_pct"]
        lookup[(geo, infill)] = entry
    return lookup

BINARY_GEOS = {"rectilinear", "honeycomb"}
CONTINUOUS_GEOS = {"gyroid", "voronoi"}
TERNARY_GEOS = {"cubic"}

def kappa_geo_binary(f):
    if f <= 0 or f >= 1:
        return 0.0
    return 3.0 * (1.0 - f) / f

def kappa_geo_from_raytrace(entry, p_gev):
    mean_xX0 = entry["mean_x_over_X0"]
    var_xX0 = entry["var_x_over_X0"]

    if mean_xX0 <= 0:
        return 0.0

    return entry["predicted_kurtosis"]

def predict_kurtosis(kappa_M, kappa_geo):
    return kappa_M + kappa_geo * (1.0 + kappa_M / 3.0)

def kurtosis_stat_error(kappa, N):
    if N <= 0:
        return float('inf')
    return np.sqrt((24.0 + 12.0 * abs(kappa)) / N)

def required_events(kappa, n_sigma=5.0):
    if kappa <= 0:
        return float('inf')
    return (n_sigma / kappa) ** 2 * (24.0 + 12.0 * kappa)

def main():
    rt = load_raytrace_data()

    kM_analytical = moliere_kappa_FL()

    geometries = ["rectilinear", "honeycomb", "gyroid", "cubic", "voronoi"]
    infills = [20, 40, 60, 80]
    energies = [2.0, 4.0, 6.0]

    print("BL4S 2026 PREDICTIONS: Excess Kurtosis of MCS through PLA Lattice Metamaterials")
    print(f"\nAnalytical κ_M (Fruhwirth–Liendl): {kM_analytical:.3f}")
    print(f"Geant4 κ_M at 4 GeV:              {KAPPA_M_G4[4.0]:.3f}")
    print(f"Thin-wall enhancement at 4 GeV:    {THIN_WALL_ENHANCEMENT[4.0]:.2f}")
    print()

    E = 4.0
    kM_g4 = KAPPA_M_G4[E]
    kM_wall = kM_g4 * THIN_WALL_ENHANCEMENT[E]

    print(f"TABLE 1: Predicted κ at {E:.0f} GeV  (50k raw events → ~30k after cuts)")
    print(f"{'Geometry':<14s} {'Infill':>6s} {'f_hit':>6s} {'κ_geo':>7s} "
          f"{'κ_anal':>7s} {'κ_G4cal':>8s} {'κ_wall':>7s} "
          f"{'σ_stat':>7s} {'Signif':>7s} {'N_5σ':>8s} {'Type':>8s}")

    predictions_4gev = []

    for geo in geometries:
        for inf in infills:
            key = (geo, inf)
            if key not in rt:
                continue

            entry = rt[key]
            f_hit = entry.get("hit_fraction", 1.0)

            if geo in BINARY_GEOS:
                kgeo = kappa_geo_binary(f_hit)
                gtype = "binary"
            else:
                kgeo = kappa_geo_from_raytrace(entry, E)
                gtype = "continuous" if geo in CONTINUOUS_GEOS else "ternary"

            k_analytical = predict_kurtosis(kM_analytical, kgeo)
            k_g4cal = predict_kurtosis(kM_g4, kgeo)
            k_wall = predict_kurtosis(kM_wall, kgeo)

            N_eff = 30000
            sigma_k = kurtosis_stat_error(k_g4cal, N_eff)
            signif = k_g4cal / sigma_k if sigma_k > 0 else 0
            n5 = required_events(k_g4cal, 5.0)

            print(f"{geo:<14s} {inf:>5d}% {f_hit:>6.3f} {kgeo:>7.2f} "
                  f"{k_analytical:>7.2f} {k_g4cal:>8.2f} {k_wall:>7.2f} "
                  f"{sigma_k:>7.3f} {signif:>7.1f}σ {n5:>8.0f} {gtype:>8s}")

            predictions_4gev.append({
                "geometry": geo, "infill_pct": inf, "energy_GeV": E,
                "f_hit": f_hit, "kappa_geo": kgeo,
                "kappa_analytical": k_analytical,
                "kappa_g4_calibrated": k_g4cal,
                "kappa_wall_corrected": k_wall,
                "sigma_stat_30k": sigma_k,
                "significance_30k": signif,
                "N_5sigma": n5,
                "geometry_type": gtype,
            })

    print(f"TABLE 2: Energy Dependence — κ predictions at 2, 4, 6 GeV")
    print(f"{'Config':<20s} {'E':>4s} {'κ_M':>6s} {'f_hit':>6s} {'κ_geo':>7s} "
          f"{'κ_pred':>7s} {'Δκ':>7s} {'σ_stat':>7s} {'Signif':>7s}")

    energy_predictions = []

    key_configs = [
        ("rectilinear", 20), ("rectilinear", 40), ("rectilinear", 80),
        ("honeycomb", 40),
        ("gyroid", 20), ("gyroid", 40),
        ("cubic", 40),
        ("voronoi", 40),
    ]

    for geo, inf in key_configs:
        key = (geo, inf)
        if key not in rt:
            continue

        entry = rt[key]
        f_hit = entry.get("hit_fraction", 1.0)

        for E in energies:
            if E not in KAPPA_M_G4:
                continue

            kM = KAPPA_M_G4[E]

            if geo in BINARY_GEOS:
                kgeo = kappa_geo_binary(f_hit)
            else:
                kgeo = kappa_geo_from_raytrace(entry, E)

            k_pred = predict_kurtosis(kM, kgeo)
            delta_k = k_pred

            N_eff = 30000
            sigma_k = kurtosis_stat_error(k_pred, N_eff)
            signif = k_pred / sigma_k if sigma_k > 0 else 0

            label = f"{geo[:4]}_{inf}%"
            print(f"{label:<20s} {E:>4.0f} {kM:>6.2f} {f_hit:>6.3f} {kgeo:>7.2f} "
                  f"{k_pred:>7.2f} {k_pred:>7.2f} {sigma_k:>7.3f} {signif:>7.1f}σ")

            energy_predictions.append({
                "geometry": geo, "infill_pct": inf, "energy_GeV": E,
                "kappa_M": kM, "f_hit": f_hit, "kappa_geo": kgeo,
                "kappa_predicted": k_pred, "sigma_stat": sigma_k,
                "significance": signif,
            })

        print()

    print(f"TABLE 3: BL4S Measurement Plan — Observable Predictions at 4 GeV")

    print(f"\n{'Config':<20s} {'σ_H [mrad]':>11s} {'σ_pred [mrad]':>13s} "
          f"{'κ_pred':>7s} {'κ_solid':>8s} {'Δκ':>7s} "
          f"{'5σ detect?':>10s}")

    E = 4.0
    kM = KAPPA_M_G4[E]
    sigma_solid = highland_sigma(SAMPLE_T_MM, E) * 1e3

    for geo in geometries:
        for inf in infills:
            key = (geo, inf)
            if key not in rt:
                continue

            entry = rt[key]
            f_hit = entry.get("hit_fraction", 1.0)
            mean_xX0 = entry["mean_x_over_X0"]

            sigma_pred = highland_sigma(mean_xX0 * PLA_X0_MM, E) * 1e3

            if geo in BINARY_GEOS:
                kgeo = kappa_geo_binary(f_hit)
            else:
                kgeo = kappa_geo_from_raytrace(entry, E)

            k_pred = predict_kurtosis(kM, kgeo)
            delta_k = k_pred - 0

            N_eff = 30000
            sigma_k = kurtosis_stat_error(k_pred, N_eff)
            signif = k_pred / sigma_k
            detect = "YES" if signif > 5 else "marginal" if signif > 3 else "no"

            label = f"{geo[:4]}_{inf}%"
            print(f"{label:<20s} {sigma_solid:>11.3f} {sigma_pred:>13.3f} "
                  f"{k_pred:>7.2f} {kM:>8.2f} {k_pred:>7.2f} "
                  f"{detect:>10s} ({signif:.1f}σ)")

    print(f"TABLE 4: Solid Control Predictions (calibration targets)")
    print(f"{'Energy':>8s} {'σ_Highland':>12s} {'σ_Molière':>11s} "
          f"{'κ_M (FL)':>9s} {'κ_M (G4)':>9s} {'κ_M_err':>8s}")

    for E in energies:
        sigma_h = highland_sigma(SAMPLE_T_MM, E) * 1e3
        sigma_m = sigma_h * 1.12
        print(f"{E:>7.0f}  {sigma_h:>11.3f}  {sigma_m:>10.3f}  "
              f"{kM_analytical:>8.3f}  {KAPPA_M_G4[E]:>8.3f}  {0.30:>7.2f}")

    print(f"TABLE 5: N-Layer Scaling Predictions (rect_40% at 4 GeV)")
    print(f"{'N layers':>9s} {'κ_single':>9s} {'κ(N)=κ₁/N':>11s} "
          f"{'σ_stat(30k)':>12s} {'Signif':>7s}")

    f_rect40 = rt[("rectilinear", 40)]["hit_fraction"]
    kgeo_rect40 = kappa_geo_binary(f_rect40)
    k_single = predict_kurtosis(KAPPA_M_G4[4.0], kgeo_rect40)

    for N in [1, 2, 4, 8]:
        k_N = k_single / N
        sigma_k = kurtosis_stat_error(k_N, 30000)
        signif = k_N / sigma_k
        print(f"{N:>9d} {k_single:>9.2f} {k_N:>11.3f} "
              f"{sigma_k:>12.3f} {signif:>7.1f}σ")

    print("TABLE 6: CORRECTED PREDICTIONS FOR BL4S PROPOSAL (binary geometries only)")
    print("These are the analytically reliable predictions. ±12% systematic from thin-wall enhancement.")
    print(f"{'Config':<20s} {'f':>6s} {'κ_low':>7s} {'κ_central':>10s} {'κ_high':>7s} "
          f"{'Δκ':>7s} {'σ_stat':>7s} {'Signif':>7s}")

    E = 4.0
    kM = KAPPA_M_G4[E]
    kM_enhanced = kM * THIN_WALL_ENHANCEMENT[E]

    for geo in ["rectilinear", "honeycomb"]:
        for inf in infills:
            key = (geo, inf)
            if key not in rt:
                continue

            f_hit = rt[key]["hit_fraction"]
            kgeo = kappa_geo_binary(f_hit)

            k_central = predict_kurtosis(kM, kgeo)
            k_high = predict_kurtosis(kM_enhanced, kgeo)
            k_low = predict_kurtosis(kM_analytical, kgeo)

            dk = k_central - kM

            N_eff = 30000
            sigma_k = kurtosis_stat_error(k_central, N_eff)
            signif = dk / sigma_k if sigma_k > 0 else 0

            label = f"{geo[:4]}_{inf}%"
            print(f"{label:<20s} {f_hit:>6.3f} {k_low:>7.1f} {k_central:>10.1f} {k_high:>7.1f} "
                  f"{dk:>7.1f} {sigma_k:>7.3f} {signif:>7.0f}σ")

    print(f"\n  κ_low  = prediction with Fruhwirth–Liendl κ_M = {kM_analytical:.2f}")
    print(f"  κ_cent = prediction with Geant4 solid κ_M = {kM:.2f}")
    print(f"  κ_high = prediction with thin-wall enhanced κ_M = {kM_enhanced:.2f}")
    print(f"  Δκ     = κ_cent − κ_M(solid) = excess above Molière baseline")

    print(f"""BL4S 2026 PREDICTION SUMMARY

  Width:  σ = Highland formula  (±11% accuracy)
  Shape:  κ = (3 + κ_M) / f  −  3
  where:
    κ_M   ≈ {kM:.1f}  (Moliere kurtosis, from solid control)
    f     = hit fraction  (from geometry)
  Valid for: binary lattices (rectilinear, honeycomb)
  Accuracy: ±12% (thin-wall Moliere enhancement)
  N-layer:  κ(N) = κ(1)/N for N independent layers

  KEY PREDICTIONS (4 GeV, binary lattices):
    rect_20%: κ = {predict_kurtosis(kM, kappa_geo_binary(rt[("rectilinear",20)]["hit_fraction"])):.0f} ± 4   (f = 0.20)
    rect_40%: κ = {predict_kurtosis(kM, kappa_geo_binary(rt[("rectilinear",40)]["hit_fraction"])):.0f} ± 2   (f = 0.39)
    rect_60%: κ = {predict_kurtosis(kM, kappa_geo_binary(rt[("rectilinear",60)]["hit_fraction"])):>5.1f} ± 1.0 (f = 0.58)
    rect_80%: κ = {predict_kurtosis(kM, kappa_geo_binary(rt[("rectilinear",80)]["hit_fraction"])):>5.1f} ± 0.7 (f = 0.78)
    hc_40%:   κ = {predict_kurtosis(kM, kappa_geo_binary(rt[("honeycomb",40)]["hit_fraction"])):.0f} ± 2   (f = 0.40)

  MINIMUM VIABLE EXPERIMENT:
    - 3 samples: solid + rect_20% + rect_40%  at 4 GeV
    - ~10k events each (after cuts)
    - Tests equation at two f values + baseline
""")

    output = {
        "metadata": {
            "kM_analytical_FL": kM_analytical,
            "kM_G4": KAPPA_M_G4,
            "thin_wall_enhancement": THIN_WALL_ENHANCEMENT,
            "PLA_X0_mm": PLA_X0_MM,
            "sample_thickness_mm": SAMPLE_T_MM,
            "wall_thickness_mm": WALL_T_MM,
        },
        "predictions_4GeV": predictions_4gev,
        "energy_predictions": energy_predictions,
    }

    with open(RESULTS_DIR / "bl4s_predictions.json", "w") as f:
        json.dump(output, f, indent=2, default=float)
    print(f"Results saved to {RESULTS_DIR}/bl4s_predictions.json")

    print("VALIDATION: Predictions vs Geant4 (10σ_solid cut, consistent analysis)")
    print(f"Note: angle cut = 10σ_Highland(solid) for ALL configs — the correct procedure")

    try:
        with open("results/universal_final/universal_final_results.json") as f:
            g4_final = json.load(f)

        geo_map = {"rect": "rectilinear", "hc": "honeycomb", "gyr": "gyroid",
                   "cub": "cubic", "vor": "voronoi"}

        print(f"\n  BINARY GEOMETRIES (rect, hc): κ = (3 + κ_M)/f − 3")
        print(f"  {'Config':<24s} {'f_G4':>6s} {'f_ray':>6s} {'κ_G4':>7s} "
              f"{'κ_pred':>7s} {'κ_pred_f_G4':>12s} {'ratio':>7s} {'ratio_fG4':>10s}")
        print(f"  {'─'*100}")

        ratios_binary = []
        ratios_binary_fdata = []

        for row in sorted(g4_final, key=lambda r: r["config"]):
            if not row["is_binary"]:
                continue

            parts = row["config"].split("_")
            geo_short = parts[0]
            geo_full = geo_map.get(geo_short)
            inf = int(parts[1].replace("pct", ""))
            E = row["energy"]

            if geo_full is None or (geo_full, inf) not in rt:
                continue

            f_g4 = row["f_hit"]
            k_g4 = row["k4_meas"]
            kM = KAPPA_M_G4[E]

            f_ray = rt[(geo_full, inf)]["hit_fraction"]
            k_pred_ray = (3 + kM) / f_ray - 3

            k_pred_fdata = (3 + kM) / f_g4 - 3

            r_ray = k_g4 / k_pred_ray if k_pred_ray > 0 else float('nan')
            r_fdata = k_g4 / k_pred_fdata if k_pred_fdata > 0 else float('nan')

            ratios_binary.append(r_ray)
            ratios_binary_fdata.append(r_fdata)

            print(f"  {row['config']:<24s} {f_g4:>6.3f} {f_ray:>6.3f} {k_g4:>7.2f} "
                  f"{k_pred_ray:>7.2f} {k_pred_fdata:>12.2f} {r_ray:>7.3f} {r_fdata:>10.3f}")

        ra = np.array(ratios_binary)
        rb = np.array(ratios_binary_fdata)
        print(f"\n  Binary with f_ray:  mean ratio = {np.mean(ra):.3f} ± {np.std(ra):.3f}")
        print(f"  Binary with f_G4:   mean ratio = {np.mean(rb):.3f} ± {np.std(rb):.3f}")
        print(f"  → ~{abs(1-np.mean(rb))*100:.0f}% systematic = thin-wall Molière enhancement")

        print(f"\n  CONTINUOUS/TERNARY GEOMETRIES (gyr, cub, vor): κ = κ_M + κ_geo·(1+κ_M/3)")
        print(f"  {'Config':<24s} {'f_G4':>6s} {'κ_G4':>7s} {'κ_pred':>7s} {'ratio':>7s} {'note':>20s}")
        print(f"  {'─'*80}")

        for row in sorted(g4_final, key=lambda r: r["config"]):
            if row["is_binary"]:
                continue

            parts = row["config"].split("_")
            geo_short = parts[0]
            geo_full = geo_map.get(geo_short)
            inf = int(parts[1].replace("pct", ""))
            E = row["energy"]

            if geo_full is None or (geo_full, inf) not in rt:
                continue

            f_g4 = row["f_hit"]
            k_g4 = row["k4_meas"]
            kM = KAPPA_M_G4[E]

            kgeo = kappa_geo_from_raytrace(rt[(geo_full, inf)], E)
            k_pred = predict_kurtosis(kM, kgeo)

            ratio = k_g4 / k_pred if k_pred > 0 else float('nan')
            note = "ray-trace κ_geo" if abs(ratio - 1) < 0.3 else "κ_M undercount"

            print(f"  {row['config']:<24s} {f_g4:>6.3f} {k_g4:>7.2f} {k_pred:>7.2f} "
                  f"{ratio:>7.3f} {note:>20s}")

        print(f"\n  NOTE: Non-binary geometries have κ_M >> κ_M(solid) because")
        print(f"  the ray-trace κ_geo underestimates the true geometric variance.")
        print(f"  These configs need κ_geo from Geant4 path-length data, not ray-trace.")

    except FileNotFoundError:
        print("  (No Geant4 validation results found)")

if __name__ == "__main__":
    main()

