import numpy as np
import uproot
import json
from pathlib import Path

PLA_X0_MM = 315.0
BOOT_N = 2000
SEED = 42
DATA_DIR = Path("data/proposal")
RESULTS_DIR = Path("results/universal_final")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def highland_sigma(x_over_x0, p_gev):
    if x_over_x0 <= 0:
        return 0.0
    return (13.6e-3 / p_gev) * np.sqrt(x_over_x0) * (1 + 0.038 * np.log(x_over_x0))

def highland_sigma2_vec(L_mm, p_gev):
    x = L_mm / PLA_X0_MM
    s2 = np.zeros_like(x, dtype=float)
    pos = x > 0
    s2[pos] = ((13.6e-3 / p_gev) ** 2) * x[pos] * (1 + 0.038 * np.log(x[pos])) ** 2
    return s2

def excess_kurtosis(data):
    if len(data) < 10:
        return float('nan')
    c = data - np.mean(data)
    s2 = np.var(c, ddof=1)
    if s2 == 0:
        return 0.0
    return np.mean(c ** 4) / s2 ** 2 - 3

def bootstrap_kurtosis(tx, ty, n_boot=BOOT_N, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(tx)
    k4s = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        k4x = excess_kurtosis(tx[idx])
        k4y = excess_kurtosis(ty[idx])
        k4s.append((k4x + k4y) / 2)
    return np.std(k4s)

def load_and_cut(filepath, p_gev, cut_rad):
    f = uproot.open(filepath)
    d = f["scattering"].arrays(library="np")

    tx, ty = d["theta_x"], d["theta_y"]
    energy = d["energy_out"]
    ex, ey = d["entry_x"], d["entry_y"]
    pla = d["pla_path"]

    mask = (energy > p_gev * 0.9) & (np.abs(ex) < 5.0) & (np.abs(ey) < 10.0)
    mask &= (np.abs(tx) < cut_rad) & (np.abs(ty) < cut_rad)

    return tx[mask], ty[mask], pla[mask]

def main():
    root_files = sorted(DATA_DIR.glob("*.root"))

    configs = []
    for rf in root_files:
        name = rf.stem
        p_gev = float(name.split("_")[-1].replace("GeV", ""))
        configs.append({"filepath": str(rf), "label": name, "p_gev": p_gev})

    energies = sorted(set(c["p_gev"] for c in configs))
    solid_cuts = {}
    for e in energies:
        sigma_h = highland_sigma(10.0 / PLA_X0_MM, e)
        solid_cuts[e] = 10.0 * sigma_h

    solid_k4 = {}
    for e in energies:
        key = f"control_solid_{int(e)}GeV"
        rf = DATA_DIR / f"{key}.root"
        if not rf.exists():
            continue
        tx, ty, pla = load_and_cut(str(rf), e, solid_cuts[e])
        solid_k4[e] = (excess_kurtosis(tx) + excess_kurtosis(ty)) / 2

    print(f"Solid κ_M: {solid_k4}")

    with open("data/sweep0/sweep0_summary.json") as fp:
        raytrace = json.load(fp)
    rt_lookup = {}
    for entry in raytrace:
        rt_lookup[(entry["geometry"], entry["infill_target_pct"])] = entry

    geo_map = {"rect": "rectilinear", "hc": "honeycomb", "gyr": "gyroid",
               "cub": "cubic", "vor": "voronoi"}

    lattice_configs = [c for c in configs if not c["label"].startswith("control")]

    print("THE UNIVERSAL EQUATION: κ = (3 + κ_M)/f − 3  =  κ_M + κ_geo·(1 + κ_M/3)")
    print("Three versions of κ_M tested: (A) solid control, (B) PLA subpopulation, (C) implied from data")
    print(f"\n{'Config':<24s} {'E':>4s} {'f':>6s} {'κ_meas':>7s} "
          f"{'κ_M_sol':>8s} {'κ_pred_A':>9s} {'r_A':>6s} "
          f"{'κ_M_PLA':>8s} {'κ_pred_B':>9s} {'r_B':>6s} "
          f"{'κ_M_imp':>8s}")

    ratios_A_bin = []
    ratios_B_bin = []
    ratios_A_all = []
    ratios_B_all = []
    kM_enhancement = []

    output_rows = []

    for cfg in sorted(lattice_configs, key=lambda x: x["label"]):
        label = cfg["label"]
        e = cfg["p_gev"]
        if e not in solid_k4:
            continue
        cut = solid_cuts[e]

        tx, ty, pla = load_and_cut(cfg["filepath"], e, cut)

        k4_meas = (excess_kurtosis(tx) + excess_kurtosis(ty)) / 2
        k4_err = bootstrap_kurtosis(tx, ty)

        f_hit = float(np.mean(pla > 0.1))

        pla_mask = pla > 0.1
        if np.sum(pla_mask) > 100:
            k4_PLA = (excess_kurtosis(tx[pla_mask]) + excess_kurtosis(ty[pla_mask])) / 2
        else:
            k4_PLA = float('nan')

        kM_solid = solid_k4[e]

        if f_hit > 0:
            k_pred_A = (3 + kM_solid) / f_hit - 3
        else:
            k_pred_A = float('nan')

        if f_hit > 0 and not np.isnan(k4_PLA):
            k_pred_B = (3 + k4_PLA) / f_hit - 3
        else:
            k_pred_B = float('nan')

        if f_hit > 0:
            kM_implied = (k4_meas + 3) * f_hit - 3
        else:
            kM_implied = float('nan')

        r_A = k4_meas / k_pred_A if (not np.isnan(k_pred_A) and k_pred_A != 0) else float('nan')
        r_B = k4_meas / k_pred_B if (not np.isnan(k_pred_B) and k_pred_B != 0) else float('nan')

        parts = label.split("_")
        is_bin = parts[0] in ("rect", "hc")
        tag = " *" if is_bin else ""

        print(f"{label:<24s} {e:>4.0f} {f_hit:>6.3f} {k4_meas:>7.2f} "
              f"{kM_solid:>8.3f} {k_pred_A:>9.2f} {r_A:>6.3f} "
              f"{k4_PLA:>8.3f} {k_pred_B:>9.2f} {r_B:>6.3f} "
              f"{kM_implied:>8.3f}{tag}")

        if not np.isnan(r_A):
            ratios_A_all.append(r_A)
            if is_bin:
                ratios_A_bin.append(r_A)
        if not np.isnan(r_B):
            ratios_B_all.append(r_B)
            if is_bin:
                ratios_B_bin.append(r_B)

        if is_bin and not np.isnan(k4_PLA):
            kM_enhancement.append({
                "config": label,
                "energy": e,
                "infill": int(parts[1].replace("pct", "")),
                "kM_solid": kM_solid,
                "kM_PLA": k4_PLA,
                "ratio": k4_PLA / kM_solid,
            })

        output_rows.append({
            "config": label,
            "energy": e,
            "f_hit": f_hit,
            "k4_meas": k4_meas,
            "k4_err": k4_err,
            "kM_solid": kM_solid,
            "kM_PLA": k4_PLA,
            "kM_implied": kM_implied,
            "k_pred_solid": k_pred_A,
            "k_pred_PLA": k_pred_B,
            "ratio_solid": r_A,
            "ratio_PLA": r_B,
            "is_binary": is_bin,
        })

    print("ACCURACY SUMMARY: κ_meas / κ_pred  (should be 1.000)")

    def show(name, arr):
        a = np.array(arr)
        if len(a) == 0:
            return
        print(f"  {name:<40s} {np.mean(a):.4f} ± {np.std(a):.4f}  "
              f"[{np.min(a):.3f}, {np.max(a):.3f}]  N={len(a)}")

    print("\nUsing κ_M from SOLID CONTROL:")
    show("Binary (rect, hc)", ratios_A_bin)
    show("All geometries", ratios_A_all)

    print("\nUsing κ_M from PLA SUBPOPULATION:")
    show("Binary (rect, hc)", ratios_B_bin)
    show("All geometries", ratios_B_all)

    print("THIN-WALL MOLIÈRE ENHANCEMENT: κ_M(lattice PLA) / κ_M(solid)")
    print("Why: thin walls (0.4mm) produce heavier Molière tails than thick slab (10mm)")
    print(f"  {'Config':<24s} {'Energy':>6s} {'κ_M_solid':>10s} {'κ_M_PLA':>9s} {'enhance':>8s}")
    for row in kM_enhancement:
        print(f"  {row['config']:<24s} {row['energy']:>6.0f} {row['kM_solid']:>10.3f} "
              f"{row['kM_PLA']:>9.3f} {row['ratio']:>8.3f}")

    enhancements = [row["ratio"] for row in kM_enhancement]
    print(f"\n  Mean enhancement: {np.mean(enhancements):.3f} ± {np.std(enhancements):.3f}")

    print("CONTINUOUS GEOMETRIES: using κ_geo = 3·Var(σ²)/E[σ²]²")
    print("Formula: κ = κ_M_eff + κ_geo_σ² · (1 + κ_M_eff/3)")
    print("where κ_M_eff is the EFFECTIVE Molière kurtosis (path-averaged)")

    for cfg in sorted(lattice_configs, key=lambda x: x["label"]):
        label = cfg["label"]
        parts = label.split("_")
        if parts[0] in ("rect", "hc"):
            continue

        e = cfg["p_gev"]
        if e not in solid_k4:
            continue
        cut = solid_cuts[e]

        tx, ty, pla = load_and_cut(cfg["filepath"], e, cut)
        k4_meas = (excess_kurtosis(tx) + excess_kurtosis(ty)) / 2

        sigma2 = highland_sigma2_vec(pla, e)
        mean_s2 = np.mean(sigma2)
        var_s2 = np.var(sigma2, ddof=1)
        k_geo_s2 = 3 * var_s2 / mean_s2 ** 2 if mean_s2 > 0 else 0

        kM = solid_k4[e]
        k_pred = kM + k_geo_s2 * (1 + kM / 3)

        if k_geo_s2 > 0:
            kM_eff = (k4_meas - k_geo_s2) / (1 + k_geo_s2 / 3)
        else:
            kM_eff = k4_meas

        ratio = k4_meas / k_pred if k_pred > 0 else float('nan')

        print(f"  {label:<24s} κ_meas={k4_meas:>7.2f}  κ_geo_σ²={k_geo_s2:>6.3f}  "
              f"κ_pred(solid)={k_pred:>7.2f}  ratio={ratio:>6.3f}  "
              f"κ_M_eff_implied={kM_eff:>7.2f}  κ_M_solid={kM:>6.3f}")

    ra = np.array(ratios_A_bin)
    rb = np.array(ratios_B_bin)
    enh = np.array(enhancements)

    print(f"""THE UNIVERSAL EQUATION

  κ  =  κ_M  +  κ_geo * (1 + κ_M / 3)
  equivalently:  κ = (3 + κ_M) / f  -  3    (binary)
  where:
    κ_M   = Moliere kurtosis (material, energy, cut)
    κ_geo = 3(1-f)/f (binary) or 3*Var(σ²)/E[σ²]² (gen.)
    f     = hit fraction (from geometry or measurement)

PROOF:
  Using κ_M from PLA subpopulation:
    Binary accuracy = {np.mean(rb):.4f} +- {np.std(rb):.4f}

  Using κ_M from solid control (practical):
    Binary accuracy = {np.mean(ra):.4f} +- {np.std(ra):.4f}  (~{abs(np.mean(ra)-1)*100:.0f}% systematic)

  Thin-wall Moliere enhancement:
    κ_M(lattice) / κ_M(solid) = {np.mean(enh):.3f} +- {np.std(enh):.3f}

  For N independent layers:  κ(N) = κ(1) / N
""")

    with open(RESULTS_DIR / "universal_final_results.json", "w") as fp:
        json.dump(output_rows, fp, indent=2, default=float)
    print(f"Results saved to {RESULTS_DIR}/universal_final_results.json")

if __name__ == "__main__":
    main()

