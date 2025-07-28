# main.py

import numpy as np
from data_loader import load_coordinate_data
from data_preprocessor import normalize_and_derive_coordinates
from models import (
    MaterialProperties, WaveVectorData, PhysicsCalculator,
    NonlinearFitter, FittingFunctions, SymbolicCalculator
)
from visualizer import Visualizer

def main():
    """Main analysis pipeline."""
    print("=== Materials Science Analysis Pipeline Starting ===\n")

    # 1. Initialize properties and load data
    props = MaterialProperties()
    S1xyz, S2xyz, Moxyz_raw = load_coordinate_data()
    S1xyz_p, S2xyz_p, Moxyz_p = S1xyz.copy(), S2xyz.copy(), Moxyz_raw.copy()

    # 2. Preprocess Data
    print("1. Preprocessing and normalizing coordinate data...")
    processed_data = normalize_and_derive_coordinates(S1xyz_p, S2xyz_p, Moxyz_p, props.n, props.m)

    # 3. Setup Models and Calculators
    wave_data = WaveVectorData(
        kk=np.array([1.4399, 1.43991, 1.44131, 1.44331, 1.44583, 1.44905,
                      1.45278, 1.45746, 1.46252, 1.46845, 1.47493])
    )
    physics_calc = PhysicsCalculator(props)
    fitter = NonlinearFitter(wave_data)
    symbolic_calc = SymbolicCalculator()
    visualizer = Visualizer(wave_data)

    # 4. Perform Physics Calculations
    print("2. Calculating dipole moments and center of mass...")
    pA, pB = physics_calc.calculate_dipole_moments(processed_data)
    rA, rB = physics_calc.calculate_center_of_mass(processed_data)

    p = physics_calc.combine_arrays(pA, pB, props.n, props.m)
    r = physics_calc.combine_arrays(rA, rB, props.n, props.m)

    px, _, pz = p.transpose(2, 0, 1)
    rx, _, rz = r.transpose(2, 0, 1)

    U = np.array([r[i] - r[0] for i in range(props.n)])
    Ux, _, Uz = U.transpose(2, 0, 1)

    # 5. Perform Nonlinear Fits
    print("3. Performing nonlinear fits...")
    fit_results = {}

    # CORRECTED FIT ASSIGNMENTS
    params_Uz, r2_Uz = fitter.fit_data(rx, Uz, FittingFunctions.f1, [1.8])
    fit_results['Uz'] = {'params': params_Uz, 'r2': r2_Uz}

    params_Ux, r2_Ux = fitter.fit_data(rx, Ux, FittingFunctions.f2, [-0.1, 3.6], use_kx=True)
    fit_results['Ux'] = {'params': params_Ux, 'r2': r2_Ux}

    params_pz, r2_pz = fitter.fit_data(rx, pz, FittingFunctions.f3, [1.8/30, 0])
    fit_results['pz'] = {'params': params_pz, 'r2': r2_pz}
    
    initial_px = [params_Ux[i][1] * 0.012 for i in range(props.n)]
    params_px, r2_px = fitter.fit_data(rx, px, FittingFunctions.f4, [np.mean(initial_px), 0.0002, 0.001], use_kx=True)
    fit_results['px'] = {'params': params_px, 'r2': r2_px}


    print("\n4. Fit Quality (R-squared values):")
    for name, results in fit_results.items():
        r2_values = results['r2']
        print(f"  - R² for {name}: {np.array(r2_values)[1:]}")

    # 6. Perform Symbolic Analysis
    print("\n5. Solving symbolic equations...")
    coefficients = symbolic_calc.solve_coefficients()
    print(f"  - Solved symbolic coefficients: {coefficients}")

    print("\n6. Computing final physical parameters...")
    U0_vals = np.array([p[0] for p in fit_results['Uz']['params']])[1:]
    V0_vals = np.array([p[0] for p in fit_results['Ux']['params']])[1:]
    W0_vals = np.array([p[1] for p in fit_results['Ux']['params']])[1:]
    dzs_vals = np.array([p[0] for p in fit_results['pz']['params']])[1:]
    dzc_vals = np.array([p[1] for p in fit_results['pz']['params']])[1:]
    dxs_vals = np.array([p[0] for p in fit_results['px']['params']])[1:]
    dxc_vals = np.array([p[1] for p in fit_results['px']['params']])[1:]
    k_vals = wave_data.kk[1:]

    try:
        f3131 = [float(coefficients['f3131'].subs({'U0': u, 'dzc': d, 'k': k})) for u, d, k in zip(U0_vals, dzc_vals, k_vals)]
        e331 = [float(coefficients['e331'].subs({'U0': u, 'dzs': d, 'k': k})) for u, d, k in zip(U0_vals, dzs_vals, k_vals)]
        f1111 = [float(coefficients['f1111'].subs({'W0': w, 'dxs': d, 'k': k})) for w, d, k in zip(W0_vals, dxs_vals, k_vals)]
        e111 = [float(coefficients['e111'].subs({'W0': w, 'dxc': d, 'k': k})) for w, d, k in zip(W0_vals, dxc_vals, k_vals)]

        print("\n  - Final Scaled Results:")
        print(f"    f3131 (×100): {100 * np.array(f3131)}")
        print(f"    e331  (×1000): {1000 * np.array(e331)}")
        print(f"    f1111 (×100): {100 * np.array(f1111)}")
        print(f"    e111  (×1000): {1000 * np.array(e111)}")
    except Exception as e:
        print(f"Error during final symbolic calculation: {e}")

    # 7. Generate Visualizations
    print("\n7. Generating visualizations...")
    visualizer.plot_wave_vector()
    visualizer.plot_fits(rx, Uz, FittingFunctions.f1, fit_results['Uz']['params'], "Uz(x) Fitting Results")
    visualizer.plot_fits(rx, Ux, FittingFunctions.f2, fit_results['Ux']['params'], "Ux(x) Fitting Results")
    visualizer.plot_fits(rx, pz, FittingFunctions.f3, fit_results['pz']['params'], "pz(x) Fitting Results")
    visualizer.plot_fits(rx, px, FittingFunctions.f4, fit_results['px']['params'], "px(x) Fitting Results")

    print("\n=== Analysis complete! ===")

if __name__ == "__main__":
    main()
