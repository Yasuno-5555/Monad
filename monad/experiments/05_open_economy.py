import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import SOESolver

def run_open_economy_experiment():
    print("--- Monad v5.0: The Samurai HANK Experiment ---")

    # 1. Setup Solver (Japan-like Calibration)
    # alpha=0.3: Consumption basket is 30% foreign
    # chi=0.2:   Exports are sticky / Hollowing out (Low Elasticity) to trigger "Disconnect" condition (chi < alpha)
    params = {'alpha': 0.3, 'chi': 0.2, 'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
    
    # Check for CSVs one level up
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    solver = SOESolver(path_R=path_R, path_Z=path_Z, T=50, params=params)

    # 2. Define Shock: "The Fed Hikes Rates"
    # Foreign Real Rate (r*) rises by 100bps (1%) permanently-ish
    # This causes immediate Depreciation of Domestic Currency (Q rises / Yen weak) via UIP
    rho = 0.9
    shock_r_star = 0.01 * (rho ** np.arange(50))

    # 3. Solve
    print("Solving for Foreign Rate Shock (Currency Depreciation)...")
    results = solver.solve_open_economy_shock(shock_r_star)

    # 4. Visualize: "The Disconnect"
    t = np.arange(50)
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"The 'Open Economy' Paradox (alpha={params['alpha']}, chi={params['chi']})", fontsize=16)

    # Exchange Rate (Q)
    # r* rose -> Domestic currency depreciated
    ax[0,0].plot(results['dQ']*100, color='purple', lw=2)
    ax[0,0].set_title("Real Exchange Rate ($Q$)\n(Up = Depreciation/Weak Yen)", fontsize=12)
    ax[0,0].set_ylabel("% Depreciation")
    ax[0,0].grid(True, alpha=0.3)

    # Net Exports (NX)
    # Weaker yen makes exports cheaper -> NX rises
    ax[0,1].plot(results['dNX']*100, color='green', lw=2)
    ax[0,1].set_title("Net Exports ($NX$)\n(Competitiveness Boost)", fontsize=12)
    ax[0,1].set_ylabel("% of GDP")
    ax[0,1].grid(True, alpha=0.3)

    # GDP vs Consumption (The Key Plot)
    ax[1,0].plot(results['dY']*100, label='GDP ($Y$)', color='blue', lw=2)
    ax[1,0].plot(results['dC']*100, label='Consumption ($C$)', color='red', lw=2, linestyle='--')
    ax[1,0].set_title("Macro ($Y$) vs People ($C$)\nTHE DISCONNECT", fontsize=12, fontweight='bold')
    ax[1,0].set_ylabel("% Deviation")
    ax[1,0].legend()
    ax[1,0].grid(True, alpha=0.3)
    ax[1,0].axhline(0, color='black', lw=1)
    
    # Check signs for automatic verification
    dY_0 = results['dY'][0]
    dC_0 = results['dC'][0]
    print(f"Impact at t=0: dY={dY_0*100:.3f}%, dC={dC_0*100:.3f}%")
    if dY_0 > 0 and dC_0 < 0:
        print("VERIFICATION PASS: The Disconnect is observed (Y > 0, C < 0).")
    else:
        print("VERIFICATION FAIL: Disconnect not observed.")

    # Real Labor Income (Z)
    # Why did C fall? Because Z fell.
    ax[1,1].plot(results['dZ']*100, color='orange', lw=2)
    ax[1,1].set_title("Real Labor Income ($Z$)\n(Purchasing Power Loss)", fontsize=12)
    ax[1,1].set_ylabel("% Deviation")
    ax[1,1].grid(True, alpha=0.3)
    ax[1,1].axhline(0, color='black', lw=1)

    plt.tight_layout()
    plt.savefig("open_economy_paradox.png")
    print("Plot saved to open_economy_paradox.png")
    # plt.show()

if __name__ == "__main__":
    run_open_economy_experiment()
