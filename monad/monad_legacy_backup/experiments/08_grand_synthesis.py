import sys, os
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import SOESolver
from monad.nonlinear import NewtonSolver

def run_project_izanagi():
    print("--- Monad Final: Project Izanagi (SOE at ZLB) ---")

    # 1. Setup: Japan-style parameters (Import dependent)
    # alpha=0.3: Import share
    # chi=0.2: Sticky exports
    params = {'alpha': 0.3, 'chi': 0.2, 'kappa': 0.05, 'beta': 0.99, 'phi_pi': 1.5}
    
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    # Linear engine (Open Economy)
    linear_soe = SOESolver(path_R=path_R, path_Z=path_Z, T=50, params=params)
    
    # Nonlinear wrapper
    solver = NewtonSolver(linear_soe, max_iter=150)

    # 2. Shock: Global Recession
    # Natural rate crashes to -1.5% (Deep recession pressure)
    # 1.5% negative shock
    shock_r_natural = -0.015 * (0.9 ** np.arange(50))

    # 3. Solve
    print("Solving Nonlinear SOE at ZLB...")
    # Passing shock_r_natural also as the foreign rate shock implicitly in NewtonSolver logic
    results = solver.solve_nonlinear(shock_path=shock_r_natural)

    # 4. Analysis
    # Does the Exchange Rate (Depreciation) offset the ZLB pain?
    # Or does the Income Effect from imports make the Liquidity Trap worse?
    dY = results['Y']
    dC = results['C_agg']
    dQ = results.get('Q', np.zeros_like(dY)) # Exchange Rate
    
    min_Y = np.min(dY)*100
    min_C = np.min(dC)*100
    
    print(f"Peak Recession Depth (Y): {min_Y:.2f}%")
    print(f"Peak Consumption Drop (C): {min_C:.2f}%")
    
    if min_C < min_Y:
        print("RESULT: Consumption fell HARDER than GDP. The paradox holds even in ZLB.")
    
    # Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Policy Rate
    r_ss = 0.005
    nominal_rate = (r_ss + results['i']) * 400
    ax[0].plot(nominal_rate, label='Nominal Rate', lw=2, color='black')
    ax[0].set_title("Policy Rate (Stuck at 0?)")
    ax[0].axhline(0, color='gray', ls=':')
    ax[0].set_ylabel("Annual %")

    # Exchange Rate
    ax[1].plot(dQ*100, color='purple', lw=2)
    ax[1].set_title("Real Exchange Rate ($Q$)\n(Up = Depreciation)")
    ax[1].set_ylabel("% Depreciation")

    # The Outcome
    ax[2].plot(dY*100, label='GDP', color='blue', lw=2)
    ax[2].plot(dC*100, label='Cons', color='red', ls='--', lw=2)
    ax[2].set_title("The Final Outcome (Y vs C)")
    ax[2].legend()
    ax[2].set_ylabel("% Deviation")
    ax[2].axhline(0, color='black', lw=1)

    for a in ax: a.grid(True, alpha=0.3)
    
    plt.suptitle("Project Izanagi: The Japanification Simulation\n(ZLB + Import Dependence)", fontsize=14)
    plt.tight_layout()
    plt.savefig("project_izanagi.png")
    print("Plot saved to project_izanagi.png")
    # plt.show()

if __name__ == "__main__":
    run_project_izanagi()
