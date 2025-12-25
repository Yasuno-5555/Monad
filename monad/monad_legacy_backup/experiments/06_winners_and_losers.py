import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import SOESolver

def run_distributional_analysis():
    print("--- Monad v5.0: Distributional Impact Analysis ---")

    # 1. Setup & Solve (Same "Samurai Paradox" Scenario)
    # alpha=0.3 (Import Share), chi=0.2 (Low Export Elasticity)
    params = {'alpha': 0.3, 'chi': 0.2, 'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
    
    # Check for CSVs one level up
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")

    solver = SOESolver(path_R=path_R, path_Z=path_Z, T=50, params=params)

    # Foreign Rate Shock (Depreciation)
    rho = 0.9
    shock_r_star = 0.01 * (rho ** np.arange(50))
    res = solver.solve_open_economy_shock(shock_r_star)

    # 2. Reconstruct Heterogeneity
    # We project the aggregate shocks onto different agent types based on HANK theory.
    # Type A: Hand-to-Mouth (Poor) - Exposed to Real Income (Z)
    # Type B: Wealthy Savers (Rich) - Exposed to Real Interest Rate (r)

    # Income Path (Z) implies the pain from Import Inflation
    dZ_path = res['dZ'] 
    # Real Rate Path (r) - Domestic rates are held fixed in this partial eq, but prices change
    # Real Rate = i (fixed) - pi (inflation). 
    # Note: In the simple SOE solver, we simplified dC directly. 
    # For visualization, we assume domestic real rates didn't spike (Central Bank inaction).
    dr_path = np.zeros(50) 

    # --- Create Synthetic Distribution Heatmap ---
    # Y-axis: Asset Percentile (0=Poorest, 100=Richest)
    # X-axis: Time
    # Color: Consumption Change

    n_percentiles = 100
    heatmap_data = np.zeros((n_percentiles, 50))

    # Interpolate MPC across distribution
    # Poor (0%) have MPC ~ 0.5 (Quarterly), Rich (100%) have MPC ~ 0.05
    # Adjusted range slightly to match "0.8 to 0.05" comment in original prompt code if intended, 
    # but the prompt code said "0.8, 0.05". Let's stick to the prompt's code values.
    mpc_profile = np.linspace(0.8, 0.05, n_percentiles)

    for p in range(n_percentiles):
        mpc = mpc_profile[p]
        # HANK Consumption Function Approximation:
        # dC_i = MPC_i * dIncome + (1 - MPC_i) * Substitution_Effect(dr)
        # Since dr ~ 0, the second term is negligible. The pain is purely from Income.
        
        # "dIncome" here is the aggregate Real Labor Income Z
        dC_i = mpc * dZ_path
        
        # Store in matrix (Flip Y so 100 is at top)
        heatmap_data[n_percentiles - 1 - p, :] = dC_i * 100 # in %

    # 3. Visualize
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot Heatmap
    # Using a "RdBu" colormap: Red = Consumption Drop (Pain), Blue = Consumption Rise (Gain)
    # Since dZ is likely negative (import inflation), dC_i will be negative -> Red
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu', vmin=-1.0, vmax=1.0,
                   extent=[0, 50, 0, 100])

    # Styling
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Consumption Change (%)', rotation=270, labelpad=15)

    ax.set_title("The 'Samurai Paradox' Distributional Impact\n(Who bears the burden of Depreciation?)", fontsize=14)
    ax.set_xlabel("Quarters since Shock")
    ax.set_ylabel("Asset Percentile (0=Poor, 100=Rich)")

    # Annotations
    ax.text(25, 10, "Working Class (High MPC)\nCrucified by Import Prices", 
            color='black', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='red'))

    ax.text(25, 90, "Wealthy Class (Low MPC)\nInsulated from Income Shock", 
            color='black', ha='center', fontsize=10, fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

    plt.tight_layout()
    plt.savefig("winners_and_losers.png")
    print("Plot saved to winners_and_losers.png")
    # plt.show()

if __name__ == "__main__":
    run_distributional_analysis()
