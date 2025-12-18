import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import NKHANKSolver

def run_forward_guidance():
    print("--- Monad Lab: Forward Guidance Experiment ---")
    
    # Check for CSVs one level up
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    solver = NKHANKSolver(path_R=path_R, path_Z=path_Z, T=50)

    # 1. Define Shocks
    # Scenario A: Immediate Cut (Standard)
    shock_now = np.zeros(50)
    shock_now[0] = -0.0025  # -25bps now

    # Scenario B: Forward Guidance (Promise to cut in 1 year / 4 quarters)
    shock_future = np.zeros(50)
    shock_future[4] = -0.0025 # -25bps at t=4 (News Shock)

    # 2. Solve
    print("Solving Scenario A: Immediate Cut...")
    res_now = solver.solve_monetary_shock(shock_now)

    print("Solving Scenario B: Forward Guidance (t=4)...")
    res_future = solver.solve_monetary_shock(shock_future)

    # 3. Visualize Comparison
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # Interest Rate Path (Input)
    ax[0].plot(res_now['shock']*10000, label='Immediate Cut', linestyle='--', color='gray')
    ax[0].plot(res_future['shock']*10000, label='Forward Guidance (t=4)', color='blue', linewidth=2)
    ax[0].set_title("Policy Path ($r^{exo}$)")
    ax[0].set_ylabel("Basis Points")
    ax[0].legend()

    # Consumption Response (Output)
    # Key check: Does C rise at t=0 even though rates interpret cut at t=4?
    ax[1].plot(res_now['dC']*100, label='Immediate Cut', linestyle='--', color='gray')
    ax[1].plot(res_future['dC']*100, label='Forward Guidance Effect', color='blue', linewidth=2)
    ax[1].set_title("Consumption Response ($C$)")
    ax[1].set_ylabel("% Deviation")
    
    # Annotation
    fg_impact_t0 = res_future['dC'][0]*100
    print(f"Forward Guidance Impact at t=0: {fg_impact_t0:.4f}%")
    
    ax[1].annotate('Anticipation Effect', xy=(0, fg_impact_t0), xytext=(5, 0.05),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    ax[1].legend()

    for a in ax: a.grid(True, alpha=0.3); a.axhline(0, color='k', lw=1)
    plt.suptitle("Forward Guidance Power: Immediate vs Future Rate Cut")
    plt.tight_layout()
    plt.savefig("forward_guidance.png")
    print("Plot saved to forward_guidance.png")
    # plt.show()

if __name__ == "__main__":
    run_forward_guidance()
