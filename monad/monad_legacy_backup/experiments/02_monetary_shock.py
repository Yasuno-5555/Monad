import sys
import os
import numpy as np

# Add parent directory to path so we can import 'monad'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import NKHANKSolver
from monad.plots import plot_impulse_responses, plot_decomposition

def run_experiment():
    # 1. Setup
    print("--- Monad Lab: Monetary Policy Experiment ---")
    
    # Check for CSVs one level up
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    solver = NKHANKSolver(
        path_R=path_R,
        path_Z=path_Z, 
        T=50
    )

    # 2. Define Shock (25bps hike, persistent)
    rho = 0.8
    shock = 0.0025 * (rho ** np.arange(50))

    # 3. Solve
    print("Solving General Equilibrium...")
    results = solver.solve_monetary_shock(shock)

    # 4. Analyze
    print(f"Impact on Output: {results['dY'][0]*100:.3f}%")
    print(f"Impact on Inflation: {results['dpi'][0]*100:.3f}%")

    # 5. Visualize
    plot_impulse_responses(results, title="Monetary Tightening (+25bps)")
    plot_decomposition(solver, results)

if __name__ == "__main__":
    run_experiment()
