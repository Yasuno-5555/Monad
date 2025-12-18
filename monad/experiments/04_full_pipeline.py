import sys
import os
import numpy as np

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.model import MonadModel
from monad.plots import plot_impulse_responses

def run_pipeline():
    print("--- Monad v4.0: Full Pipeline Test (Step 4) ---")
    
    # Path to the compiled executable
    # Adjust this path if your build setup differs (e.g., Release/Debug)
    exe_path = os.path.join("build_phase3", "Release", "MonadTwoAssetCUDA.exe")
    
    # 1. Initialize Model Wrapper
    model = MonadModel(binary_path=exe_path, working_dir=".")
    
    # 2. Run Engine & Get Solver
    # We can perform parameter sweeps here easily!
    # Let's verify with standard parameters.
    # Note: Ensure test_model.json exists in root.
    solver = model.run()

    # 3. Define Analysis (Monetary Shock)
    rho = 0.8
    shock = 0.0025 * (rho ** np.arange(50))

    # 4. Solve GE
    print("Solving General Equilibrium...")
    results = solver.solve_monetary_shock(shock)

    # 5. Result
    print(f"Computed Output Gap: {results['dY'][0]*100:.3f}%")
    
    # 6. Visualize
    plot_impulse_responses(results, title="Pipeline Verification (+25bps)")

if __name__ == "__main__":
    run_pipeline()
