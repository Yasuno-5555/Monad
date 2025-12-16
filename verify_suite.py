
"""
Verification of Monad Analysis Suite v1.8
Runs scenarios and generates plots.
"""
import matplotlib.pyplot as plt
from monad.lab import ExperimentRunner
from monad.plot import MonadPlotter
import os

def run_suite():
    print("--- Monad v1.8 Analysis Suite Verification ---")
    
    # 1. Setup Runner
    runner = ExperimentRunner(base_config={
        "theta_w": 0.75,
        "phi_pi": 2.0      # Baseline Stable
    })
    
    # 2. Run Scenarios
    # Baseline
    runner.run_scenario("Baseline", {})
    
    # Hawkish (Aggressive Taylor)
    runner.run_scenario("Hawkish", {"phi_pi": 2.5})
    
    # Fiscal Dove (Less tax response) -> might be unstable? 
    # But let's try something safe like Higher Unemployment Benefit
    runner.run_scenario("GenerousUI", {"replacement_rate": 0.6})
    
    # 3. Visualization
    plotter = MonadPlotter()
    
    # Macro Panel
    fig1 = plotter.plot_macro_responses(runner)
    fig1.savefig("suite_macro.png")
    print("Saved suite_macro.png")
    
    # Inequality
    fig2 = plotter.plot_inequality_wedge(runner, metric="C_bottom50")
    fig2.savefig("suite_inequality.png")
    print("Saved suite_inequality.png")
    
    # Fiscal
    fig3 = plotter.plot_fiscal_decomposition(runner)
    fig3.savefig("suite_fiscal.png")
    print("Saved suite_fiscal.png")
    
    # Verify outputs exist
    files = ["suite_macro.png", "suite_inequality.png", "suite_fiscal.png"]
    missing = [f for f in files if not os.path.exists(f)]
    
    if not missing:
        print("SUCCESS: All plots generated.")
    else:
        print(f"FAILURE: Missing plots: {missing}")

if __name__ == "__main__":
    run_suite()
