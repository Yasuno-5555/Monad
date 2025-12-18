from monad.dsl import model_config, shock, run, AR1
from monad.analytics import plot_irf, plot_multiplier_decomposition
import matplotlib.pyplot as plt

# 1. Define Model Configuration
@model_config(
    alpha=0.3,   # Import Share
    chi=1.0,     # Trade Elasticity
    kappa=0.05,  # Flattened Phillips Curve
    phi_pi=1.5   # Hawkish Central Bank
)
class JapanScenario:
    """Japan-like parameter set with high openness and flat PC."""
    pass

# 2. Define Shocks
@shock
def fiscal_expansion(T=50):
    """Gov Spending Shock: 1% increase, persistence 0.9"""
    dG = 0.01 * AR1(0.9, 1.0, T)
    dTrans = AR1(0.9, 0.0, T)
    return {'dG': dG, 'dTrans': dTrans}

@shock
def natural_rate_shock(T=50):
    """Recessionary Shock: r* drops by 2%, persistence 0.9"""
    return {'dr_star': -0.02 * AR1(0.9, 1.0, T)}

# 3. Run Experiments (Declarative)
try:
    print("\n=== Experiment 1: Fiscal Stimulus ===")
    res_fiscal = run(fiscal_expansion, T=40, model_type="two_asset")
    
    # Analyze
    fig_f = plot_irf(res_fiscal, variables=['dY', 'dC', 'multiplier'], title="Fiscal Expansion")
    # Decompose Multiplier (requires backend access, done inside api ideally, but here manual)
    # The DSL returns a dict. To get decomposition, we'd use the backend.
    # Future: DSL result object should have .decompose() method.
    
    print("\n=== Experiment 2: Recession (ZLB) ===")
    # Override config on the fly
    res_recession = run(natural_rate_shock, T=50, zlb=True)
    fig_r = plot_irf(res_recession, variables=['Y', 'i', 'pi'], title="Recession with ZLB")
    
    print("\nSuccess! DSL is working.")
    
except ImportError as e:
    print(f"\n[SKIP] Simulation skipped (C++ backend needed): {e}")
except Exception as e:
    print(f"\n[ERROR] {e}")
