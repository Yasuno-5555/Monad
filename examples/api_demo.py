from monad.api import Model
from monad.analytics import compare_scenarios
from monad.dsl import AR1
import matplotlib.pyplot as plt

try:
    # 1. Initialize Orchestrator
    # Cache ensures SS/Jacobians are loaded from disk if available
    mdl = Model(model_type="two_asset", T=50, cache_dir=".monad_cache")
    mdl.initialize()

    # 2. Define Shock Path (Natural Rate Shock)
    dr_star = -0.01 * AR1(0.9, 1.0, 50)
    shock = {'dr_star': dr_star}

    # 3. Compare Scenarios
    print("Running Scenario A: No ZLB...")
    res_A = mdl.run_experiment(shock, zlb=False)

    print("Running Scenario B: With ZLB...")
    res_B = mdl.run_experiment(shock, zlb=True)

    print("Running Scenario C: Robust Mode...")
    res_C = mdl.run_experiment(shock, zlb=True, robust=True)

    # 4. Compare
    fig = compare_scenarios({
        'No ZLB': res_A,
        'ZLB': res_B,
        'Robust': res_C
    }, variable='Y', title="Impact of ZLB on Output")

    print("Comparison Complete.")
    
except ImportError as e:
    print(f"Simulation skipped: {e}")
