"""
Canonical Example: Advanced Regime & Welfare Analysis
Demonstrates:
1. Setting up a model with different Regimes (ZLB, Fiscal)
2. Comparing Welfare across different policy rules
3. Using the Fluent Objects API
"""

import sys
import os

# Ensure monad is in path (for development repo)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from monad.model import MonadModel
from monad.welfare import WelfareResult

def main():
    print("--- Canonical: Regime & Welfare ---")
    
    # 1. Initialize
    m = MonadModel("bin/MonadEngine")
    
    # 2. Define Scenarios
    
    # Scenario A: Baseline
    # Standard Taylor Rule (phi=1.5), No ZLB check (Linear)
    print("Solving Baseline...")
    res_base = m.run()
    
    # Scenario B: "Dove" Policy in Crisis
    # Phi=1.1, ZLB is Active (Non-linear Solver used)
    print("Solving Dove (phi=1.1) with ZLB...")
    res_dove = (
        m.object("policy_rule").mutate(phi=1.1)
         .object("solver").toggle("zlb")
         .run()
    )
    
    # Scenario C: "Hawk" Policy in Crisis
    # Phi=2.0, ZLB is Active
    # Note: We must reset mutation or mutate from fresh 'm' if 'm' preserves state
    # MonadModel.run() writes config but 'm' keeps overrides if we don't clear them?
    # Our implementation keeps overrides. Ideally we might want .reset() or use fresh object.
    # For now, we manually set phi=2.0 which overrides the previous 1.1
    print("Solving Hawk (phi=2.0) with ZLB...")
    res_hawk = (
        m.object("policy_rule").mutate(phi=2.0)
         .object("solver").toggle("zlb")
         .run()
    )
    
    # 3. Welfare Comparison (Consumption Equivalent Variation)
    # Compare Dove and Hawk against Baseline
    
    print("\n--- Welfare Analysis (CEV) ---")
    print("Positive CEV means Alternative is better than Base.")
    
    cev_dove = WelfareResult(res_dove).compare(WelfareResult(res_base))
    cev_hawk = WelfareResult(res_hawk).compare(WelfareResult(res_base))
    
    print(f"\nSummary:")
    print(f"Dove Policy CEV: {cev_dove*100:.4f}%")
    print(f"Hawk Policy CEV: {cev_hawk*100:.4f}%")
    
    if cev_hawk > cev_dove:
         print("=> Hawk policy is preferred in this crisis scenario.")
    else:
         print("=> Dove policy is preferred.")

    # 4. Audit History
    m.history()

if __name__ == "__main__":
    # Mocking run for demonstration if binary missing
    if not os.path.exists(MonadModel("").binary_path) and not os.path.exists("bin/MonadEngine"):
         import numpy as np
         print("[Note] Running in Mock Mode for Demonstration")
         def mock_run(self, params=None, config_path=None):
             phi = self.overrides.get('phi', 1.5)
             val = -10.0
             if phi < 1.0: val = -15.0
             elif phi > 1.8: val = -9.0 # Hawk better?
             return {'value': val*np.ones(50), 'distribution': np.ones(50)/50}
         MonadModel.run = mock_run

    main()
