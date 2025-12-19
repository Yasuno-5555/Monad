import sys
import os
import numpy as np

# Adjust imports to local setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.model import MonadModel
from monad.welfare import WelfareResult

# --- "One Page Killer Example" ---

def main():
    print("=== Monad 'Killer' Demo ===")
    
    # 0. Setup Mock Environment (In Real usage, this points to real binary)
    # We patch MonadModel.run to return mock results for this demo
    # since we don't want to run the heavy C++ engine for a quick UI demo
    # but the API flow is identical.
    
    _mock_run = True
    
    if _mock_run:
        # Patch run method for demo
        def mock_run(self, params=None, config_path=None):
            # Print state to prove mutation works
            print(f"  [Engine] Running with overrides: {self.overrides}")
            
            # Return a dummy solver-like result dict
            base_val = -10.0
            if 'phi' in self.overrides and self.overrides['phi'] < 1.0:
                 base_val = -15.0 # Bad policy penalty
            
            return {
                'value': np.ones(100) * base_val,
                'distribution': np.ones(100)/100.0,
                'regimes': self.objects['solver'].parent.overrides.get('solver_settings.zlb', False)
            }
        
        MonadModel.run = mock_run

    # 1. Initialize
    m = MonadModel("bin/MonadEngine")
    
    # 2. Base Case: Standard Model
    print("\n1. Solving Baseline...")
    base_res = m.run()
    
    # 3. Counterfactual: Weak Policy + ZLB Regime
    print("\n2. Solving Counterfactual (Weak Policy + ZLB)...")
    
    # Fluent API: Mutate Policy -> Toggle ZLB -> Run
    alt_res = (
        m.object("policy_rule").mutate(phi=0.8)
         .object("solver").toggle("zlb")
         .run()
    )
    
    # 4. Analysis: Welfare Comparison
    print("\n3. Welfare Analysis...")
    # WelfareResult(alt_res).compare_with(base_res) style
    # Or just print
    
    # Using our new WelfareAPI
    WelfareResult(alt_res).compare(WelfareResult(base_res))

    # 5. History / Diff
    m.history()
    
if __name__ == "__main__":
    main()
