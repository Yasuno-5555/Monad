
import sys
import os
import numpy as np

# Adjust path to find monad package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from monad.cpp_backend import CppBackend

def test_rank():
    print("Testing RANK Logic (OneAssetSolver with Nz=1)...")
    
    # 1. Initialize Backend in RANK mode
    # Standard NK params: beta=0.99, r=0.0101 -> beta*(1+r) ~ 1.0 (Approx)
    # If beta*(1+r) != 1 in RANK, Euler eq implies growth/shrinkage. 
    # EGM will handle it by drifting to bounds, but we should test near steady state.
    # r = 1/beta - 1
    beta = 0.99
    r_steady = (1.0/beta) - 1.0
    
    params = {
        'beta': beta,
        'r_m': r_steady,
        'sigma': 1.0, # Log utility
        'tax_lambda': 1.0, 'tax_tau': 0.0, 'tax_transfer': 0.0
    }
    
    backend = CppBackend(T=50, model_type="rank", params=params)
    
    # Check income override
    print(f"Income Type: {type(backend._income)}")
    
    # 2. Solve Steady State
    try:
        res = backend.solve_steady_state()
        print("RANK Solved Successfully.")
        
        # 3. Analyze Results
        c_pol = np.array(res['c_pol'])
        dist = np.array(res['distribution'])
        
        print(f"Policy Grid Size: {len(c_pol)}")
        print(f"Distribution Size: {len(dist)}")
        
        # In RANK, distribution should be degenerate or uniform if no shocks? 
        # Actually with OneAssetSolver logic, it initializes uniform and iterates.
        # Ideally it converges to a single point if beta*(1+r)=1?
        # Or if grids are finite, it might be between two points.
        
        # Check if C_pol ~ C_steady
        # C_steady = r*A + Y. If A=0, C=Y.
        # Income is normalized to 1.0 in backend default construction if not loaded?
        # Actually CppBackend constructs income from params usually.
        
        print(f"Agg Liquid: {res['agg_liquid']}")
        print(f"Value Func Mean: {np.mean(res['value'])}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"FAILED: {e}")

if __name__ == "__main__":
    test_rank()
