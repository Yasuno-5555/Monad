import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from monad.solver import NKHANKSolver

def run_compare():
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    # 1. Base
    solver1 = NKHANKSolver(path_R=path_R, path_Z=path_Z, T=50, 
                           params={'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5})
    
    # 2. Sticky
    solver2 = NKHANKSolver(path_R=path_R, path_Z=path_Z, T=50, 
                           params={'kappa': 0.01, 'beta': 0.99, 'phi_pi': 1.5})

    shock = 0.0025 * (0.8 ** np.arange(50))

    res1 = solver1.solve_monetary_shock(shock)
    res2 = solver2.solve_monetary_shock(shock)
    
    y1 = res1['dY'][0]
    y2 = res2['dY'][0]
    
    print(f"VAL1: {y1}")
    print(f"VAL2: {y2}")
    
    if abs(y1 - y2) > 1e-9:
        print("RESULT: DIFFERENT (Engine is Active)")
    else:
        print("RESULT: IDENTICAL (Engine is Static)")

if __name__ == "__main__":
    run_compare()
