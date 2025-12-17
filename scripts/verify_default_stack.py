import sys
import os
import shutil
import numpy as np
import pandas as pd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.model import MonadModel
from monad.solver import NKHANKSolver, SOESolver
from monad.nonlinear import NewtonSolver

def verify_default_stack():
    print("--- Verifying Default Solver Stack (Closed Economy / No ZLB) ---")
    
    # 1. Ensure Dummy Data Exists (Since we cleaned up)
    # The solver expects 50x50 matrices
    T = 50
    path_R = "gpu_jacobian_R.csv"
    path_Z = "gpu_jacobian_Z.csv"
    
    if not os.path.exists(path_R):
        print("Creating dummy cached Jacobians for verification...")
        df = pd.DataFrame(np.eye(T))
        df.to_csv(path_R, header=False, index=False)
        df.to_csv(path_Z, header=False, index=False)
        created_dummies = True
    else:
        created_dummies = False

    try:
        # 2. Initialize Model
        # This will try to run "MonadTwoAsset.exe". If missing, it uses Cache Mode.
        model = MonadModel("MonadTwoAsset.exe")
        
        # 3. Run safely
        # params=None implies "use defaults from test_model.json"
        # We assume test_model.json has "solver_settings": { "open_economy": false, "zlb": false }
        solver = model.run()
        
        # 4. Check Type
        print(f"Solver Type: {type(solver).__name__}")
        
        is_soe = isinstance(solver, SOESolver)
        is_nonlinear = isinstance(solver, NewtonSolver)
        is_standard = isinstance(solver, NKHANKSolver) and not is_soe
        
        if is_standard and not is_nonlinear:
            print("SUCCESS: Default stack is Standard NKHANKSolver.")
        else:
            print(f"FAILURE: Unexpected solver stack. SOE={is_soe}, Nonlinear={is_nonlinear}")
            sys.exit(1)
            
    finally:
        # Cleanup dummies if we created them
        if created_dummies:
            print("Cleaning up dummy files...")
            if os.path.exists(path_R): os.remove(path_R)
            if os.path.exists(path_Z): os.remove(path_Z)

if __name__ == "__main__":
    verify_default_stack()
