import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from monad.solver import NKHANKSolver

def run_diag():
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    solver = NKHANKSolver(path_R=path_R, path_Z=path_Z, T=10)
    
    J = solver.backend.J_C_r
    print("--- Top-Left 5x5 of J_C_r ---")
    print(J[:5, :5])
    
    diag = np.diag(J)
    print(f"Diagonal Sum: {np.sum(diag)}")
    print(f"Is Toeplitz-like? {np.allclose(J[1:,1:], J[:-1,:-1], atol=1e-5)}")

if __name__ == "__main__":
    run_diag()
