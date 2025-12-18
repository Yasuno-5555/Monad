import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.modeling.schema import ModelSpec
from monad.modeling.solver import GenericStaticSolver

# 1. The "Model File" (YAML)
# Ideally this would be 'examples/islm.yaml'
islm_yaml = """
name: IS-LM Benchmark
type: static_equilibrium
parameters:
  G: 100       # Gov Spending
  T: 100       # Taxes
  Ms: 500      # Money Supply
  P: 1.0       # Price Level
  c1: 0.8      # MPC
  c0: 50       # Autonomous Consumption
  I0: 150      # Autonomous Investment
  Ir: 1000     # Interest Sensitivity
  Ky: 1.0      # Money Demand Income Sensitivity
  Kr: 1000     # Money Demand Interest Sensitivity

variables:
  Y: 1000      # Output (Guess)
  r: 0.05      # Interest Rate (Guess)

equations:
  # Goods Market (Residual = 0)
  - "Y - (c0 + c1*(Y - T) + I0 - Ir*r + G)"
  
  # Money Market (Residual = 0)
  # Ms/P = Ky*Y - Kr*r  => Ms/P - Demand = 0
  - "Ms/P - (Ky*Y - Kr*r)"
"""

def main():
    print("--- Monad Generic Solver Demo: IS-LM ---")
    
    # 2. Load Model
    print("Loading Model Specification...")
    model = ModelSpec.from_yaml(islm_yaml)
    
    # 3. Initialize Generic Solver
    print("Initializing Solver (Sympy Compilation)...")
    solver = GenericStaticSolver(model)
    
    # 4. Solve
    result = solver.solve()
    
    # 5. Verification (Analytical)
    # IS: Y = c0 + c1(Y-T) + I0 - Ir*r + G
    #     Y(1-c1) + Ir*r = A_aut  where A_aut = c0 - c1T + I0 + G
    # LM: Ms/P = Ky*Y - Kr*r
    #     -Ky*Y + Kr*r = -Ms/P
    
    # Matrix:
    # [ (1-c1)   Ir ] [ Y ] = [ A_aut ]
    # [ -Ky      Kr ] [ r ]   [ -Ms/P ]
    
    p = model.parameters
    A_aut = p['c0'].value - p['c1'].value*p['T'].value + p['I0'].value + p['G'].value
    real_m = p['Ms'].value / p['P'].value
    
    det = (1 - p['c1'].value) * p['Kr'].value + p['Ky'].value * p['Ir'].value
    Y_exact = (A_aut * p['Kr'].value + p['Ir'].value * real_m) / det
    r_exact = (-(1 - p['c1'].value) * (-real_m) + p['Ky'].value * A_aut) / det # Cramer's rule check? 
    # det * r = (1-c1)*(-Ms/P) - (-Ky)*A_aut ? No.
    # Inverse:
    # 1/det * [ Kr  -Ir ] [ A ]
    #         [ Ky 1-c1 ] [-M ]
    # r = 1/det * (Ky*A + (1-c1)*(-M))  <-- Wait, LM is Ms/P - (KyY - Krr) = 0
    # So Ms/P = KyY - Krr
    # Krr = KyY - Ms/P
    # r = (Ky/Kr)Y - (1/Kr)(Ms/P)
    
    # Let's rely on Python to compute exact inverse for verification
    import numpy as np
    M = np.array([
        [(1 - p['c1'].value), p['Ir'].value],
        [-p['Ky'].value,      p['Kr'].value]
    ])
    B = np.array([A_aut, -real_m])
    sol_exact = np.linalg.solve(M, B)
    
    print("\n--- Verification ---")
    print(f"Numerical: Y = {result['Y']:.4f}, r = {result['r']:.4f}")
    print(f"Analytic : Y = {sol_exact[0]:.4f}, r = {sol_exact[1]:.4f}")
    
    err_y = abs(result['Y'] - sol_exact[0])
    err_r = abs(result['r'] - sol_exact[1])
    
    if err_y < 1e-6 and err_r < 1e-6:
        print("\n[PASS] Solution matches Analytical benchmarks.")
    else:
        print("\n[FAIL] Discrepancy detected.")

if __name__ == "__main__":
    main()
