import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.modeling.schema import ModelSpec
from monad.modeling.solver import GenericStaticSolver

# 1. AS-AD Model Spec (YAML)
# Short-Run Equilibrium with Linear AS Curve: P = Pe + gamma * (Y - Yn)
asad_yaml = """
name: AS-AD Static Benchmark
type: static_equilibrium
parameters:
  # Goods Market (IS)
  G: 150       # Gov Spending
  T: 100       # Taxes
  c0: 50       # Auto Cons
  c1: 0.8      # MPC
  I0: 100      # Auto Invest
  Ir: 1000     # Invest Sensitivity to r
  
  # Money Market (LM)
  Ms: 1000     # Nominal Money Supply
  Ky: 1.0      # MD Income Sensitivity
  Kr: 1000     # MD Interest Sensitivity
  
  # Aggregate Supply (AS)
  Pe: 1.0      # Expected Price Level
  Yn: 1000     # Natural Output
  gamma: 0.002 # Price Sensitivity to Output Gap (Slope of AS)

variables:
  Y: 1000      # Output
  r: 0.05      # Interest Rate
  P: 1.0       # Price Level

equations:
  # IS Curve: Y = C + I + G
  # Y - (c0 + c1*(Y-T) + I0 - Ir*r + G) = 0
  - "Y - (c0 + c1*(Y - T) + I0 - Ir*r + G)"
  
  # LM Curve: Ms/P = L(Y, r)
  # Ms/P - (Ky*Y - Kr*r) = 0
  - "Ms/P - (Ky*Y - Kr*r)"
  
  # AS Curve: P = Pe + gamma*(Y - Yn)
  # P - (Pe + gamma*(Y - Yn)) = 0
  - "P - (Pe + gamma*(Y - Yn))"
"""

def main():
    print("--- Monad Generic Solver Demo: AS-AD ---")
    
    # 2. Load
    print("Loading AS-AD Model...")
    model = ModelSpec.from_yaml(asad_yaml)
    
    # 3. Compile
    print("Compiling (Sympy)...")
    solver = GenericStaticSolver(model)
    
    # 4. Solve
    print("Solving via C++ Engine...")
    res = solver.solve()
    
    if res:
        Y, r, P = res['Y'], res['r'], res['P']
        
        # 5. Simple Check
        # Check AS
        gamma = model.parameters['gamma'].value
        Yn = model.parameters['Yn'].value
        Pe = model.parameters['Pe'].value
        
        P_check = Pe + gamma * (Y - Yn)
        err = abs(P - P_check)
        
        print(f"\n[Validation]")
        print(f"  Y = {Y:.4f}")
        print(f"  r = {r:.4f}")
        print(f"  P = {P:.4f} (Calculated from AS: {P_check:.4f})")
        
        if err < 1e-6:
            print("[PASS] AS relation holds.")
        else:
            print(f"[FAIL] AS discrepancy: {err}")

if __name__ == "__main__":
    main()
