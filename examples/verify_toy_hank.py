import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.modeling.schema import ModelSpec
from monad.modeling.ssj import LinearSSJBuilder

# Toy HANK YAML
# Includes implicit 'household' block.
# Equations reference block outputs 'C' and 'A'.
toy_hank_yaml = """
name: Toy HANK Benchmark
type: hank_ssj
parameters:
  phi_pi: 1.5
  rho_r: 0.8
  r_star: 0.01

variables:
  y: 0.0
  pi: 0.0
  i: 0.01
  r: 0.01
  w: 1.0
  C: 1.0   # Block Output
  A: 0.0   # Block Output

blocks:
  household:
    type: heterogeneous
    kernel: TwoAssetFake
    inputs: [r, w]
    outputs: [C, A]

equations:
  # Market Clearing
  # y[t] = C[t] (Simple economy)
  - "y[t] - C[t]"
  
  # Fisher Equation
  # i[t] = r[t] + pi[t+1]
  - "i[t] - (r[t] + pi[t+1])"
  
  # Taylor Rule
  - "i[t] - (r_star + phi_pi*pi[t])"
  
  # Wage (Linear production)
  - "w - 1"
"""

def main():
    print("--- Monad Toy HANK Verification ---")
    
    # 1. Load Spec
    print("Loading Spec...")
    model = ModelSpec.from_yaml(toy_hank_yaml)
    
    # Check block parsing
    if 'household' in model.blocks:
        print("[PASS] Block 'household' parsed successfully.")
    else:
        print("[FAIL] Block parsing failed.")
        return

    # 2. Build Matrices
    builder = LinearSSJBuilder(model)
    builder.compile()
    
    # 3. Get Explicit Matrices (A,B,C)
    # These represent the derivatives of the equations w.r.t variables.
    # Note: C[t] is treated as a variable in the derivative.
    ss = {'y': 1.0, 'pi': 0.0, 'i': 0.01, 'r': 0.01, 'w': 1.0, 'C': 1.0, 'A': 5.0}
    A, B, C = builder.get_matrices(ss)
    
    print("\n[Explicit Matrices Structure]")
    # B matrix (Current time)
    # d(y-C)/dC_t should be -1.
    # Indexes: y=0, pi=1, i=2, r=3, w=4, C=5, A=6
    
    idx_C = builder.var_names.index('C')
    idx_eq_y = 0 # First equation
    
    val_check = B[idx_eq_y, idx_C]
    print(f"B[{idx_eq_y},{idx_C}] (d(y-C)/dC): {val_check}")
    
    if abs(val_check + 1.0) < 1e-9:
        print("[PASS] Explicit derivative w.r.t block variable correct.")
    else:
        print("[FAIL] Derivative error.")
        
    # 4. Get Block Jacobians
    T = 20
    block_jacs = builder.get_block_jacobians(T=T)
    
    hh_jacs = block_jacs.get('household', {})
    print(f"\n[Block Jacobians]")
    print(f"Keys: {list(hh_jacs.keys())}")
    
    # Check if 'dC_dr' exists
    if 'dC_dr' in hh_jacs:
        J = hh_jacs['dC_dr']
        print(f"J_C_r Shape: {J.shape}")
        
        # Verify Density
        sparsity = np.count_nonzero(J) / J.size
        print(f"Sparsity: {sparsity:.2f} (Should be > 0.05 for dense-ish)")
        
        if sparsity > 0.1: # Our mock puts diag + upper tri
            print("[PASS] Block Jacobian is Dense (as expected for HA).")
        else:
            print("[WARN] Jacobian seems sparse?")
    else:
        print("[FAIL] Missing dC_dr Jacobian.")

if __name__ == "__main__":
    main()
