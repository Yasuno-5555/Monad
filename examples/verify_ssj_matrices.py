import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.modeling.schema import ModelSpec
from monad.modeling.ssj import LinearSSJBuilder

# Use the same NK YAML as before
nk_yaml = """
name: NK Benchmark
type: dsge_perfect_foresight
parameters:
  sigma: 1.0     # IES
  beta: 0.99     # Discount
  kappa: 0.1     # Phillips curve slope
  phi_pi: 1.5    # Taylor rule
  r_star: 0.01   # Natural rate

variables:
  y: 0.0
  pi: 0.0
  i: 0.01

equations:
  # 0: IS: y[t] - y[t+1] + (1/sigma)*(i[t] - pi[t+1] - r_star) = 0
  - "y[t] - y[t+1] + (1/sigma)*(i[t] - pi[t+1] - r_star)"
  
  # 1: NKPC: pi[t] - beta*pi[t+1] - kappa*y[t] = 0
  - "pi[t] - beta*pi[t+1] - kappa*y[t]"
  
  # 2: Taylor: i[t] - (r_star + phi_pi*pi[t]) = 0
  - "i[t] - (r_star + phi_pi*pi[t])"
"""

def main():
    print("--- Monad SSJ Bridge Verification ---")
    
    # 1. Load Spec
    model = ModelSpec.from_yaml(nk_yaml)
    
    # 2. Build Matrices
    builder = LinearSSJBuilder(model)
    builder.compile()
    
    # 3. Get Numerical Matrices at SS (y=0, pi=0, i=r_star)
    ss = {'y': 0.0, 'pi': 0.0, 'i': model.parameters['r_star'].value}
    A, B, C = builder.get_matrices(ss)
    
    print("\n[Extracted Matrices]")
    print("Variables:", builder.var_names)
    print("A (dF/dX_{t+1}):\n", A)
    print("B (dF/dX_t):\n", B)
    print("C (dF/dX_{t-1}):\n", C)
    
    # 4. Integrity Check (Manual)
    # Vars: [y, pi, i]
    # Eq 2 (Taylor): i[t] - phi_pi*pi[t] - r_star = 0
    # dF/di_t = 1  (B[2,2])
    # dF/dpi_t = -phi_pi = -1.5 (B[2,1])
    
    phi_pi = model.parameters['phi_pi'].value
    
    val_check = B[2, 1]
    expected = -phi_pi
    
    pass_check = abs(val_check - expected) < 1e-9
    print(f"\n[Check 1] Taylor Rule Coeff (B[2,1]): {val_check} vs Expected {expected}")
    
    if pass_check:
        print("[PASS] Taylor Rule Coefficient extracted correctly.")
    else:
        print("[FAIL] Taylor Rule Coefficient mismatch.")
        
    # 5. BK Condition (Blanchard-Kahn)
    # System: A X_{t+1} + B X_t = 0  (C is zero interaction)
    # X_{t+1} = -A^{-1} B X_t
    # Eigenvalues of M = -A^{-1} B should be examined.
    # Note: Since A might be singular (static equations like Taylor Rule), we might need Generalized Schur.
    # But here:
    # IS: involves y_{t+1}, pi_{t+1}.
    # NKPC: involves pi_{t+1}.
    # Taylor: Static. i[t] only. A[2, :] is all 0.
    
    # A is singular. So we cannot invert A directly.
    # We must eliminate static variables or use generalized eigenvalues.
    # Simpler: Substitute static i[t] into IS curve.
    # IS: y[t] - y[t+1] + (1/sigma)*(phi_pi*pi[t] - pi[t+1]) = 0
    # NKPC: pi[t] - beta*pi[t+1] - kappa*y[t] = 0
    # New system for [y, pi].
    
    # Let's see if numpy can handle generalized eigenvalues? 
    # scipy.linalg.eig(a, b) solves a v = lambda b v.
    # Here: A X_{t+1} = -B X_t.
    # So we want lambda such that A v = lambda (-B) v ? No.
    # We want z_{t+1} = lambda z_t
    # A (lambda v) + B v = 0
    # (lambda A + B) v = 0
    # lambda A v = -B v
    # A v = (1/lambda) (-B) v ? 
    # Typically generalized eigenproblem is A v = lambda B v.
    # Let's map: (lambda A + B) v = 0 => -B v = lambda A v
    # So using scipy.linalg.eig(-B, A), we get lambda.
    
    from scipy.linalg import eig
    # scipy.linalg.eig(a, b) returns w, vr (eigenvalues, right eigenvectors) if left=False
    vals, vr = eig(-B, A)
    print("Generalized Eigenvalues of (-B, A):")
    # Filter infinite standard eigenvalues (due to singular A)
    # A singular -> infinite eigenvalues. These correspond to static relations (instantaneous adjustment).
    # We are interested in the dynamic eigenvalues.
    
    finite_vals = []
    for v in vals:
        if np.isfinite(v) and v != 0:
             print(f"  {v:.4f} (Abs: {abs(v):.4f})")
             finite_vals.append(v)
        elif v == 0:
             print(f"  {v:.4f} (Zero)")
        else:
             print(f"  {v} (Infinite/NaN - Static constraint)")
             
    # For determinacy in 3-var NK with Taylor Principle > 1:
    # We expect unique stable path. 
    # Since y, pi are jump variables, we need eigenvalues of transition matrix M (where x_{t+1} = M x_t) to be OUTSIDE unit circle?
    # Wait, usually notation is x_{t+1} = M x_t.
    # If M has roots > 1, and x is jump, unique solution exists (to kill unstable root).
    # If roots < 1, indeterminacy (multiple paths converge).
    # Let's check magnitude.
    
    magnitudes = [abs(v) for v in finite_vals if np.isfinite(v)]
    unstable_count = sum(1 for m in magnitudes if m > 1.0)
    print(f"Number of Unstable Eigenvalues (>1): {unstable_count}")
    print(f"Number of Jump Variables: 2 (y, pi) [i is static subst]")
    
    # Theory: For determinacy, # unstable roots = # jump variables.
    if unstable_count == 2:
        print("[PASS] Blanchard-Kahn Condition Satisfied (Determinacy).")
    else:
        print("[WARN] Indeterminacy or Explosiveness detected.")

if __name__ == "__main__":
    main()
