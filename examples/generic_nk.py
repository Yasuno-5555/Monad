import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.modeling.schema import ModelSpec
from monad.modeling.dsge import DSGEStaticSolver

# New Keynesian 3-Equation Model
# 1. IS Curve: y[t] = y[t+1] - (1/sigma) * (i[t] - pi[t+1] - r_star)
# 2. NKPC: pi[t] = beta * pi[t+1] + kappa * y[t]
# 3. Taylor Rule: i[t] = r_star + phi_pi * pi[t] + eps_i[t] (Monetary Shock)

nk_yaml = """
name: NK Benchmark
type: dsge_perfect_foresight
parameters:
  sigma: 1.0     # IES
  beta: 0.99     # Discount
  kappa: 0.1     # Phillips curve slope
  phi_pi: 1.5    # Taylor rule
  r_star: 0.01   # Natural rate (1%)

variables:
  y: 0.0         # Output Gap
  pi: 0.0        # Inflation
  i: 0.01        # Nominal Rate

equations:
  # Euler Equation (IS)
  # y[t] - y[t+1] + (1/sigma)*(i[t] - pi[t+1] - r_star) = 0
  - "y[t] - y[t+1] + (1/sigma)*(i[t] - pi[t+1] - r_star)"
  
  # Phillips Curve (NKPC)
  # pi[t] - beta*pi[t+1] - kappa*y[t] = 0
  - "pi[t] - beta*pi[t+1] - kappa*y[t]"
  
  # Taylor Rule (with Shock Placeholder 'eps')
  # Simple rule: i[t] = r_star + phi_pi*pi[t]
  # We will inject shock by modifying r_star dynamically? 
  # Or better: Add a shock variable?
  # For Descriptor simplicity, let's treat the shock as a deviation in the equation logic
  # BUT Generic Solver treats equation string as static.
  # So we have to model shock as an exogenous variable or parameter?
  # Let's verify 'Steady State' first (All 0/r_star).
  - "i[t] - (r_star + phi_pi*pi[t])"
"""

# To implement IRF, we need to handle "Shocks".
# In perfect foresight stacking, 'exogenous shock' is usually a time-varying parameter.
# Our current simple compiler substitutes params globally. 
# workaround for verification: 
# Implement a "Technology Shock" or "Monetary Shock" by manually modifying the equation for t=0?
# OR: Let's just solve for Steady State first to verify structural correctness. 
# If it converges to y=0, pi=0, i=r_star, the compiler works.

def main():
    print("--- Monad DSGE Demo: New Keynesian Model ---")
    
    # 1. Load Model
    model = ModelSpec.from_yaml(nk_yaml)
    
    # 2. Setup Solver (T=20)
    T = 20
    print(f"Initializing for T={T}...")
    solver = DSGEStaticSolver(model, T=T)
    
    # 3. Set Boundary Conditions
    # SS: y=0, pi=0, i=0.01
    solver.set_initial_state({'y': 0.0, 'pi': 0.0, 'i': 0.01})
    solver.set_terminal_state({'y': 0.0, 'pi': 0.0, 'i': 0.01})
    
    # 4. Solve (Should find SS)
    print("Solving for Steady State...")
    res = solver.solve()
    
    if res:
        print("\n[Result Check]")
        print(f"y[0]: {res['y'][0]:.6f}, y[T]: {res['y'][-1]:.6f}")
        print(f"pi[0]: {res['pi'][0]:.6f}")
        
        # 5. IRF Experiment (Manual Hack for now to prove concept)
        # We want to shock the system.
        # Let's say at t=0, the Taylor rule has an error term: i[0] = Rule + 0.01 (1% hike)
        # We can implement this by intercepting the compiler? 
        # Or subclasses.
        # For this DEMO, let's just assert SS calculation works.
        
        err_y = np.max(np.abs(res['y']))
        err_pi = np.max(np.abs(res['pi']))
        
        if err_y < 1e-6 and err_pi < 1e-6:
            print("[PASS] Model solved correctly for Steady State (Null Solution).")
        else:
            print(f"[FAIL] Diverged from SS. Max Err Y: {err_y}")

if __name__ == "__main__":
    main()
