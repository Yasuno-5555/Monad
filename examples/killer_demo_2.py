"""
Killer Demo 2: The Anchor Slips (1970s Great Inflation)
========================================================
Demonstrates Policy-as-Object HANK with Learning Engine.

Scenario:
- Shock: Cost-push (markup) shock
- Case A (Anchored): CB knows r* is constant (gain=0)
- Case B (Unanchored): CB learns r* incorrectly (gain>0)
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.solver import NKHANKSolver
from monad.nonlinear import PiecewiseSolver
from monad.policy import Rules

def run_scenario(name, gain, linear_solver, shock):
    """Run a single scenario with given learning gain."""
    print(f"\n--- Running: {name} (gain={gain}) ---")
    
    fed = Rules.InertialTaylor(phi_pi=1.5, rho=0.8, name="Fed")
    fed.learns("r_star", gain=gain)
    
    solver = PiecewiseSolver(linear_solver, policy_block=fed, max_iter=200, tol=0.5, damping=0.3)
    result = solver.solve(shock)
    
    return result

def main():
    print("=" * 60)
    print("  Killer Demo 2: The Anchor Slips (1970s)")
    print("=" * 60)
    
    # 1. Initialize
    print("\n[1] Initialize Linear Solver...")
    linear = NKHANKSolver(
        path_R="gpu_jacobian_R.csv",
        path_Z="gpu_jacobian_Z.csv",
        T=50
    )
    
    # 2. Define Shock
    shock = {"markup": 0.05}
    
    # 3. Run Counterfactuals
    res_anchored = run_scenario("Anchored (Rational)", gain=0.0, linear_solver=linear, shock=shock)
    res_unanchored = run_scenario("Unanchored (Learning)", gain=0.05, linear_solver=linear, shock=shock)
    
    # 4. Visualization
    print("\n[4] Plotting Results...")
    T = linear.T
    t = np.arange(T)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("The Great Inflation Logic: When the Anchor Slips", fontsize=14, fontweight='bold')
    
    # Panel A: Inflation
    axes[0].plot(t, res_anchored['pi'], label="Anchored", color='blue', linewidth=2)
    axes[0].plot(t, res_unanchored['pi'], label="Unanchored", color='red', linestyle='--', linewidth=2)
    axes[0].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[0].set_title("Inflation (π)")
    axes[0].set_xlabel("Time")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Panel B: Central Bank Belief
    belief_key = [k for k in res_unanchored.keys() if 'belief' in k.lower()]
    if belief_key:
        bk = belief_key[0]
        axes[1].plot(t, res_unanchored[bk], color='purple', linewidth=2)
        axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5, label="True r* = 0")
        axes[1].set_title(f"CB Misperception ({bk})")
        axes[1].set_xlabel("Time")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No belief variable found", ha='center', va='center')
    
    # Panel C: Policy Rate
    axes[2].plot(t, res_anchored['i'], label="Anchored", color='blue', linewidth=2)
    axes[2].plot(t, res_unanchored['i'], label="Unanchored", color='red', linestyle='--', linewidth=2)
    axes[2].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_title("Policy Rate (i)")
    axes[2].set_xlabel("Time")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(os.path.dirname(__file__), "killer_demo_2_output.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n=> Figure saved: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    main()
