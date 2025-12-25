"""
Monad GE Solver - Linearized General Equilibrium Solver using SSJ

This module implements the Sequence Space Jacobian (SSJ) method for solving
general equilibrium in the Two-Asset HANK model. It uses GPU-computed 
Jacobians exported from C++ via CSV.

Based on Auclert et al. (2021) methodology.
"""

import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
from typing import Tuple, Optional


class LinearizedGESolver:
    """
    Solves linearized general equilibrium using Toeplitz Jacobian matrices.
    
    The core idea from SSJ: A single IRF vector encodes the entire TxT
    Jacobian matrix due to time-invariance (Toeplitz structure).
    
    Market Clearing: B_demand(r_m) = B_supply
    Linear Approx:   J_B_rm @ dr_m = dB_target
    Solution:        dr_m = solve(J_B_rm, dB_target)
    """
    
    def __init__(self, csv_path: str, T: int = 50):
        """
        Initialize the GE solver from GPU-exported Jacobians.
        
        Args:
            csv_path: Path to 'gpu_jacobian.csv' with columns [t, dC, dB]
            T: Time horizon (truncation point for infinite-horizon problem)
        """
        self.T = T
        
        # Load IRF data from C++ export
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Could not find {csv_path}. "
                "Run C++ Phase3Test.exe first to generate Jacobian data."
            )
        
        # Extract IRF vectors (truncate to T if longer)
        self.dB_irf = df['dB'].values[:T]
        self.dC_irf = df['dC'].values[:T]
        
        # Construct Sequence Space Jacobian (Toeplitz matrices)
        # J[t, s] = response at time 't' to shock at time 's'
        # Lower-triangular due to causality (no backward effects)
        self.J_B_rm = toeplitz(self.dB_irf, np.zeros(T))
        self.J_C_rm = toeplitz(self.dC_irf, np.zeros(T))
        
        print(f"[Monad GE] Loaded Jacobians from {csv_path}")
        print(f"           T = {T}, J_B_rm[0,0] = {self.J_B_rm[0,0]:.6f}")
        print(f"           dB mean-reversion: {abs(self.dB_irf[-1]) < abs(self.dB_irf[0])}")
    
    def solve_for_rm(self, dB_target: np.ndarray) -> np.ndarray:
        """
        Solve for equilibrium interest rate path.
        
        Market Clearing Condition:
            dB_demand(dr_m) + dB_supply_shock = 0
            J_B_rm @ dr_m = dB_target
        
        Args:
            dB_target: Exogenous shock to bond supply (length T).
                       Positive = government issues more debt.
                       
        Returns:
            dr_m: Equilibrium interest rate deviations (length T).
        """
        if len(dB_target) != self.T:
            raise ValueError(f"Length mismatch: {len(dB_target)} vs T={self.T}")
        
        dr_m = np.linalg.solve(self.J_B_rm, dB_target)
        return dr_m
    
    def compute_consumption_response(self, dr_m: np.ndarray) -> np.ndarray:
        """
        Compute aggregate consumption response given equilibrium dr_m.
        
        dC = J_C_rm @ dr_m
        """
        return self.J_C_rm @ dr_m
    
    def solve_full_ge(self, dB_shock: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Full GE solution: shock -> dr_m -> dC
        
        Returns:
            Tuple of (dr_m, dC)
        """
        dr_m = self.solve_for_rm(dB_shock)
        dC = self.compute_consumption_response(dr_m)
        return dr_m, dC
    
    def diagnostic_report(self):
        """Print diagnostic information about the Jacobians."""
        print("\n=== Jacobian Diagnostics ===")
        print(f"J_B_rm condition number: {np.linalg.cond(self.J_B_rm):.2e}")
        print(f"J_B_rm diagonal (impact effects):")
        print(f"  dB/dr[0]: {self.J_B_rm[0,0]:.6f}")
        print(f"  dB/dr[10]: {self.J_B_rm[10,10]:.6f}")
        print(f"  dB/dr[T-1]: {self.J_B_rm[-1,-1]:.6f}")
        
        # Check mean-reversion
        half_life = -1
        for t in range(self.T):
            if abs(self.dB_irf[t]) < 0.5 * abs(self.dB_irf[0]):
                half_life = t
                break
        print(f"  Half-life of dB: {half_life} periods")


def visualize_ge_solution(dr_m: np.ndarray, dC: np.ndarray, 
                          title: str = "GE Solution"):
    """Quick visualization of GE transition path."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(dr_m, 'r-', linewidth=2)
        axes[0].axhline(0, color='gray', linestyle=':', linewidth=0.8)
        axes[0].set_title(f"{title}: Interest Rate Path (dr_m)")
        axes[0].set_xlabel("Time")
        axes[0].set_ylabel("dr_m")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(dC, 'b-', linewidth=2)
        axes[1].axhline(0, color='gray', linestyle=':', linewidth=0.8)
        axes[1].set_title(f"{title}: Consumption Response (dC)")
        axes[1].set_xlabel("Time")
        axes[1].set_ylabel("dC")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("ge_solution.png", dpi=150)
        plt.show()
        print("[Monad GE] Plot saved to ge_solution.png")
        
    except ImportError:
        print("[Warning] matplotlib not available for visualization")


# --- Self-Test ---
if __name__ == "__main__":
    import os
    
    print("=== Monad GE Solver Test ===\n")
    
    csv_path = "gpu_jacobian.csv"
    
    # Check if real data exists
    if not os.path.exists(csv_path):
        print(f"[INFO] {csv_path} not found. Creating dummy data for testing...")
        t = np.arange(50)
        # Dummy physics: r up -> B up (positive response)
        dB_dummy = 1.0 * (0.85 ** t)  # Exponential decay
        dC_dummy = -0.5 * (0.85 ** t)  # Consumption falls when r rises
        df = pd.DataFrame({'t': t, 'dC': dC_dummy, 'dB': dB_dummy})
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Created dummy {csv_path}")
    
    # Initialize solver
    solver = LinearizedGESolver(csv_path, T=50)
    solver.diagnostic_report()
    
    # Test: Permanent 1% debt increase
    print("\n=== Test: 1% Permanent Debt Shock ===")
    shock = np.ones(50) * 0.01
    
    dr_m, dC = solver.solve_full_ge(shock)
    
    print(f"Initial reaction:")
    print(f"  dr_m[0]:  {dr_m[0]:.6f}")
    print(f"  dC[0]:    {dC[0]:.6f}")
    print(f"\nLong-run (t=49):")
    print(f"  dr_m[49]: {dr_m[49]:.6f}")
    print(f"  dC[49]:   {dC[49]:.6f}")
    print(f"\nMean-reversion check: |dr_m[49]| < |dr_m[0]|: {abs(dr_m[49]) < abs(dr_m[0])}")
    
    # Visualize if matplotlib available
    visualize_ge_solution(dr_m, dC, "Debt Shock (+1%)")
