import numpy as np
from .core import GPUBackend
from .blocks import NKBlock

class NKHANKSolver:
    """
    General Equilibrium Solver for the Monad Engine.
    Assembles GPU micro-Jacobians and analytical macro-blocks.
    """
    def __init__(self, path_R, path_Z, T=50, params=None):
        # 1. Initialize Infrastructure
        self.backend = GPUBackend(path_R, path_Z, T)
        self.T = T
        
        # 2. Initialize Theory
        # Default params if none provided
        if params is None:
            params = {'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
        self.block = NKBlock(T, params)

    def solve_monetary_shock(self, shock_path):
        """
        Solves GE for a monetary policy shock (e.g., r_exogenous or i_shock).
        System: dY = J_C_y @ dY + J_C_r @ (dr_endo + shock)
        """
        # A. Fetch Analytical Matrices
        M_pi_y = self.block.get_phillips_curve()
        M_i_pi = self.block.get_taylor_rule()
        S      = self.block.get_fisher_equation()
        
        # B. Construct Aggregate Demand Logic (Chain Rule)
        # Y -> pi -> i -> r
        # dr_endo = (M_i_pi - S) @ M_pi_y @ dY
        J_r_y = (M_i_pi - S) @ M_pi_y
        
        # C. Assemble Linear System (A @ dY = b)
        # dY = J_C_y @ dY + J_C_r @ (J_r_y @ dY + shock)
        # (I - J_C_y - J_C_r @ J_r_y) @ dY = J_C_r @ shock
        
        I = np.eye(self.T)
        A = I - self.backend.J_C_y - self.backend.J_C_r @ J_r_y
        b = self.backend.J_C_r @ shock_path
        
        # D. Solve
        dY = np.linalg.solve(A, b)
        
        # E. Back out other variables
        dpi = M_pi_y @ dY
        dr  = J_r_y @ dY + shock_path
        dC  = dY # Goods market clearing
        
        return {
            'dY': dY,
            'dpi': dpi,
            'dr': dr,
            'dC': dC,
            'shock': shock_path
        }
