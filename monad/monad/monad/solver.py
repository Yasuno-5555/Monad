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

class SOESolver(NKHANKSolver):
    """
    Small Open Economy HANK Solver.
    Extends the closed economy solver with Exchange Rate and Trade Balance channels.
    """
    def __init__(self, path_R, path_Z, T=50, params=None):
        super().__init__(path_R, path_Z, T, params)
        # alpha: Import share (0.4 means 40% of basket is foreign)
        # chi:   Trade elasticity (how much NX responds to Q)
        self.alpha = params.get('alpha', 0.4) if params else 0.4
        self.chi   = params.get('chi', 1.0)   if params else 1.0

    def solve_open_economy_shock(self, dr_star_path):
        """
        Solves GE for a Foreign Interest Rate Shock (r*).
        
        Mechanism:
        1. r* rises -> Domestic currency depreciates (Q rises) via UIP.
        2. Q rises -> Net Exports (NX) increase (Competitiveness).
        3. Q rises -> Import prices rise -> Real Income (Z) falls.
        
        System of Equations:
        (1) dQ = -UIP @ (dr - dr*)  [Assume dr is fixed for partial eq, or solve endo]
            *For this specific experiment, let's assume Domestic Central Bank holds r fixed 
             to isolate the exchange rate channel (Peg or Inertia).*
             So dr = 0, and Q moves purely due to r*.
        
        (2) dZ = dY - alpha * dQ  [Purchasing Power Effect]
        (3) dC = J_C_y @ dZ + J_C_r @ dr
        (4) dNX = chi * dQ        [Marshall-Lerner Condition]
        (5) dY = dC + dNX         [Goods Market Clearing]
        
        Combine:
        dY = J_C_y @ (dY - alpha * dQ) + J_C_r @ 0 + chi * dQ
        (I - J_C_y) @ dY = (chi * I - alpha * J_C_y) @ dQ
        """
        # A. Calculate Exchange Rate Response (Q) from Foreign Rate (r*)
        # UIP: Q_t = Sum_{k=t}^T (r*_k - r_k)
        # Since r (domestic) is 0 for this test, Q accumulates r*.
        # We need the summation matrix (Upper Triangular of 1s)
        U_sum = np.triu(np.ones((self.T, self.T)))
        dQ = U_sum @ dr_star_path 

        # B. Construct Linear System for Output (Y)
        # dY = J_C_y @ dY - J_C_y @ (alpha * dQ) + chi * dQ
        # (I - J_C_y) @ dY = (chi * I - self.alpha * J_C_y) @ dQ
        
        I = np.eye(self.T)
        A = I - self.backend.J_C_y
        
        # RHS: The net force of "Export Boom" vs "Import Inflation"
        # term1: +chi * dQ (Exports increase Y)
        # term2: -alpha * J_C_y @ dQ (Import costs decrease C, which decreases Y)
        RHS = (self.chi * I - self.alpha * self.backend.J_C_y) @ dQ
        
        # C. Solve for Y
        dY = np.linalg.solve(A, RHS)
        
        # D. Back out Consumption
        # dZ = dY - alpha * dQ
        dZ = dY - self.alpha * dQ
        dC = self.backend.J_C_y @ dZ # + J_C_r @ 0
        
        # E. Net Exports
        dNX = self.chi * dQ
        
        return {
            'dY': dY,
            'dC': dC,
            'dQ': dQ,
            'dNX': dNX,
            'dZ': dZ, # Real Labor Income
            'shock': dr_star_path
        }

