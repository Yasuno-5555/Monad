import numpy as np
import scipy.linalg

class NewtonSolver:
    """
    Newton-Raphson Solver for General Equilibrium on Sequence Space.
    Solves F(X) = 0 where F is a system of nonlinear equations.
    """
    def __init__(self, linear_solver, max_iter=50, tol=1e-8):
        """
        Args:
            linear_solver: An instance of the linear solver (e.g., SOESolver/NKHANKSolver).
                           We use its Jacobian (J) to guide the Newton step.
            max_iter: Maximum iterations.
            tol: Convergence tolerance for residuals.
        """
        self.linear_solver = linear_solver
        self.max_iter = max_iter
        self.tol = tol
        self.T = linear_solver.T

    def solve_nonlinear(self, shock_path, initial_guess_path=None):
        """
        Finds the path of endogenous variables (e.g., Y) that satisfies nonlinear equilibrium.
        
        Equation to solve:
        Residual(Y) = Aggregate Demand(Y) - Aggregate Supply(Y) = 0
        
        But with ZLB, the Interest Rate 'i' is the tricky part.
        We iterate on 'Y' (Output) as the main unknown.
        """
        # 1. Initialize Guess (Y_path)
        # If no guess, start from Steady State (deviations = 0)
        # Or use the Linear Solution as a smart starting point.
        if initial_guess_path is None:
            Y_guess = np.zeros(self.T)
        else:
            Y_guess = initial_guess_path.copy()

        print(f"--- Nonlinear Solver Started (Max Iter: {self.max_iter}) ---")
        
        for it in range(self.max_iter):
            # A. Evaluate the Nonlinear Economy given current guess Y
            # This computes: pi(Y), i(pi, Y) with ZLB check, r(i, pi), and C(r, Y)
            block_results = self._evaluate_blocks(Y_guess, shock_path)
            
            # B. Compute Residuals (Excess Demand)
            # Market Clearing: Y = C + NX + G
            # Res = (C + NX) - Y
            if 'NX' in block_results:
                aggregate_demand = block_results['C_agg'] + block_results['NX']
            else:
                aggregate_demand = block_results['C_agg']
                
            residual = aggregate_demand - Y_guess
            
            # C. Check Convergence
            err = np.max(np.abs(residual))
            if it % 5 == 0:
                print(f"  Iter {it}: Max Residual = {err:.2e}")
            
            if err < self.tol:
                print(f"  [CONVERGED] Iter {it}, Error: {err:.2e}")
                return block_results

            # D. Newton Step: dY = - J_inv * Residual
            # We use the Linear General Equilibrium Jacobian (H) from the linear solver.
            # H = d(Y - C)/dY = I - dC/dY_total
            # This is exactly the 'A' matrix in the linear solver!
            
            # Reconstruct the 'A' matrix (I - J_C_y - J_C_r @ J_r_y)
            # Note: J_r_y changes in ZLB! (If i=0, di/dY = 0).
            # Ideally we update J at every step. For now, let's use the 'active set' approximation.
            
            # Determine which periods are at ZLB
            is_constrained = block_results['is_at_zlb'] # Boolean vector
            
            # Build the Jacobian dynamically based on constraint
            J_step = self._build_dynamic_jacobian(is_constrained)
            
            # Newton Update
            # J * dY = residual  => dY = J_inv * (C - Y)
            # (Derivation: F(Y) = Y - C.  dY = -J_inv * F = J_inv * C - J_inv * Y = J_inv * (C-Y))
            dY_step = np.linalg.solve(J_step, residual)
            
            # Damping (to prevent explosion)
            damping = 0.5
            Y_guess = Y_guess + damping * dY_step

        raise RuntimeError("Newton Solver failed to converge.")

    def _evaluate_blocks(self, Y_path, shock_r_star, foreign_r_star=None):
        """
        Forward pass: Given Y, calculate everything else using EXACT nonlinear formulas.
        Supports both Closed and Open Economy (SOE) if linear_solver has SOE attributes.
        """
        # 1. NKPC: pi is linear in Y
        M_pi_y = self.linear_solver.block.get_phillips_curve()
        pi_path = M_pi_y @ Y_path
        
        # 2. Taylor Rule with ZLB
        # i_target = r_natural + phi * pi
        # Using shock_r_star as the Natural Rate deviation
        # Access steady state params 
        r_ss = 0.005 # e.g. 2% annualized
        phi = self.linear_solver.block.params['phi_pi']
        
        target_i_level = r_ss + shock_r_star + phi * pi_path
        actual_i_level = np.maximum(0.0, target_i_level)
        
        di_path = actual_i_level - r_ss
        is_at_zlb = (actual_i_level <= 1e-6)
        
        # 3. Real Rate (Fisher)
        S = self.linear_solver.block.get_fisher_equation()
        dr_path = di_path - S @ pi_path

        # 4. Household Consumption & Aggregates
        # Check for Open Economy attributes
        is_soe = hasattr(self.linear_solver, 'alpha')
        
        if is_soe:
            # Open Economy Logic
            alpha = self.linear_solver.alpha
            chi   = self.linear_solver.chi
            
            # Exchange Rate (UIP): Q_t = Sum(r*_t - r_t)
            # We assume foreign_r_star is passed, or 0 if not.
            if foreign_r_star is None:
                # If global recession, maybe r* also drops? 
                # For simplicity, let's assume foreign r* follows the same shock as domestic r* natural
                # effectively a global shock.
                r_foreign = shock_r_star 
            else:
                r_foreign = foreign_r_star
                
            # UIP Summation (Upper Triangular of 1s)
            diff_r = r_foreign - dr_path
            # Q = Sum(diff_r)
            U_sum = np.triu(np.ones((self.T, self.T)))
            dQ_path = U_sum @ diff_r
            
            # Real Income Z = Y - alpha * Q
            dZ_path = Y_path - alpha * dQ_path
            
            # Net Exports NX = chi * Q
            dNX_path = chi * dQ_path
            
            # Consumption C(r, Z)
            dC_path = self.linear_solver.backend.J_C_r @ dr_path + \
                      self.linear_solver.backend.J_C_y @ dZ_path
                      
            # Store SOE variables
            extras = {'Q': dQ_path, 'NX': dNX_path, 'Z': dZ_path}
            
        else:
            # Closed Economy Logic
            dC_path = self.linear_solver.backend.J_C_r @ dr_path + \
                      self.linear_solver.backend.J_C_y @ Y_path
            extras = {}

        results = {
            'Y': Y_path,
            'pi': pi_path,
            'i': di_path,
            'r': dr_path,
            'C_agg': dC_path,
            'is_at_zlb': is_at_zlb
        }
        results.update(extras)
        return results

    def _build_dynamic_jacobian(self, is_constrained):
        """
        Constructs the Jacobian matrix J = d(Residual)/dY
        Residual = C + NX - Y (SOE) or C - Y (Closed)
        We want J such that J * dY = -Residual.
        Actually Newton is: Y_new = Y_old - J_inv * Residual.
        So J should be d(Residual)/dY.
        
        Residual = C(r, Is) + NX(Q) - Y
        dRes/dY = dC/dY + dNX/dY - I
        
        Let's compute derivative of RHS (Agg Demand) w.r.t Y: J_AD.
        Then J_step = J_AD - I.
        And dY = - (J_AD - I)^-1 * Res = (I - J_AD)^-1 * Res.
        
        My code performs: dY = solve(J_step, Res).
        So J_step needs to be (I - J_AD).
        """
        # 1. Basic Derivatives
        M_pi_y = self.linear_solver.block.get_phillips_curve()
        M_i_pi = self.linear_solver.block.get_taylor_rule() 
        S      = self.linear_solver.block.get_fisher_equation()
        
        # ZLB Adjustment for Interest Rate Rule
        M_i_pi_constrained = M_i_pi.copy()
        for t in range(self.T):
            if is_constrained[t]:
                M_i_pi_constrained[t, :] = 0.0
        
        # Chain Rule: dr/dY
        J_r_y = (M_i_pi_constrained - S) @ M_pi_y
        
        # 2. Check SOE
        is_soe = hasattr(self.linear_solver, 'alpha')
        
        if is_soe:
            alpha = self.linear_solver.alpha
            chi   = self.linear_solver.chi
            
            # dQ/dr (UIP)
            # Q = U_sum @ (r* - r). So dQ/dr = -U_sum.
            U_sum = np.triu(np.ones((self.T, self.T)))
            M_q_r = -U_sum
            
            # dQ/dY = dQ/dr @ dr/dY
            J_q_y = M_q_r @ J_r_y
            
            # dZ/dY = I - alpha * J_q_y
            J_z_y = np.eye(self.T) - alpha * J_q_y
            
            # dNX/dY = chi * J_q_y
            J_nx_y = chi * J_q_y
            
            # dC/dY = J_C_r @ J_r_y + J_C_y @ J_z_y
            J_c_y = self.linear_solver.backend.J_C_r @ J_r_y + \
                    self.linear_solver.backend.J_C_y @ J_z_y
                    
            # J_AD (Total Demand Jacobian) = dC/dY + dNX/dY
            J_AD = J_c_y + J_nx_y
            
        else:
            # Closed Economy
            # dC/dY = J_C_r @ J_r_y + J_C_y
            J_c_y = self.linear_solver.backend.J_C_r @ J_r_y + \
                    self.linear_solver.backend.J_C_y
            J_AD = J_c_y

        # Final Jacobian for Newton: (I - J_AD)
        # Because we want to solve (I - J_AD) dY = ExcessDemand
        J = np.eye(self.T) - J_AD
            
        return J
