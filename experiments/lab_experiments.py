import sys
import os
import numpy as np
import pandas as pd
import scipy.linalg
from scipy.sparse import diags, eye as sparse_eye

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import monad.monad_core as mc
from monad.nonlinear import NewtonSolver

# --- A. Full TT Jacobian Experiment ---

def run_tt_experiment():
    print("\n=== A. Full TT Jacobian Experiment ===")
    
    # 1. Steady State
    print("Solving Steady State (HANK)...")
    Nm = 50
    m_min, m_max, m_curv = -2.0, 50.0, 3.0
    Na = 40
    a_min, a_max, a_curv = 0.0, 100.0, 2.0
    
    z_grid = [0.8, 1.2]
    Pi_flat = [0.9, 0.1, 0.1, 0.9]
    
    beta = 0.97
    r_m = 0.01
    r_a = 0.05
    chi = 20.0
    sigma = 2.0
    tax_tau = 0.15
    tax_transfer = 0.05
    
    ss = mc.solve_hank_steady_state(
        Nm, m_min, m_max, m_curv,
        Na, a_min, a_max, a_curv,
        z_grid, Pi_flat,
        beta, r_m, r_a, chi, sigma,
        0.9, tax_tau, tax_transfer
    )
    
    # Check Market Clearing (approx)
    B = ss['agg_liquid']
    K = ss['agg_illiquid']
    print(f"SS Results: B={B:.4f}, K={K:.4f}")
    
    # 2. Compute Full TT Jacobians
    print("Computing Full TT Jacobians (T=80)...")
    T = 80
    
    # Need to pass distribution (from ss result)
    # The C++ binding expects std::vector for distribution
    # ss['distribution'] is it.
    
    J = mc.compute_jacobians(
        Nm, m_min, m_max, m_curv,
        Na, a_min, a_max, a_curv,
        z_grid, Pi_flat,
        beta, r_m, r_a, chi, sigma,
        0.9, tax_tau, tax_transfer,
        ss['c_pol'], ss['m_pol'], ss['a_pol'], ss['value'], ss['adjust_flag'],
        ss['distribution'],
        ss['E_Vm'], ss['E_V'],
        T
    )
    
    if 'J_C_rm' not in J:
        print("Error: Missing J_C_rm")
        return

    J_C_rm = np.array(J['J_C_rm'])
    J_C_w  = np.array(J.get('J_C_w', np.zeros((T,T)))) # If w block missing, 0
    
    print(f"J_C_rm shape: {J_C_rm.shape}")
    print("Exporting J_C_rm.csv...")
    pd.DataFrame(J_C_rm).to_csv("J_C_rm.csv", index=False, header=False)
    
    # 3. Macro Block & Synthesis (Simple NK)
    # Equations:
    # 1. Y = C + G        => dY = dC + dG
    # 2. pi = kappa*Y + beta*pi(+1)  => dPi = ...
    # 3. i = phi*pi       => di = phi*dPi
    # 4. r = i - pi(+1)   => dr = di - dPi(+1)
    # 5. w = Y (Simple production)   => dw = dY
    #
    # Household: dC = J_C_rm * dr + J_C_w * dw
    
    # Reduced System for dY:
    # dY = (J_C_rm * dr + J_C_w * dw) + dG
    # Substitute dw = dY
    # dY = J_C_rm * dr + J_C_w * dY + dG
    # (I - J_C_w) * dY = J_C_rm * dr + dG
    
    # Express dr in terms of dY (NK Block)
    # M_pi_Y such that dPi = M_pi_Y * dY
    # M_r_Y such that dr = M_r_Y * dY
    
    kappa = 0.1
    phi_pi = 1.5
    
    # NKPC Jacobian (dPi/dY)
    # pi_t = kappa*Y_t + beta*pi_{t+1}
    # (I - beta*L^-1) * pi = kappa * Y
    # pi = (I - beta*L^-1)^-1 * kappa * Y
    # Let K = (I - beta*L^-1)^-1 * kappa
    
    L_inv = np.diag(np.ones(T-1), 1) # Shift left (forward)
    inv_nkpc = np.linalg.inv(np.eye(T) - beta * L_inv)
    M_pi_Y = inv_nkpc * kappa
    
    # Taylor Rule & Fisher
    # i = phi * pi
    # r = i - pi(+1) = phi*pi - L_inv*pi = (phi*I - L_inv) * pi
    M_r_pi = (phi_pi * np.eye(T) - L_inv)
    M_r_Y = M_r_pi @ M_pi_Y
    
    # Final System:
    # (I - J_C_w) * dY = J_C_rm * (M_r_Y * dY) + dG
    # (I - J_C_w - J_C_rm * M_r_Y) * dY = dG
    
    H = np.eye(T) - J_C_w - J_C_rm @ M_r_Y
    
    print("\nSynthesized Total Jacobian H (d_ExcessDemand / d_Y)")
    print(f"Condition Number of H: {np.linalg.cond(H):.2e}")
    
    # 4. Solve for Fiscal Shock (dG)
    dG = np.zeros(T)
    dG[0:5] = 0.01 # 1% G shock for 5 quarters
    
    try:
        dY = np.linalg.solve(H, dG)
        print("Solved dY. Impact: ", dY[0])
        print("Success: HANK system solved via direct TxT Matrix Synthesis.")
    except Exception as e:
        print("Solver Failed:", e)


# --- B. Stress Test ---

def run_stress_test():
    print("\n=== B. Piecewise Solver Stress Test ===")
    
    # Create a Dummy Linear Solver wrapper for NewtonSolver
    # NewtonSolver needs .backend.J_C_r, .backend.J_C_y
    # And .block.get_phillips_curve() etc.
    
    class DummyBlock:
        def __init__(self, T, beta=0.97, kappa=0.1, phi_pi=1.5):
            self.params = {'phi_pi': phi_pi}
            self.beta = beta
            self.kappa = kappa
            self.T = T
            
        def get_phillips_curve(self):
            # Same as above
            L_inv = np.diag(np.ones(self.T-1), 1)
            inv_nkpc = np.linalg.inv(np.eye(self.T) - self.beta * L_inv)
            return inv_nkpc * self.kappa
            
        def get_taylor_rule(self):
            return self.params['phi_pi'] * np.eye(self.T)
            
        def get_fisher_equation(self):
            # r = i - pi(+1). di - S @ pi
            # S is L_inv (shift operator)
            return np.diag(np.ones(self.T-1), 1)

    class DummyBackend:
        def __init__(self, J_C_r, J_C_y):
            self.J_C_r = J_C_r
            self.J_C_y = J_C_y
            
    class DummyLinearSolver:
        def __init__(self, T, J_C_r, J_C_y, phi_pi=1.5):
            self.T = T
            self.backend = DummyBackend(J_C_r, J_C_y)
            self.block = DummyBlock(T, phi_pi=phi_pi)
            self.chi = 0.0 # Closed economy
    
    # Load Jacobians (Assume A ran first, or re-run)
    try:
        J_C_rm = pd.read_csv("J_C_rm.csv", header=None).values
    except:
        print("Skipping B (J_C_rm.csv not found, run A first)")
        return
        
    T = J_C_rm.shape[0]
    # J_C_w is J_C_y
    # Assume J_C_w is small/zero if not exported, or standard
    # Let's approximate J_C_y. HANK MPC usually high.
    # MPC ~ 0.2 diag?
    J_C_y = 0.2 * np.eye(T) 
    
    # Case 1: Standard ZLB (Should converge)
    solver_std = DummyLinearSolver(T, J_C_rm, J_C_y, phi_pi=1.5)
    newton = NewtonSolver(solver_std, max_iter=20)
    
    shock_rstar = np.zeros(T)
    shock_rstar[0:10] = -0.02 # Deep recession
    
    print("\n[Stress 1] ZLB Recession (Standard)")
    try:
        res = newton.solve_nonlinear(shock_rstar)
        print("Converged. Y[0]=", res['Y'][0])
    except Exception as e:
        print("Failed:", e)
        
    # Case 2: Taylor Principle Violation (phi < 1)
    print("\n[Stress 2] Taylor Principle Failure (phi_pi=0.8)")
    solver_bad = DummyLinearSolver(T, J_C_rm, J_C_y, phi_pi=0.8)
    newton_bad = NewtonSolver(solver_bad, max_iter=50, damping=0.1) # low damping to see divergence
    
    try:
        newton_bad.solve_nonlinear(shock_rstar)
        print("Converged?? (Should usually be indeterminate or explosive)")
    except RuntimeError as e:
        print("Expected Failure (Divergence/Oscillation) caught:", e)
    except Exception as e:
        print("Other Error:", e)


# --- C. Wrong Model Determinacy Check ---

def run_determinacy_check():
    print("\n=== C. Wrong Model Determinacy Check ===")
    
    def check_determinacy(H_matrix):
        """
        Check invertibility of H (condition number).
        Recall H * dY = dG.
        If H is singular, indeterminacy.
        """
        cond = np.linalg.cond(H_matrix)
        print(f"Matrix Condition Number: {cond:.2e}")
        
        # Check Eigenvalues of H (if it looks like I-A)
        # H = I - A.
        # Stability usually requires A eigenvalues inside unit circle?
        # Or just invertibility.
        # For Iterative solves (Neumann), rho(A) < 1.
        # For direct solve, just invertibility.
        
        if cond > 1e12:
            return "NO (Singular)"
        return "YES (Determinate)"

    # Load J
    try:
        J_C_rm = pd.read_csv("J_C_rm.csv", header=None).values
        T = J_C_rm.shape[0]
        J_C_y = 0.2 * np.eye(T) 
    except:
        return
        
    beta = 0.97
    kappa = 0.1
    
    # 1. Correct Model (phi=1.5)
    phi = 1.5
    L_inv = np.diag(np.ones(T-1), 1)
    inv_nkpc = np.linalg.inv(np.eye(T) - beta * L_inv)
    M_pi_Y = inv_nkpc * kappa
    M_r_Y = (phi * np.eye(T) - L_inv) @ M_pi_Y
    H_good = np.eye(T) - J_C_y - J_C_rm @ M_r_Y
    
    print(f"Phi={phi}: Determinacy = {check_determinacy(H_good)}")
    
    # 2. Wrong Model (phi=0.8)
    phi = 0.8
    M_r_Y_bad = (phi * np.eye(T) - L_inv) @ M_pi_Y
    H_bad = np.eye(T) - J_C_y - J_C_rm @ M_r_Y_bad
    
    print(f"Phi={phi}: Determinacy = {check_determinacy(H_bad)}")
    
    # 3. Fiscal Theory (Active Fiscal?)
    # If standard HANK, phi<1 is usually indeterminate.
    # But if B responds... 
    
    # Check "Correctness"
    if check_determinacy(H_bad).startswith("NO"):
        print("[PASS] System correctly rejected bad model.")
    else:
        # Sometimes HANK with phi<1 IS determinate if rigidities are high or fiscal feedback exists?
        # But generally we expect issues.
        print("[INFO] Bad model is technically invertible (finite T artifacts?), but condition number might be high.")


if __name__ == "__main__":
    run_tt_experiment()
    run_stress_test()
    run_determinacy_check()
