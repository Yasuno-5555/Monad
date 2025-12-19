import numpy as np
import pandas as pd
from scipy.linalg import toeplitz
import matplotlib.pyplot as plt
import os

class NKHANKSolver:
    def __init__(self, path_R, path_Z, T=50, params=None):
        """
        Full New Keynesian HANK Solver.
        Combines GPU Micro Jacobians with Analytical Macro Jacobians.
        """
        self.T = T
        if params is None:
            # Standard NK parameters
            self.params = {'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
        else:
            self.params = params

        # 1. Load GPU Micro Jacobians (Household Block)
        # J_C_r: Consumption response to Interest Rate
        # J_C_y: Consumption response to Income (MPC)
        if not os.path.exists(path_R) or not os.path.exists(path_Z):
             raise FileNotFoundError(f"Missing Jacobians: {path_R}, {path_Z}")

        df_R = pd.read_csv(path_R)
        df_Z = pd.read_csv(path_Z)
        
        # Create Toeplitz Matrices (TxT)
        # Input: r -> Output: C
        self.J_C_r = toeplitz(df_R['dC'].values[:T], np.zeros(T))
        # Input: Y -> Output: C (Note: Z shock implies Y shock)
        self.J_C_y = toeplitz(df_Z['dC'].values[:T], np.zeros(T))

    def build_macro_jacobians(self):
        """
        Construct Analytical Jacobians for NK Blocks.
        Returns matrices representing derivatives of the block outputs w.r.t inputs.
        """
        T = self.T
        beta = self.params['beta']
        kappa = self.params['kappa']
        phi = self.params['phi_pi']

        # Block 1: NKPC (Inflation pi depends on Y)
        # pi_t = kappa * Y_t + beta * pi_{t+1}
        # Matrix M_pi_y: d(pi)/d(Y)
        # Backwards substitution: pi_t = kappa * sum(beta^k * Y_{t+k})
        # This is an Upper Triangular Matrix
        M_pi_y = np.zeros((T, T))
        for t in range(T):
            for s in range(t, T):
                M_pi_y[t, s] = kappa * (beta ** (s - t))
        
        # Block 2: Taylor Rule (Nominal rate i depends on pi)
        # i_t = phi * pi_t
        M_i_pi = np.eye(T) * phi
        
        # Block 3: Fisher Equation (Real rate r depends on i and pi)
        # r_t = i_t - pi_{t+1}
        # dr/di = I
        # dr/dpi: depends on t+1. 
        # Shift matrix S: S[t, t+1] = 1
        S = np.zeros((T, T))
        for t in range(T-1):
            S[t, t+1] = 1.0
            
        # Total derivative of r w.r.t pi: dr = di - d(pi_next) = phi*d(pi) - S*d(pi)
        # But we need chain rule from Y.
        
        return M_pi_y, M_i_pi, S

    def solve_monetary_shock(self, shock_path):
        """
        Solves for General Equilibrium response to a Monetary Policy Shock (e.g., epsilon_i).
        Market Clearing: Y = C (Goods Market)
        
        System of Equations:
        1. dC = J_C_y * dY + J_C_r * dr
        2. dY = dC (Equilibrium)
        3. dr depends on dY via NKPC & Taylor Rule
        
        Substitute (3) into (1) & (2) to solve for dY.
        """
        M_pi_y, M_i_pi, S = self.build_macro_jacobians()
        
        # --- Chain Rule: How does Y affect r? ---
        # Y -> pi (via NKPC) -> i (via Taylor) -> r (via Fisher)
        
        # 1. d(pi) = M_pi_y * dY
        # 2. d(i)_endogenous = M_i_pi * d(pi) = M_i_pi * M_pi_y * dY
        # 3. d(r)_endogenous = d(i) - d(pi_next) 
        #                    = (M_i_pi - S) * d(pi)
        #                    = (M_i_pi - S) * M_pi_y * dY
        
        J_r_y = (M_i_pi - S) @ M_pi_y  # The "Aggregate Demand" Logic Matrix
        
        # --- Total System ---
        # dY = dC
        # dY = J_C_y * dY + J_C_r * (dr_endogenous + dr_shock)
        # dY = J_C_y * dY + J_C_r * (J_r_y * dY + shock_path)
        
        # Collect dY terms:
        # (I - J_C_y - J_C_r * J_r_y) * dY = J_C_r * shock_path
        
        I = np.eye(self.T)
        A = I - self.J_C_y - self.J_C_r @ J_r_y
        b = self.J_C_r @ shock_path
        
        # Solve linear system
        dY = np.linalg.solve(A, b)
        
        # Back out other variables
        dC = dY # Market clearing
        dpi = M_pi_y @ dY
        di_endo = M_i_pi @ dpi
        dr = J_r_y @ dY + shock_path
        
        return dY, dpi, dr

    def visualize_decomposition(self, dr_path, dY_path):
        """
        Decomposes the consumption response into Direct (Rate) and Indirect (Income) effects.
        """
        # 1. Compute Components
        # Direct Effect: How much C changes due to r moving (holding Y constant)
        # J_C_r is (TxT). dr_path is (T). Result is (T).
        dC_direct = self.J_C_r @ dr_path
        
        # Indirect Effect: How much C changes due to Y moving (holding r constant)
        dC_indirect = self.J_C_y @ dY_path
        
        # Total (Verification)
        dC_total = dC_direct + dC_indirect
        
        # 2. Plot
        t = np.arange(len(dr_path))
        plt.figure(figsize=(10, 6))
        
        # Stacked Area / Bar Chart
        # Use simple plotting for clarity. 
        # Note: If effects have different signs, stacking can be visually confusing in pure 'stackplot' or 'bar' without care.
        # But for standard contractionary shock:
        # dr > 0 -> dC_direct < 0 (Substitution)
        # dY < 0 -> dC_indirect < 0 (Income Loss)
        # Both are negative. So stacking works well downwards.
        
        plt.bar(t, dC_direct, label='Direct Effect (Intertemporal Subst.)', color='#1f77b4', alpha=0.7, width=1.0)
        plt.bar(t, dC_indirect, bottom=dC_direct, label='Indirect Effect (Income Loss)', color='#d62728', alpha=0.7, width=1.0)
        plt.plot(t, dC_total, label='Total Consumption Change', color='black', linewidth=2, linestyle='--')
        
        plt.title('Decomposition of Consumption (Monetary Contraction +25bps)')
        plt.xlabel('Quarters')
        plt.ylabel('% Deviation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("decomposition.png")
        print("Decomposition plot saved to decomposition.png")
        # plt.show() # Non-interactive

# --- Usage Example for Verification ---
if __name__ == "__main__":
    # Ensure CSVs exist from Step 1
    # Assuming valid paths to CSVs in CWD
    try:
        solver = NKHANKSolver("gpu_jacobian_R.csv", "gpu_jacobian_Z.csv", T=50)
        
        # persistent monetary shock
        rho = 0.8
        shock_r = 0.0025 * (rho ** np.arange(50))

        dY, dpi, dr = solver.solve_monetary_shock(shock_r)
        
        print("--- NK-HANK Solver Results ---")
        print(f"dY impact: {dY[0]:.6f}")
        print(f"dpi impact: {dpi[0]:.6f}")
        print(f"dr impact: {dr[0]:.6f}")
        
        # Run Decomposition Analysis
        # Note: For decomposition, we need the ENDOGENOUS real rate response 'dr', which includes the shock.
        # solver.solve_monetary_shock returns 'dr' which is (J_r_y @ dY + shock_path).
        # This is exactly what the household faces.
        solver.visualize_decomposition(dr, dY)
        
    except Exception as e:
        print(f"Error: {e}")
