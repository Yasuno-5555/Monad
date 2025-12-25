import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from monad.solver import NKHANKSolver
from monad.nonlinear import NewtonSolver

def run_liquidity_trap():
    print("--- Monad v6.0: The Liquidity Trap Experiment ---")

    # 1. Setup Linear Solver (Base Model)
    # Using the standard "closed economy" solver or SOE doesn't matter much for this 
    # specific ZLB test, but let's use NKHANKSolver (Closed) to isolate ZLB dynamics 
    # without FX interference.
    
    # Check for CSVs one level up
    path_R = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_R.csv")
    path_Z = os.path.join(os.path.dirname(__file__), "../gpu_jacobian_Z.csv")
    
    # Standard Calibration
    params = {'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
    linear_solver = NKHANKSolver(path_R=path_R, path_Z=path_Z, T=50, params=params)

    # 2. Setup Nonlinear Solver
    newton = NewtonSolver(linear_solver)

    # 3. Define Shock: "Great Recession"
    # Natural Rate (r*) drops by 200bps (2%) for 10 quarters
    # Steady state r* is approx 0.5% (annual 2%). 
    # So r* becomes -1.5%.
    # Target Rate without ZLB would be -1.5% + phi*pi.
    
    T = 50
    shock_r_star = np.zeros(T)
    shock_r_star[0:10] = -0.01 # -1% shock for 10 periods
    
    # 4. Linear Solution (Impossible World)
    print("Solving Linear Model (No ZLB)...")
    # In linear solver, 'shock' is usually dr_endo + shock. 
    # Here solve_monetary_shock handles 'shock' as the exogenous r component.
    # Wait, solve_monetary_shock in NKHANKSolver solves:
    # dY = J_C_y dY + J_C_r (dr + shock)
    # where dr is endogenous response.
    # Here the shock is to r* (natural rate) in the Taylor Rule.
    # Taylor Rule: i = r* + phi*pi.
    # dr = i - pi_next - r*. (Fisher equation defines r)
    # Actually, usually r* shock enters Taylor rule intercept.
    # Let's verify NKHANKSolver.solve_monetary_shock logic.
    # It solves: dY = ... + J_C_r @ shock.
    # If the shock is to the Interest Rate Rule intercept, it is effectively a monetary shock.
    # So valid to use solve_monetary_shock(shock_r_star).
    res_linear = linear_solver.solve_monetary_shock(shock_r_star)


    # 5. Nonlinear Solution (Real World)
    print("Solving Nonlinear Model (With ZLB)...")
    try:
        res_nonlinear = newton.solve_nonlinear(shock_path=shock_r_star)
    except Exception as e:
        print(f"Nonlinear solver failed: {e}")
        return

    # 6. Visualize: The Trap
    t = np.arange(T)
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"The Liquidity Trap: Linear vs Nonlinear ZLB", fontsize=16)

    # Nominal Interest Rate
    # Linear: goes negative
    # Nonlinear: sticks at 0
    # Note: res_nonlinear['i'] is deviation. Level = 0.005 + deviation.
    r_ss = 0.005
    i_linear_level = r_ss + res_linear['dr'] + res_linear['dpi'] # Approx i = r + pi
    # Actually solve_monetary_shock returns 'dr'.
    # i = r + pi.
    # Let's compute 'i' for linear.
    # But wait, solve_monetary_shock applies shock to 'b' in A dY = b.
    # It assumes shock is added to r.
    # Let's just use what we have.
    # Ideally: nonlinear solver output has strict 'i'.
    
    i_nonlinear_level = r_ss + res_nonlinear['i']
    
    # For Linear, let's look at implied i
    # Taylor: i = r* + shock + phi * pi
    # i_linear_level = r_ss + shock_r_star + 1.5 * res_linear['dpi']
    i_linear_level = r_ss + shock_r_star + 1.5 * res_linear['dpi']
    
    ax[0,0].plot(i_linear_level*400, label='Linear (Negative Rates)', linestyle='--', color='gray')
    ax[0,0].plot(i_nonlinear_level*400, label='Nonlinear (ZLB)', color='red', lw=2)
    ax[0,0].axhline(0, color='black', lw=1)
    ax[0,0].set_title("Nominal Interest Rate (Annual %)")
    ax[0,0].legend()
    ax[0,0].grid(True, alpha=0.3)

    # Output Gap
    ax[0,1].plot(res_linear['dY']*100, label='Linear', linestyle='--', color='gray')
    ax[0,1].plot(res_nonlinear['Y']*100, label='Nonlinear (Deep Recession)', color='blue', lw=2)
    ax[0,1].set_title("Output Gap (% Deviation)")
    ax[0,1].legend()
    ax[0,1].grid(True, alpha=0.3)
    ax[0,1].axhline(0, color='black', lw=1)
    
    # Inflation
    ax[1,0].plot(res_linear['dpi']*400, label='Linear', linestyle='--', color='gray')
    ax[1,0].plot(res_nonlinear['pi']*400, label='Nonlinear (Deflation)', color='green', lw=2)
    ax[1,0].set_title("Inflation (Annual % Deviation)")
    ax[1,0].grid(True, alpha=0.3)
    ax[1,0].axhline(0, color='black', lw=1)

    # Real Interest Rate
    # In ZLB: i fixed at 0, pi drops -> r = i - pi = 0 - (-large) = +large
    # Real rates RISE during recession -> kills demand further.
    r_linear = r_ss + res_linear['dr']
    r_nonlinear = r_ss + res_nonlinear['r']
    
    ax[1,1].plot(r_linear*400, label='Linear', linestyle='--', color='gray')
    ax[1,1].plot(r_nonlinear*400, label='Nonlinear (Too High)', color='orange', lw=2)
    ax[1,1].set_title("Real Interest Rate (Annual %)")
    ax[1,1].grid(True, alpha=0.3)
    ax[1,1].legend()

    plt.tight_layout()
    plt.savefig("liquidity_trap.png")
    print("Plot saved to liquidity_trap.png")
    # plt.show()

if __name__ == "__main__":
    run_liquidity_trap()
