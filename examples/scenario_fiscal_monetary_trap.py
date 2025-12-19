"""
Scenario Analysis: The Fiscal-Monetary Trap
===========================================
Phase 1 (Regime) + Phase 3 (Welfare)

Scenario:
1. Cost-Push Shock (Supply Chain Crisis) hits the economy -> Inflation spikes.
2. Central Bank wants to raise rates (Taylor Rule).
3. BUT, raising rates increases Government Debt Service (r * B).
4. If Debt breaches a "Psychological Limit", the economy snaps into "Fiscal Dominance".
   - CB is forced to peg rates to save the Treasury.
   - Inflation spirals.

Analyse:
- Path A: Monetary Dominance (Assume Debt Limit is infinite). High Rates, Low Inflation.
- Path B: Fiscal Dominance (The Trap). Pegged Rates, High Inflation.

Welfare Question:
- High Rates hurt Borrowers (Mortgage holders).
- High Inflation hurts Savers (Pensioners).
- Who wins? Who loses?
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.welfare import WelfareAnalysis

class FiscalTrapModel:
    """
    Mock New Keynesian Model with Debt Dynamics and Regime Switching.
    """
    def __init__(self, T=50):
        self.T = T
        # Parameters
        self.sigma = 1.0 # IES
        self.kappa = 0.1 # Slope of PC
        self.beta = 0.99
        self.phi_pi = 1.5 # Taylor Rule
        
        # Fiscal
        self.B_init = 1.0 # Initial Debt
        self.Debt_Limit = 1.08 # Strict Limit (8% increase triggers crisis)
        
    def simulate(self, regime="monetary_dominance", shock_size=0.02):
        """
        Simulate the economy under a specific regime or endogenous switch.
        regime: 'monetary_dominance', 'fiscal_dominance', 'endogenous'
        """
        T = self.T
        
        # Paths
        pi = np.zeros(T)
        y = np.zeros(T)
        i = np.zeros(T)
        r_real = np.zeros(T)
        B = np.zeros(T)
        
        # Shock: Cost Push (u_t) decaying
        u = np.zeros(T)
        for t in range(T):
            u[t] = shock_size * (0.8 ** t)
            
        # Initial State
        B_prev = self.B_init
        
        # Forward Simulation (Simplified Loop)
        # Note: Proper NK model requires backward solution for Expectations.
        # Here we use a heuristic "Adaptive/Perfect" mix for the mock.
        # pi_t = kappa * y_t + beta * pi_{t+1|t} + u_t
        # To simplify: We solve 'static' equilibrium at each t assuming E[pi] tracks shock?
        # Let's use a simple ad-hoc law of motion for specific demo purposes.
        
        current_regime = "MD" # Start in Monetary Dominance
        
        for t in range(T):
            # 1. Determine Regime (If Endogenous)
            if regime == "endogenous":
                if B_prev > self.Debt_Limit:
                    current_regime = "FD"
                else:
                    current_regime = "MD"
            elif regime == "fiscal_dominance":
                current_regime = "FD"
            else:
                current_regime = "MD"
                
            # 2. Set Interest Rate
            # Inflation expectation proxy: current shock impact
            expected_pi = u[t] * 0.5 # Simple proxy
            
            if current_regime == "MD":
                # Taylor Rule: Fight inflation
                # i = r* + phi * pi
                # But pi depends on y, y depends on i.
                # Reduced form guess:
                # High u -> High pi.
                pi_val = u[t] * 1.5 # Inflation rises
                i_val = self.phi_pi * pi_val
            else:
                # Fiscal Dominance: Peg Rate
                i_val = 0.0
                pi_val = u[t] * 2.5 # Unanchored inflation!
                
            # 3. Real Rate
            r_real_val = i_val - pi_val
            
            # 4. Output (IS Curve)
            # y = -sigma * r_real
            y_val = -self.sigma * r_real_val
            
            # 5. Debt Dynamics
            # B_next = (1 + r_real) * B_prev (Simplified budget)
            # Higher real rates increase debt burden.
            B_curr = (1.0 + r_real_val) * B_prev
            
            # Store
            pi[t] = pi_val
            y[t] = y_val
            i[t] = i_val
            r_real[t] = r_real_val
            B[t] = B_curr
            
            B_prev = B_curr
            
        return {
            'pi': pi, 
            'y': y, 
            'i': i, 
            'r': r_real,
            'B': B,
            'u': u
        }

    def compute_consumption(self, res):
        """
        Derive heterogeneous consumption for Welfare Analysis.
        Two Agents: Borrower (b), Saver (s).
        """
        T = self.T
        y = res['y']
        r = res['r']
        pi = res['pi']
        
        # Agent Shares
        # Borrowers: Have nominal debt. Inflation HELPS (erodes real value). Rates HURT.
        # Savers: Have nominal assets. Inflation HURTS. Rates HELP.
        
        C = np.zeros((T, 2, 1)) # (T, n_a, n_e=1)
        
        # Sensitivity Parameters
        # Borrower: C = Y - Theta_B * (i - pi) ... Roughly
        # Real Rate Effect: High r -> High payments -> Low C_b
        theta_b = 2.0 
        
        # Saver: C = Y + Theta_S * (i - pi) - InflationTax
        # High r -> High income -> High C_s
        # But High pi -> Wealth erosion.
        theta_s = 2.0
        
        # Calibration for Welfare calculation (Deviation from SS)
        # Borrower
        # Hurt by Recession (y) and High Rates (r)
        C[:, 0, 0] = y - theta_b * r 
        
        # Saver
        # Hurt by Recession (y) but Helped by High Rates (r)
        # However, Fiscal Dominance (Low r, High pi) -> r is very low (negative real).
        # So Saver is getting crushed by negative real rates in FD.
        C[:, 1, 0] = y + theta_s * r
        
        return {'C': C}


def main():
    print("=" * 60)
    print("  Scenario: The Fiscal-Monetary Trap")
    print("  (A Welfare Analysis of Fiscal Dominance)")
    print("=" * 60)
    
    model = FiscalTrapModel(T=40)
    
    # Run Scenarios
    print("\n[1] Running Scenarios...")
    
    # Path A: Ignore Debt (Monetary Dominance) - "Fight Inflation"
    print("    A. Monetary Dominance (Fighting Inflation)...")
    path_md = model.simulate("monetary_dominance")
    
    # Path B: The Trap (Endogenous Switch) - "Succumb to Debt"
    print("    B. Fiscal Trap (Endogenous Regime Switch)...")
    path_fd = model.simulate("endogenous")
    
    # Welfare Inputs
    res_md = model.compute_consumption(path_md)
    res_fd = model.compute_consumption(path_fd)
    
    # print debug
    print(f"    Max Inflation (MD): {np.max(path_md['pi']):.2%}")
    print(f"    Max Inflation (FD): {np.max(path_fd['pi']):.2%}")
    print(f"    Max Debt (MD):      {np.max(path_md['B']):.2f} (Explosive!)")
    print(f"    Max Debt (FD):      {np.max(path_fd['B']):.2f} (Stabilized)")

    # 2. Welfare Analysis
    print("\n[2] The Verdict (Welfare Analysis)...")
    # Base: MD (Fight Inflation). Alt: FD (Allow Inflation).
    # Positive CEV -> FD is better. Negative -> MD is better.
    judge = WelfareAnalysis(base=res_md, alt=res_fd, sigma=2.0, beta=0.99)
    
    cev_grid = judge.compute_cev_grid()
    cev_borrower = np.mean(cev_grid[0, :])
    cev_saver = np.mean(cev_grid[1, :])
    
    print(f"    Borrower Limit (Low Asset): {cev_borrower*100:+.2f}% CEV")
    print(f"    Saver Limit (High Asset):   {cev_saver*100:+.2f}% CEV")
    
    winner = "Borrowers" if cev_borrower > 0 else "Savers"
    print(f"\n    => Under Fiscal Dominance (Inflation), {winner} are better off.")
    if cev_borrower > 0:
        print("       (Inflation eroded real debt, and low rates kept payments low.)")
    else:
        print("       (Recession was too deep?)")
        
    print(f"    => Savers suffering: {cev_saver*100:+.2f}%")
    print("       (Negative real rates destroyed their wealth return.)")

    # 3. Plotting
    print("\n[3] Visualizing the Trap...")
    t = np.arange(model.T)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Inflation & Rates
    ax = axes[0]
    ax.plot(t, path_md['pi'], label="Inflation (MD)", color='blue', linestyle=":")
    ax.plot(t, path_md['i'], label="Nominal Rate (MD)", color='blue')
    ax.plot(t, path_fd['pi'], label="Inflation (Trap)", color='red', linestyle=":")
    ax.plot(t, path_fd['i'], label="Nominal Rate (Trap)", color='red')
    ax.set_title("Policy: Fight or Flight?")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 2: Debt Dynamics
    ax = axes[1]
    ax.plot(t, path_md['B'], label="Debt (MD)", color='blue')
    ax.plot(t, path_fd['B'], label="Debt (Trap)", color='red')
    ax.axhline(model.Debt_Limit, color='k', linestyle='--', label="Debt Limit")
    ax.set_title("The Fiscal Trap (Debt)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Panel 3: Welfare (Distribution)
    ax = axes[2]
    welfare_diffs = [cev_borrower*100, cev_saver*100]
    bars = ax.bar(['Borrowers', 'Savers'], welfare_diffs, color=['green', 'red'])
    ax.axhline(0, color='k')
    ax.set_ylabel("Welfare Gain/Loss from Fiscal Dominance (%)")
    ax.set_title("Who wins in the Trap?")
    
    plt.suptitle(f"Scenario: The Fiscal-Monetary Trap (Shock={0.02:.1%})", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "scenario_fiscal_trap_output.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n=> Report saved: {output_path}")

if __name__ == "__main__":
    main()
