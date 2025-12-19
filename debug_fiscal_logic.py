"""
Debug: Fiscal Trap Logic Verification
=====================================
User asked: "Is the simulation correct?"
Hypothesis: The result is driven by Real Interest Rate (r - pi).

This script replays the Fiscal Trap logic and explicitly plots:
1. Nominal Rate (i)
2. Inflation (pi)
3. Real Rate (r = i - pi) -> The 'Price of Time'

If r < 0, then:
- Borrowers gain (Debt erodes)
- Savers lose (Wealth erodes)
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from examples.scenario_fiscal_monetary_trap import FiscalTrapModel

def main():
    print("--- Debug: Verifying Mechanism ---")
    model = FiscalTrapModel(T=40)
    
    # 1. Re-run Fiscal Trap Scenario
    path_fd = model.simulate("endogenous")
    
    pi = path_fd['pi']
    i = path_fd['i']
    r_real = path_fd['r'] # i - pi
    
    # 2. Analysis
    min_r = np.min(r_real)
    print(f"Min Real Rate: {min_r:.2%}")
    
    if min_r < 0:
        print("=> CONFIRMED: Real Interest Rate is NEGATIVE.")
        print("   This explains why Savers lose and Borrowers win.")
        print("   Theory: Euler Equation implies consumption/saving relies on r_real.")
    else:
        print("=> WARNING: Real Rate is positive? Result might be wrong.")
        
    # 3. Plot for User
    t = np.arange(model.T)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.plot(t, i, label="Nominal Rate (i)", color='gray', linestyle='--')
    ax.plot(t, pi, label="Inflation (pi)", color='red', linestyle='--')
    
    # Fill the 'Gap' (Real Rate)
    ax.fill_between(t, i, pi, where=(i < pi), color='red', alpha=0.2, label="Negative Real Rate (The Transfer)")
    ax.fill_between(t, i, pi, where=(i >= pi), color='green', alpha=0.2, label="Positive Real Rate")
    
    ax.plot(t, r_real, label="Real Rate (r = i - pi)", color='black', linewidth=2)
    
    ax.set_title("The Mechanism: Negative Real Rates", fontsize=14)
    ax.axhline(0, color='black', linewidth=1)
    ax.legend()
    ax.grid(alpha=0.3)
    
    output_path = os.path.join(os.path.dirname(__file__), "debug_mechanism_output.png")
    plt.savefig(output_path)
    print(f"\n=> Evidence saved: {output_path}")

if __name__ == "__main__":
    main()
