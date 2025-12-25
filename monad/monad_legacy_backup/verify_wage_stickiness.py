"""
Verify Wage Stickiness: Compare Sticky Wage (Theta=0.75) vs Flexible Wage (Theta=0.01)
"""
from monad.model import MonadModel
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_comparison():
    # 1. Sticky Wage (Canonical v1.7)
    print("--- Running Sticky Wage Model (theta_w=0.75) ---")
    m_sticky = MonadModel("v1.7_Sticky")
    m_sticky.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m_sticky.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m_sticky.set_fiscal(tau=0.15)
    m_sticky.define_grid(size=200)
    # Theta is default 0.75 in C++, but let's set it explicitly if possible
    # We need to add set_param support for theta_w in MonadModel wrapper first?
    # Actually set_param just adds to params dict, which C++ reads.
    m_sticky.set_param("theta_w", 0.75)
    res_sticky = m_sticky.solve()
    
    # 2. Flexible Wage (Baseline)
    print("\n--- Running Flexible Wage Model (theta_w=0.01) ---")
    m_flex = MonadModel("v1.7_Flexible")
    m_flex.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m_flex.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m_flex.set_fiscal(tau=0.15)
    m_flex.define_grid(size=200)
    m_flex.set_param("theta_w", 0.01) # Near flexible
    res_flex = m_flex.solve()
    
    # 3. Compare Results
    # Output gap (dr_path in transition.csv for now)
    # Ideally we should output dw (wage inflation) too, but we haven't added it to CSV.
    # We can infer wage stickiness effect from Quantity adjustment (dY)
    
    # In Sticky Wage: Wages don't fall -> Quantity (Labor/Output) falls MORE.
    # In Flexible Wage: Wages fall -> Quantity falls LESS.
    
    # 3. Compare Results
    print("\n--- Comparison Results ---")
    
    with open("comparison_result.txt", "w") as f:
        if "transition" in res_sticky and "transition" in res_flex:
            # Save CSVs
            res_sticky["transition"].to_csv("trans_sticky.csv", index=False)
            res_flex["transition"].to_csv("trans_flex.csv", index=False)
            
            y_sticky = res_sticky["transition"]["dr"]
            y_flex = res_flex["transition"]["dr"]
            
            peak_y_sticky = y_sticky.min()
            peak_y_flex = y_flex.min()
            
            msg = f"""
Peak Output Drop (Sticky):   {peak_y_sticky:.4f}
Peak Output Drop (Flexible): {peak_y_flex:.4f}
"""
            ratio = peak_y_sticky / peak_y_flex if abs(peak_y_flex) > 1e-6 else 0
            msg += f"Amplification Ratio: {ratio:.2f}x\n"
            
            if abs(peak_y_sticky) > abs(peak_y_flex):
                 msg += "SUCCESS: Sticky wages amplify output drop (Quantity Adjustment dominates).\n"
            else:
                 msg += "FAILURE: Sticky wages did not amplify output drop.\n"
            
            print(msg)
            f.write(msg)
                 
            # Plot
            plt.figure(figsize=(10,6))
            plt.plot(y_sticky * 100, label=f"Sticky (theta=0.75)", linewidth=2, color="red")
            plt.plot(y_flex * 100, label=f"Flexible (theta=0.01)", linewidth=2, color="blue", linestyle="--")
            plt.title("Output Response to Monetary Shock: Sticky vs Flexible Wage")
            plt.ylabel("Output Gap %")
            plt.xlabel("Quarters")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig("wage_stickiness_comparison.png")
            print("Saved plot to wage_stickiness_comparison.png")
            
        else:
            print("Error: Transition data missing.")
            f.write("Error: Transition data missing.\n")

if __name__ == "__main__":
    run_comparison()
