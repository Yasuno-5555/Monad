"""
Verify Stabilization: Tuning phi_pi
"""
import os
from monad.model import MonadModel
import pandas as pd

def clean_outputs():
    for f in ["transition_nk.csv"]:
        if os.path.exists(f): 
            os.remove(f)

def run_stabilization_test(phi_pi_val):
    clean_outputs()
    print(f"\n--- Running phi_pi={phi_pi_val} ---")
    
    m = MonadModel(f"stabilization_{phi_pi_val}")
    m.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m.set_fiscal(tau=0.15)
    m.define_grid(size=200)
    
    # Parameters tuning
    m.set_param("theta_w", 0.75)   # Sticky
    m.set_param("phi_pi", phi_pi_val) # Tuning target
    
    try:
        res = m.solve()
        if "transition" in res:
            dr = res["transition"]["dr"]
            peak_drop = dr.min()
            print(f"[Result] phi_pi={phi_pi_val} -> Peak Drop: {peak_drop:.4f}")
            
            # Save for record
            res["transition"].to_csv(f"trans_phi_{phi_pi_val}.csv", index=False)
            return peak_drop
        else:
            print("[Error] No transition data.")
            return None
    except Exception as e:
        print(f"[Error] {e}")
        return None

if __name__ == "__main__":
    # Baseline (Explosive)
    run_stabilization_test(1.5)
    
    # Tuned (Target)
    run_stabilization_test(2.0)
    run_stabilization_test(3.0)
