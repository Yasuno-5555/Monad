
import os
from monad.model import MonadModel
import pandas as pd

def check_columns():
    print("--- Verifying CSV Columns ---")
    if os.path.exists("transition_nk.csv"):
        os.remove("transition_nk.csv")
        
    m = MonadModel("check_col")
    m.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m.set_fiscal(tau=0.15)
    m.define_grid(size=200)
    m.set_param("phi_pi", 2.0) # Stable
    m.set_param("theta_w", 0.75)
    
    try:
        res = m.solve()
        if "transition" in res:
            cols = list(res["transition"].columns)
            print(f"Columns: {cols}")
            
            required = ["dY", "dC", "dN", "dw", "dpi", "di", "dreal_r", "dB", "dT"]
            missing = [c for c in required if c not in cols]
            
            if not missing:
                print("SUCCESS: All required columns present.")
            else:
                print(f"FAILURE: Missing columns: {missing}")
        else:
            print("FAILURE: No transition data loaded.")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_columns()
