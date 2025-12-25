import os
from monad.model import MonadModel
import pandas as pd
import numpy as np

def run_sweep():
    print("--- Stabilization Sweep ---")
    results = {}
    
    # Range of phi_pi to test stability frontier
    for phi in [1.5, 1.55, 1.6, 1.8, 2.0, 2.5]:
        m = MonadModel(f"sweep_{phi}")
        m.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
        m.set_unemployment(u_rate=0.05, replacement_rate=0.4)
        m.set_fiscal(tau=0.15)
        m.define_grid(size=200)
        
        m.set_param("theta_w", 0.75)
        m.set_param("phi_pi", phi)
        
        try:
            res = m.solve()
            if "transition" in res:
                dr = res["transition"]["dr"]
                peak = dr.min()
                results[phi] = peak
                msg = f"phi={phi:.2f} -> Peak Drop: {peak:.6e}"
                print(msg)
                with open("sweep_results.txt", "a") as f:
                    f.write(msg + "\n")
            else:
                results[phi] = "NaN"
                msg = f"phi={phi:.2f} -> No Data"
                print(msg)
                with open("sweep_results.txt", "a") as f:
                    f.write(msg + "\n")
        except Exception as e:
            print(f"phi={phi:.2f} -> Error: {e}")

if __name__ == "__main__":
    run_sweep()
