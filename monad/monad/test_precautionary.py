from monad.model import MonadModel
import pandas as pd
import numpy as np

print("--- Testing Monad Engine v1.2 (HANK) ---")

model = MonadModel()
model.set_param("beta", 0.96)
model.set_param("sigma", 2.0)
model.set_param("alpha", 0.33)
model.set_param("A", 1.0)
model.set_param("r_guess", 0.03) # Start lower than 1/beta-1=0.0416

# Introduce Risk
rho = 0.9
sigma_eps = 0.4 # VERY High risk to force savings
n_z = 7
print(f"Setting Risk: rho={rho}, sigma_eps={sigma_eps}, n_z={n_z}")
model.set_risk(rho, sigma_eps, n_z)

model.define_grid(size=200, type="Log-spaced", max_asset=150.0)

try:
    results = model.solve()
    print("Solved successfully.")
    
    if "steady_state" in results:
        df = results["steady_state"]
        print(f"Dataframe Shape: {df.shape}")
        print(df.head())
        print(df.tail())
        
        # Calculate aggregate K
        dist_sum = df["distribution"].sum()
        print(f"Distribution Sum: {dist_sum}")
        
        K = (df["asset"] * df["distribution"]).sum()
        print(f"Aggregate Capital K: {K}")
        
        # Determine SS r
        alpha = 0.33
        A = 1.0
        beta = 0.96
        
        r_time_pref = (1.0/beta) - 1.0
        print(f"Time Preference Rate (rho): {r_time_pref:.5f}")

        if K > 0.001:
            r_implied = alpha * A * (K ** (alpha - 1.0))
            print(f"Implied Equilibrium r: {r_implied:.5f}")
            
            if r_implied < r_time_pref:
                print(f"[PASS] r ({r_implied:.5f}) < rho ({r_time_pref:.5f}). Precautionary Savings confirmed!")
            else:
                 print(f"[WARN] r ({r_implied:.5f}) >= rho ({r_time_pref:.5f}). Not enough savings.")
        else:
             print("[FAIL] Capital K is near zero/negative.")
    else:
        print("No steady_state results found.")

except Exception as e:
    print(f"Error: {e}")
