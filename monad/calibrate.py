from monad.model import MonadModel
import pandas as pd
import numpy as np

print("--- Monad Engine v1.3 Calibration ---")

model = MonadModel()
# TARGET: r ~ 2-4% per annum (quarterly 0.5-1.0%?) 
# Usually Aiyagari is annual. Let's assume Annual parameters.
# Beta = 0.96 -> Rho = 4.16%
# If we want r ~ 3%, we need Beta slightly higher or just verify r < 4%.
# User suggested Beta=0.99 for r=4%.

beta = 0.985
model.set_param("beta", beta) 
model.set_param("sigma", 2.0)
model.set_param("alpha", 0.33)
model.set_param("A", 1.0)
model.set_param("r_guess", 0.03)

# Risk Params (Milder)
rho_z = 0.96
sigma_eps = 0.20 
n_z = 7

print(f"Calibration: beta={beta}, rho_z={rho_z}, sigma_eps={sigma_eps}")
model.set_risk(rho_z, sigma_eps, n_z)
model.define_grid(size=200, type="Log-spaced", max_asset=200.0)

try:
    results = model.solve()
    
    if "steady_state" in results:
        df = results["steady_state"]
        K = (df["asset"] * df["distribution"]).sum()
        print(f"Aggregate Capital K: {K:.4f}")
        
        # Implied r
        alpha = 0.33
        r_implied = alpha * (K ** (alpha - 1.0)) # Assuming delta=0 for now/implicit
        # Wait, usually r = alpha * K^(alpha-1) - delta. 
        # C++ engine code: r_dual / (alpha * A) ... K_dem = (r/alpha A)^(1/alpha-1)
        # So r = alpha A K^(alpha-1)
        
        print(f"Implied Equilibrium r: {r_implied:.5f}")
        print(f"Time Preference (1/beta - 1): {1.0/beta - 1.0:.5f}")
        
    else:
        print("Model failed to solve.")

except Exception as e:
    print(f"Error: {e}")
