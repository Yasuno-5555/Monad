
from monad.model import MonadModel
import subprocess

def run_debug():
    # Setup config
    m = MonadModel("debug_run")
    m.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
    m.set_unemployment(u_rate=0.05, replacement_rate=0.4)
    m.set_fiscal(tau=0.15)
    m.define_grid(size=200)
    # Manually generate config to avoid MonadModel attribute error
    import json
    import numpy as np
    
    config = {
        "model_name": "debug",
        "parameters": {
            "beta": 0.96, "sigma": 2.0, "alpha": 0.33, "A": 1.0, "r_guess": 0.02,
            "phi_pi": 2.0, "theta_w": 0.75,
            "unemployment_benefit": 0.4
        },
        "agents": [{
            "name": "Household",
            "grids": { "asset_a": {"type": "Log-spaced", "n_points": 200, "min": 0.001, "max": 100.0, "potency": 2.0} }
        }],
        "income_process": {
            "n_z": 1,
            "z_grid": [1.0],
            "transition_matrix": [1.0]
        }
    }
    
    with open("debug_config.json", "w") as f:
        json.dump(config, f, indent=4)
        
    print("Running MonadEngine.exe debug_config.json...")
    result = subprocess.run(["MonadEngine.exe", "debug_config.json"], capture_output=True, text=True)
    
    print("--- STDOUT ---")
    print(result.stdout)
    print("--- STDERR ---")
    print(result.stderr)
    print(f"Exit Code: {result.returncode}")

if __name__ == "__main__":
    run_debug()
