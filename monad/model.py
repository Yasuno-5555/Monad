import json
import subprocess
import pandas as pd
import numpy as np
import os
import shutil
from monad.process import rouwenhorst

class MonadModel:
    def __init__(self, name="Standard_HANK"):
        self.name = name
        self.params = {
            "beta": 0.96,
            "sigma": 2.0, 
            "alpha": 0.33,
            "A": 1.0,
            "r_guess": 0.02
        }
        # Income Risk Parameters (Default: No Risk)
        self.risk_params = {
            "rho": 0.9,
            "sigma_eps": 0.0, # 0.0 means deterministic
            "n_z": 1
        }
        self.grid_config = {
            "type": "Log-spaced", 
            "n_points": 500, 
            "min": 0.001, # Avoid 0.0 for CRRA
            "max": 100.0,
            "potency": 2.0
        }
        self.run_shock = True

    def set_param(self, key, value):
        self.params[key] = value

    def set_risk(self, rho, sigma_eps, n_z=7):
        self.risk_params["rho"] = rho
        self.risk_params["sigma_eps"] = sigma_eps
        self.risk_params["n_z"] = n_z
    
    def set_fiscal(self, lambda_=1.0, tau=0.0, transfer=0.0):
        self.params["tax_lambda"] = lambda_
        self.params["tax_tau"] = tau
        self.params["tax_transfer"] = transfer
    
    def set_unemployment(self, u_rate: float = 0.05, replacement_rate: float = 0.4):
        """
        Enable unemployment risk in the income process.
        
        Parameters:
        -----------
        u_rate : float
            Steady-state unemployment rate (default 5%)
        replacement_rate : float
            Unemployment benefit as fraction of mean wage (default 40%)
        """
        self.risk_params["unemployment"] = True
        self.risk_params["u_rate"] = u_rate
        self.risk_params["replacement_rate"] = replacement_rate
        
        # Also store in params for C++ to access
        self.params["unemployment_benefit"] = replacement_rate

    def define_grid(self, size=500, type="Log-spaced", max_asset=100.0):
        self.grid_config["n_points"] = size
        self.grid_config["type"] = type
        self.grid_config["max"] = max_asset

    def solve(self, exe_path="MonadEngine"):
        # 1. Generate Income Process
        rho = self.risk_params["rho"]
        sigma_eps = self.risk_params["sigma_eps"]
        n_z = self.risk_params["n_z"]
        use_unemployment = self.risk_params.get("unemployment", False)

        if use_unemployment and sigma_eps > 0.0 and n_z > 1:
            # v1.7: Labor process with unemployment
            from monad.process import build_labor_process
            u_rate = self.risk_params.get("u_rate", 0.05)
            replacement_rate = self.risk_params.get("replacement_rate", 0.4)
            
            z_grid, Pi, is_unemployed = build_labor_process(
                rho, sigma_eps, n_z, u_rate, replacement_rate
            )
            n_z = len(z_grid)  # Now includes unemployment state
            
        elif sigma_eps > 0.0 and n_z > 1:
            # Standard Rouwenhorst (no unemployment)
            z_grid, Pi = rouwenhorst(rho, sigma_eps, n_z)
        else:
            # Deterministic / Representative Agent Fallback
            z_grid = np.array([1.0])
            Pi = np.array([[1.0]])
            n_z = 1

        # 2. Generate IR JSON
        config_data = {
            "model_name": self.name,
            "parameters": self.params,
            "agents": [{
                "name": "Household",
                "grids": { "asset_a": self.grid_config }
            }],
            "income_process": {
                "n_z": n_z,
                "z_grid": z_grid.tolist(),
                "transition_matrix": Pi.flatten().tolist() # Flatten for Easy C++ Parse
            }
        }
        
        config_filename = "model_config.json"
        with open(config_filename, "w") as f:
            json.dump(config_data, f, indent=4)

        # 3. Resolve Executable Path
        if not os.path.exists(exe_path):
             # Try seeking in standard build folders
             candidates = [
                 os.path.join("build_phase3", "Release", "MonadEngine.exe"),
                 os.path.join("build_phase3", "MonadEngine.exe"),
                 os.path.join(".", "MonadEngine.exe"),
                 os.path.join("build", "Release", "MonadEngine.exe"),
                 os.path.join("build", "MonadEngine.exe")
             ]
             for c in candidates:
                 if c.endswith(".exe") and os.path.exists(c): # Windows Check
                     exe_path = c
                     break
                 elif os.path.exists(c): # Linux/Mac without extension
                     exe_path = c
                     break
             else:
                 raise FileNotFoundError(f"Executable {exe_path} not found.")

        print(f"Running {exe_path} with {config_filename}...")
        
        # 3. Run Engine
        try:
            result = subprocess.run([exe_path, config_filename], capture_output=False, text=True)
            
            if result.returncode != 0:
                print("--- Engine Output (stdout) ---")
                print(result.stdout)
                print("--- Engine Error (stderr) ---")
                print(result.stderr)
                raise RuntimeError("Engine execution failed.")
            
            # print(result.stdout) # Uncomment for debug

        except Exception as e:
            raise e

        # 4. Load Results
        results = {}
        if os.path.exists("steady_state.csv"):
            results["steady_state"] = pd.read_csv("steady_state.csv")
            print("Loaded steady_state.csv")
            
        # v1.7: Prioritize NK transition file
        if os.path.exists("transition_nk.csv"):
            results["transition"] = pd.read_csv("transition_nk.csv")
            print("Loaded transition_nk.csv")
        elif os.path.exists("transition.csv"):
            results["transition"] = pd.read_csv("transition.csv")
            print("Loaded transition.csv")
            
        # v1.8: Load inequality path
        if os.path.exists("inequality_path.csv"):
            results["inequality"] = pd.read_csv("inequality_path.csv")
            print("Loaded inequality_path.csv")
            
        return results
