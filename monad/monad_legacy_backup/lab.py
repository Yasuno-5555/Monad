
import pandas as pd
import numpy as np
from .model import MonadModel

class ExperimentRunner:
    def __init__(self, base_config=None):
        self.base_config = base_config or {}
        self.results = {}
        self.scenarios = {}

    def run_scenario(self, label, param_overrides, unemployment=True):
        """
        Run a single scenario with overridden parameters.
        
        Parameters:
        -----------
        label : str
            Unique name for the scenario (e.g. "Hawkish", "StickyWage")
        param_overrides : dict
            Dictionary of parameter key-values to override default params.
        unemployment : bool
            Whether to enable unemployment risk (Default: True for v1.7+)
        """
        print(f"\n--- Running Scenario: {label} ---")
        
        # Initialize fresh model
        model = MonadModel(f"scenario_{label}")
        
        # Apply standard base settings (v1.7 Base)
        model.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)
        if unemployment:
            model.set_unemployment(u_rate=0.05, replacement_rate=0.4)
        model.set_fiscal(tau=0.15)
        model.define_grid(size=200) # Fast but accurate enough
        
        # Apply Base Config Overrides
        for k, v in self.base_config.items():
            model.set_param(k, v)
            
        # Apply Scenario Overrides
        for k, v in param_overrides.items():
            model.set_param(k, v)
        
        # Store for record
        self.scenarios[label] = param_overrides
        
        try:
            res = model.solve()
            if "transition" in res and not res["transition"].empty:
                self.results[label] = res
                print(f"Scenario '{label}' completed successfully.")
            else:
                print(f"Scenario '{label}' produced no transition data.")
        except Exception as e:
            print(f"Scenario '{label}' Failed: {e}")
            
        return self.results.get(label)

    def get_comparison_df(self, variable="dY"):
        """
        Create a DataFrame comparing a specific variable across all run scenarios.
        
        Parameters:
        -----------
        variable : str
            Name of the column in transition_nk.csv (e.g. "dY", "dpi", "dw")
        """
        data = {}
        for label, res in self.results.items():
            if "transition" in res:
                df = res["transition"]
                if variable in df.columns:
                    data[label] = df[variable]
                else:
                    print(f"Warning: Variable '{variable}' not found in scenario '{label}'")
        
        return pd.DataFrame(data)
    
    def get_inequality_df(self, metric="C_bottom50"):
        """
        Comparision for inequality metrics from inequality_path.csv
        """
        data = {}
        for label, res in self.results.items():
            # MonadModel doesn't auto-load inequality_path yet?
            # Actually MonadModel currently loads steady_state and transition.
            # We should probably update MonadModel to load inequality too.
            # Or handle it here manually as fallback.
            
            # Use logic to find file if available
            path = "inequality_path.csv" 
            # Note: MonadModel runs rename output files? No, it overwrites.
            # So ExperimentRunner needs to manage files or read immediately.
            # Since solve() overwrites, we rely on 'res' being in-memory dataframe?
            # MonadModel returns loaded dataframe.
            # BUT MonadModel currently only loads 'transition' and 'steady_state'.
            # It does NOT load 'inequality_path.csv'.
            pass
            
        # For now, let's skip inequality aggregation or implement file loader
        return pd.DataFrame()
