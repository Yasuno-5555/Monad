import os
import subprocess
import json
from .solver import NKHANKSolver

class MonadModel:
    """
    High-level wrapper for the Monad Engine.
    Automates the pipeline: Config -> C++ Engine -> GPU -> CSV -> Python Solver.
    """
    def __init__(self, binary_path, working_dir="."):
        self.binary_path = binary_path
        self.working_dir = working_dir

    def run(self, params=None, config_path="test_model.json"):
        """
        Executes the full pipeline.
        1. Updates JSON configuration (if params provided).
        2. Runs C++ Engine.
        3. Initializes and returns NKHANKSolver.
        """
        if params:
            self._update_config(config_path, params)

        print(f"--- Monad Engine: Launching {self.binary_path} ---")
        try:
            # Run C++ executable
            # Capture output to avoid cluttering python console unless error
            result = subprocess.run(
                [self.binary_path], 
                cwd=self.working_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print("Engine finished successfully.")
            # print(result.stdout) # Uncomment for debug
        except subprocess.CalledProcessError as e:
            print("Engine Execution Failed!")
            print(e.stderr)
            raise RuntimeError("C++ Engine crashed.")

        # Paths are relative to working_dir
        path_R = os.path.join(self.working_dir, "gpu_jacobian_R.csv")
        path_Z = os.path.join(self.working_dir, "gpu_jacobian_Z.csv")

        print("--- Monad Lab: Loading Data ---")
        return NKHANKSolver(path_R, path_Z)

    def _update_config(self, config_path, params):
        """
        Reads existing config, updates params, writes back.
        This assumes a simple flat structure or knows the schema.
        Note: The C++ loader expects specific nesting. 
        For v4.0 verification, we used a fixed `test_model.json`.
        Here we implement a simple update if the JSON structure permits.
        """
        full_path = os.path.join(self.working_dir, config_path)
        
        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            # Update 'parameters' section
            if 'parameters' not in data:
                data['parameters'] = {}
            
            for k, v in params.items():
                data['parameters'][k] = v
                
            with open(full_path, 'w') as f:
                json.dump(data, f, indent=4)
                
            print(f"Configuration updated: {config_path}")
            
        except Exception as e:
            print(f"Warning: Failed to update config {config_path}: {e}")
            print("Proceeding with existing configuration.")
