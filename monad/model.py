import os
import subprocess
import json
from .solver import NKHANKSolver

class ModelObject:
    """Helper to mutate model components."""
    def __init__(self, name, parent):
        self.name = name
        self.parent = parent
    
    def mutate(self, **kwargs):
        self.parent._log_change(self.name, kwargs)
        if self.name == "solver":
            for k, v in kwargs.items():
                self.parent.overrides[f"solver_settings.{k}"] = v
        else:
            self.parent.overrides.update(kwargs)
        return self.parent # Return parent model to allow chaining objects: m.object(A).mutate().object(B)...

    def toggle(self, feature):
        self.mutate(**{feature: True})
        return self.parent

class MonadModel:
    """
    High-level wrapper for the Monad Engine.
    Automates the pipeline: Config -> C++ Engine -> GPU -> CSV -> Python Solver.
    """
    def __init__(self, binary_path, working_dir="."):
        self.binary_path = binary_path
        self.working_dir = working_dir
        self.objects = {
            "policy_rule":  ModelObject("policy_rule", self),
            "fiscal_rule":  ModelObject("fiscal_rule", self),
            "household":    ModelObject("household", self),
            "market":       ModelObject("market", self),
            "solver":       ModelObject("solver", self)
        }
        self.overrides = {}
        self.history_log = []
        self.policy_block = None

    def attach(self, policy_block):
        """
        Inject a Policy Object (Phase 2).
        1. Registers new variables (Superset Strategy).
        2. Applies overrides.
        """
        # 1. Variable Injection (Mock for now, normally affects DAG)
        new_vars, new_eqs = policy_block.export()
        if new_vars:
            self._log_change("system", {"new_vars": list(new_vars.keys())})
        if new_eqs:
            # Here we would normally "inject" into the DAG builder
            print(f"[MonadModel] Injecting {len(new_eqs)} equations from Policy.")
            # For verification test, let's store them in a temp attribute if not existing
            if not hasattr(self, 'injected_equations'):
                self.injected_equations = {}
            self.injected_equations.update(new_eqs)
            
        # 2. Apply Parameter Overrides immediately
        overrides = policy_block.get_overrides()
        if overrides:
            self.overrides.update(overrides)
            
        self.policy_block = policy_block
        self._log_change("policy", {"attached": policy_block.name})
        
        # Link for live mutation updates
        policy_block.link_to_model(self)
        
        return self

    def _log_change(self, obj_name, changes):
        """Record a mutation event."""
        event = {
            "object": obj_name,
            "changes": changes,
            "step": len(self.history_log) + 1
        }
        self.history_log.append(event)

    def history(self):
        """Print the mutation history."""
        print("--- Model Mutation History ---")
        for h in self.history_log:
            print(f"[{h['step']}] {h['object']}: {h['changes']}")
        return self.history_log

    def object(self, name):
        return self.objects.get(name)

    def run(self, params=None, config_path="test_model.json", diff_inputs=None):
        """
        Executes the full pipeline.
        1. Updates JSON configuration (if params provided).
        2. Runs C++ Engine.
        3. Initializes and returns NKHANKSolver (or Result if diff_inputs provided).
        """
        # Merge manual params with overrides
        combined_params = self.overrides.copy()
        if params:
            combined_params.update(params)

        if combined_params:
            self._update_config(config_path, combined_params)

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
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Engine Execution Failed or not found ({e}).")
            print("Switched to [Cache-based Analysis Mode].")
            pass

        # Paths are relative to working_dir
        path_R = os.path.join(self.working_dir, "gpu_jacobian_R.csv")
        path_Z = os.path.join(self.working_dir, "gpu_jacobian_Z.csv")

        if not os.path.exists(path_R) or not os.path.exists(path_Z):
            raise RuntimeError("Critical Failure: No GPU hardware found AND no cached data available.\n"
                               "To run Monad Engine, you need a CUDA GPU or pre-computed 'gpu_jacobian_*.csv' files.")
        
        if 'e' in locals():
            print("\n" + "="*50)
            print("[SAFE MODE]")
            print("Engine execution failed.")
            print("Using cached GPU-generated artifacts.")
            print("="*50 + "\n")

        # Config Loader for Solver Settings
        # Re-read config (since params might have updated it, or we use defaults)
        full_config_path = os.path.join(self.working_dir, config_path)
        solver_config = {}
        try:
            with open(full_config_path, 'r') as f:
                data = json.load(f)
                solver_config = data.get('solver_settings', {})
        except:
            pass # Use defaults
            
        use_soe = solver_config.get('open_economy', False)
        use_zlb = solver_config.get('zlb', False)
        
        # Policy Object FORCE (Phase 2)
        if self.policy_block:
             use_zlb = True # Force PiecewiseSolver if policy object attached
        
        print("--- Monad Lab: Initializing Solver Stack ---")
        print(f"  > Open Economy: {'ON' if use_soe else 'OFF'}")
        print(f"  > Nonlinear/Policy: {'ON' if use_zlb else 'OFF'}")

        # Factory Logic
        from .solver import NKHANKSolver, SOESolver
        from .nonlinear import PiecewiseSolver

        if use_soe:
            # Base is Open Economy
            base_solver = SOESolver(path_R, path_Z)
        else:
            # Base is Standard Closed Economy (Default)
            base_solver = NKHANKSolver(path_R, path_Z)
            
        if use_zlb:
            # Wrap in Nonlinear Solver
            # Pass policy block if relevant
            solver = PiecewiseSolver(base_solver, policy_block=self.policy_block)
            if diff_inputs:
                return solver.solve(diff_inputs)
            return solver
        else:
            if diff_inputs:
                return base_solver.solve(diff_inputs)
            return base_solver

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
