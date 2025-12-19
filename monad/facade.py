"""
Monad Facade API
The "Thinking Library" interface for Monad Studio.
Provides a high-level, fluent API for model orchestration, scenario comparison, and visualization.
"""

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict, List, Optional
from pathlib import Path
from enum import Enum

# Internal imports
import copy
import json
import datetime
import traceback
import hashlib

from .model import MonadModel
from .solver import NKHANKSolver, SOESolver
from .nonlinear import NewtonSolver
from .analysis import MonadAnalysis

class DeterminacyStatus(str, Enum):
    UNIQUE = "unique"
    INDETERMINATE = "indeterminate"
    UNSTABLE = "unstable"
    UNKNOWN = "unknown"

class MonadResult:
    """
    Unified container for simulation results.
    "An atom of thought."
    """
    def __init__(self, data: Dict[str, np.ndarray], params: dict, meta: dict = None):
        self.data = data
        self.params = copy.deepcopy(params) # Immutable snapshot
        self.meta = copy.deepcopy(meta or {})
        self.meta['timestamp'] = datetime.datetime.now().isoformat()
        self.T = len(next(iter(data.values()))) if data else 0

    def determinacy(self) -> Dict[str, Union[str, List[float]]]:
        """
        Diagnose the stability/uniqueness of the equilibrium.
        Returns:
            dict with 'status' (DeterminacyStatus), 'eigenvalues', and 'notes'.
        """
        status = DeterminacyStatus.UNKNOWN
        notes = "No diagnostic data available."
        metrics = {}
        
        # Check metadata for solver type
        nonlinear = self.meta.get('nonlinear', False)
        
        if nonlinear:
            # Newton Solver Diagnostics
            if 'convergence' in self.meta:
                conv = self.meta['convergence']
                if conv.get('success', False):
                    status = DeterminacyStatus.UNIQUE # Tentative
                    notes = f"Newton solver converged in {conv.get('iterations')} iterations."
                else:
                    status = DeterminacyStatus.UNSTABLE
                    notes = f"Newton solver failed to converge. Error: {conv.get('message')}"
        else:
            # Linear / SSJ Diagnostics
            status = DeterminacyStatus.UNIQUE
            notes = "Linear system solved successfully. (Eigenvalue analysis not yet implemented)"

        return {
            "status": status.value, # Export as string for JSON
            "notes": notes,
            "metrics": metrics
        }
    
    def fingerprint(self) -> str:
        """
        Generate a Reproducilibity Hash (SHA-256).
        Hashes parameters, shocks, and solver logic.
        """
        # Create a stable string representation
        content = {
            'params': self.params,
            'shocks': self.meta.get('shocks', {}),
            'nonlinear': self.meta.get('nonlinear'),
            'zlb': self.meta.get('zlb')
        }
        # Sort keys for stability
        serialized = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def cite(self) -> str:
        """
        Return the BibTeX citation for the engine used.
        """
        return """
@article{Auclert2021,
  title={Using the Sequence-Space Jacobian to Solve and Estimated Heterogeneous-Agent Models},
  author={Auclert, Adrien and Bardóczy, Bence and Rognlie, Matthew and Straub, Ludwig},
  journal={Econometrica},
  year={2021},
  note={Compute by Monad Studio Engine v4.0}
}
"""

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame(self.data)

    def to_csv(self, path: str):
        """Export results to CSV."""
        self.to_dataframe().to_csv(path, index=False)
        print(f"[Monad] Results exported to {path}")

    def export(self, path: str):
        """
        Export results with reproducibility metadata sidecar.
        Args:
            path: Path to main output file (e.g., 'fig1.csv' or 'fig1.pdf')
        """
        p = Path(path)
        # 1. Save Data/Figure
        if p.suffix == '.csv':
            self.to_csv(path)
        elif p.suffix in ['.pdf', '.png', '.jpg']:
             # If plot() was called before, we might save fig, but here we usually assume
             # export is for data or we trigger a default plot?
             # For API clarity, if user calls export on Result, usually expects Data serialization
             # OR if they want plot, they call plot(save_path=...).
             # Let's support CSV export primarily here, or Plot if explicitly requested?
             # User request implies: res.export("fig1.pdf") implies saving a plot.
             # But we don't know WHAT to plot. 
             # Let's assume export is primarily for DATA snapshot (JSON/CSV).
             # If extension is image, we call default plot.
             self.plot(save_path=path)
        else:
            # Default to JSON dump of data
            pass

        # 2. Save Metadata Sidecar (.meta.json)
        # remove extension from path and add .meta.json
        meta_path = p.with_name(p.stem + ".meta.json")
        
        sidebar_content = {
            "snapshot_timestamp": self.meta['timestamp'],
            "determinacy": self.determinacy(),
            "solver_settings": {
                "nonlinear": self.meta.get('nonlinear'),
                "zlb": self.meta.get('zlb')
            },
            "parameters": self.params,
            # 'shocks' should be in meta if passed from Monad
            "shock_definitions": self.meta.get('shocks', {})
        }
        
        with open(meta_path, 'w') as f:
            json.dump(sidebar_content, f, indent=2, default=str)
            
        print(f"[Monad] Reproducibility data saved to {meta_path}")

    def plot(self, variables: Union[str, List[str]] = None, title: str = None, save_path: str = None):
        """
        Quick visualization of results.
        Args:
            variables: Variable name or list of names to plot (e.g. "Y", ["Y", "pi"]). 
                       If None, plots key macro vars.
            title: Plot title.
            save_path: If provided, save figure to this path.
        """
        if variables is None:
            variables = ['Y', 'pi', 'r', 'C']
            # Filter available keys
            variables = [v for v in variables if v in self.data]
        
        if isinstance(variables, str):
            variables = [variables]

        n_vars = len(variables)
        cols = min(n_vars, 2)
        rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
        if n_vars == 1: axes = [axes]
        axes = np.array(axes).flatten()

        for i, var in enumerate(variables):
            if var in self.data:
                axes[i].plot(self.data[var], label=var, linewidth=2)
                axes[i].set_title(var)
                axes[i].grid(True, alpha=0.3)
                axes[i].axhline(0, color='gray', linestyle=':', linewidth=0.8)
        
        if title:
            fig.suptitle(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"[Monad] Figure saved to {save_path}")
        else:
            plt.show()
            
        return self

class Monad:
    """
    The Agent.
    Orchestrates the model lifecycle: Setup -> Shock -> Solve.
    """
    def __init__(self, config: Union[str, Dict] = "us_normal"):
        self.config_params = {}
        self.model_config = {}
        self.shocks = {}
        self.name = "Monad Model"
        
        # Load Configuration
        if isinstance(config, str):
            self.name = Path(config).stem
            self._load_preset(config)
        elif isinstance(config, dict):
            self.config_params = config
        
        # Core Components
        self._engine = None # MonadModel instance
        self._solver = None # Python Solver instance
        self.T = self.config_params.get('T', 50)

    def _load_preset(self, name: str):
        """Load parameters from a preset file (yaml/json)."""
        # Look in likely locations
        candidates = [
            name,
            f"presets/{name}",
            f"presets/{name}.yaml",
            f"presets/{name}.json"
        ]
        
        loaded = False
        for path in candidates:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    if path.endswith('.yaml') or path.endswith('.yml'):
                        raw_config = yaml.safe_load(f) or {}
                    else:
                        import json
                        raw_config = json.load(f)
                
                # Flatten or Extract Parameters
                # Monad Config expects flat: {'kappa': 0.1, ...}
                # Preset YAML might be: {'parameters': {'kappa': {'value': 0.1}}}
                
                self.config_params = {}
                if 'parameters' in raw_config:
                    for k, v in raw_config['parameters'].items():
                        if isinstance(v, dict) and 'value' in v:
                            self.config_params[k] = v['value']
                        else:
                            self.config_params[k] = v
                    # Copy other top-level keys if needed (like T)
                    for k, v in raw_config.items():
                        if k != 'parameters':
                            self.config_params[k] = v
                else:
                    self.config_params = raw_config

                print(f"[Monad] Loaded preset: {path}")
                loaded = True
                break
        
        if not loaded:
            print(f"[Monad] Warning: Preset '{name}' not found. Using empty config.")

    def setup(self, **kwargs):
        """
        Fluent configuration update.
        Example: m.setup(phi_pi=1.5, kappa=0.1)
        """
        self.config_params.update(kwargs)
        # Update T if changed
        if 'T' in kwargs:
            self.T = kwargs['T']
        return self

    def shock(self, name: str = 'monetary', size: float = -0.01, persistence: float = 0.8, path: np.ndarray = None):
        """
        Define a shock.
        Args:
            name: 'monetary', 'fiscal', 'productivity', 'r_star', etc.
            size: Initial magnitude.
            persistence: AR(1) persistence (rho).
            path: Custom shock path (overrides size/persistence).
        """
        if path is not None:
            val = np.array(path)
            if len(val) < self.T:
                val = np.pad(val, (0, self.T - len(val)))
            self.shocks[name] = val[:self.T]
        else:
            # Generate AR(1)
            t = np.arange(self.T)
            series = size * (persistence ** t)
            self.shocks[name] = series
            
        return self

    def solve(self, nonlinear: bool = False, zlb: bool = False, method: str = "auto") -> MonadResult:
        """
        Execute the thought process.
        
        Auto Solver Selection Logic:
        1. ZLB=True -> nonlinear=True, NewtonSolver with zlb checks.
        2. nonlinear=True -> NewtonSolver wrapping linear solver.
        3. Default -> Linear GE Solver.
        """
        # 0. Initialize Engine/Backend if needed
        # We assume the binary exists or we use cache.
        # Ideally MonadModel handles the C++ execution.
        if self._engine is None:
            self._engine = MonadModel("MonadTwoAsset.exe") # Default binary name
            # Run engine to ensure Jacobians are ready (or use cache)
            # This step might be heavy, but it ensures 'm' is ready to think.
            # We only run if we haven't or if params changed drastically?
            # For simplicity in Facade, we assume we want to run/ensure readiness.
            
            # Map facade params to MonadModel config json structure
            # MonadModel expects a JSON file. We might need to write a temp one or update test_model.json
            self._engine.run(params=self.config_params)

        # 1. Determine Solver Strategy
        if zlb:
            nonlinear = True
            
        print(f"[Monad] Solving... (Nonlinear={nonlinear}, ZLB={zlb})")

        # 2. Instantiate Base Linear Solver
        # Check if Open Economy
        is_soe = self.config_params.get('open_economy', False)
        
        # Solver paths (from MonadModel convention)
        working_dir = self._engine.working_dir
        path_R = os.path.join(working_dir, "gpu_jacobian_R.csv")
        path_Z = os.path.join(working_dir, "gpu_jacobian_Z.csv")
        
        if is_soe:
            base_solver = SOESolver(path_R, path_Z, T=self.T, params=self.config_params)
        else:
            base_solver = NKHANKSolver(path_R, path_Z, T=self.T, params=self.config_params)
            
        # 3. Apply Wrappers (The "Auto Selection")
        final_solver = base_solver
        
        if nonlinear:
            # Wrap in Newton
            nl_solver = NewtonSolver(base_solver)
            nl_solver.zlb_enabled = zlb
            final_solver = nl_solver
            
            # Newton solver expects .solve_nonlinear(shock_path)
            # We need to map our named shocks to what Newton expects.
            # Typically Newton solves for a sequence of 'r_star' or similar shocks.
            
            # Map abstract shocks to solver inputs
            # TODO: robustify this mapping.
            # For now, support 'monetary' (dr_star) and 'r_star'
            shock_vec = np.zeros(self.T)
            if 'monetary' in self.shocks:
                shock_vec += self.shocks['monetary']
            if 'r_star' in self.shocks:
                shock_vec += self.shocks['r_star']
            if 'dr_star' in self.shocks:
                shock_vec += self.shocks['dr_star']
            
            try:
                result_dict = final_solver.solve_nonlinear(shock_vec)
                meta_convergence = {'success': True, 'iterations': 'N/A'} # TODO get from solver
                # If solver returns comprehensive dict, extract meta
            except Exception as e:
                # User-friendly Error Reporting
                print(f"Failed to find a stable equilibrium.")
                print(f"Solver Error: {str(e)}")
                raise e # Re-raise to stop execution or return Partial result?
                
        else:
            # Linear Solve
            # Map shocks
            if 'monetary' in self.shocks:
                result_dict = final_solver.solve_monetary_shock(self.shocks['monetary'])
            elif 'fiscal' in self.shocks:
                # SOESolver might allow this, or we fallback
                 raise NotImplementedError("Fiscal linear shock not yet fully mapped in Facade.")
            elif is_soe and 'r_star' in self.shocks:
                result_dict = final_solver.solve_open_economy_shock(self.shocks['r_star'])
            else:
                 # Default generic monetary shock if nothing else matches
                 if len(self.shocks) > 0:
                    val = next(iter(self.shocks.values()))
                    result_dict = final_solver.solve_monetary_shock(val)
                 else:
                    raise ValueError("No shocks defined for the linear solver.")

        # Capture Shocks in Metadata for Reproducibility
        meta_data = {
            'nonlinear': nonlinear, 
            'zlb': zlb,
            'shocks': {k: v.tolist() for k,v in self.shocks.items()}
        }
        
        return MonadResult(result_dict, self.config_params, meta=meta_data)


class Study:
    """
    The Laboratory.
    Manages multiple Monad instances for comparison.
    """
    def __init__(self, name: str = "New Study"):
        self.name = name
        self.cases: Dict[str, Monad] = {}
        self.results: Dict[str, MonadResult] = {}
        
    def add(self, label: str, model: Union[Monad, str]):
        """
        Add a case to the study.
        Args:
            label: Human-readable label (e.g. "Japan ZLB").
            model: Monad instance OR preset name string.
        """
        if isinstance(model, str):
            self.cases[label] = Monad(model)
        else:
            self.cases[label] = model
        return self
        
    def run(self, **solve_kwargs):
        """
        Run all cases.
        """
        print(f"--- Study: {self.name} ---")
        for label, model in self.cases.items():
            print(f"Running Case: {label}")
            # Ensure model has shocks defined. If not, maybe we should share shocks?
            # For now, assume models are pre-configured or 'shock' was called on them.
            if not model.shocks:
                print(f"Warning: Case '{label}' has no shocks defined.")
            
            self.results[label] = model.solve(**solve_kwargs)
        return self

    def plot(self, variables: Union[str, List[str]] = None, save_path: str = None):
        """
        Comparative Plot.
        """
        if variables is None:
            variables = ['Y', 'pi', 'r']
            
        if isinstance(variables, str):
            variables = [variables]
            
        n_vars = len(variables)
        cols = min(n_vars, 2)
        rows = (n_vars + 1) // 2
        
        fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
        if n_vars == 1: axes = [axes]
        axes = np.array(axes).flatten()
        
        # Colors for cases
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.results)))
        
        for i, var in enumerate(variables):
            ax = axes[i]
            for j, (label, res) in enumerate(self.results.items()):
                if var in res.data:
                    ax.plot(res.data[var], label=label, color=colors[j], linewidth=2)
            
            ax.set_title(var)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
            if i == 0:
                ax.legend()
                
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"[Study] Figure saved to {save_path}")
        else:
            plt.show()

    def sweep(self, param: str, values: List[float], baseline_model: Monad, shock_name='monetary', **solve_kwargs):
        """
        Conduct a parameter sweep experiment.
        Creates implied cases for each parameter value.
        """
        self.cases = {} # Reset cases
        self.results = {}
        
        base_shocks = baseline_model.shocks
        
        for v in values:
            label = f"{param}={v:.2f}"
            # Clone model (simplistic clone)
            # Re-init fresh Monad with same base config
            # Note: This is expensive if we re-run engine every time. 
            # Ideally we reuse engine if param is solver-only (like phi_pi).
            # If param is structural (kappa, beta), we MUST re-run engine.
            
            m = Monad(baseline_model.config_params.copy())
            m.setup(**{param: v})
            m.shocks = base_shocks
            self.add(label, m)
            
        return self.run(**solve_kwargs)
