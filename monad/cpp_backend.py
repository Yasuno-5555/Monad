"""
C++ Backend for Monad Engine.
Provides direct Python access to C++ SSJ solvers when available.
Falls back to CSV-based approach if C++ module is not built.
"""

import numpy as np
from scipy.linalg import toeplitz
import scipy.sparse
import os

# Try to import C++ module
_CPP_AVAILABLE = False
try:
    from . import monad_core as _cpp
    _CPP_AVAILABLE = True
except ImportError:
    _cpp = None


class CppBackend:
    """
    C++ Backend that directly calls pybind11-exposed functions.
    Computes steady state, Jacobians, and solves GE transitions.
    """
    
    def __init__(self, T=50, model_type="two_asset", params=None):
        if not _CPP_AVAILABLE:
            raise ImportError("C++ monad_core module not available. Build with: python setup_pybind.py build_ext --inplace")
        
        self.T = T
        self.model_type = model_type
        self.params = params or {}
        
        # Default parameters
        self._grid_params = {
            'Nm': 50, 'm_min': -2.0, 'm_max': 50.0, 'm_curv': 3.0,
            'Na': 40, 'a_min': 0.0, 'a_max': 100.0, 'a_curv': 2.0,
        }
        
        self._model_params = {
            'beta': 0.97,
            'r_m': 0.01,
            'r_a': 0.05,
            'chi': 20.0,
            'sigma': 2.0,
            'tax_lambda': 0.9,
            'tax_tau': 0.15,
            'tax_transfer': 0.05,
        }
        
        # Income process (default 2-state)
        self._income = {
            'z_grid': [0.8, 1.2],
            'Pi_flat': [0.9, 0.1, 0.1, 0.9],
        }
        
        # Override with user params
        self._model_params.update(self.params)
        
        # For RANK: single income state
        if model_type == "one_asset":
            self._income = {
                'z_grid': [1.0],
                'Pi_flat': [1.0],
            }
        
        # Computed results
        self._ss_result = None
        self._jacobians = None
        self.J_C_r = None
        self.J_C_y = None
    
    def solve_steady_state(self):
        """Solve for steady state policy and distribution."""
        if self.model_type == "one_asset" or self.model_type == "rank":
            # For RANK, force degenerate Z grid
            if self.model_type == "rank":
                z_grid = [1.0] # Single state, no variance
                Pi_flat = [1.0]
            else:
                z_grid = self._income['z_grid']
                Pi_flat = self._income['Pi_flat']

            self._ss_result = _cpp.solve_one_asset_steady_state(
                # Grid
                self._grid_params['Nm'], self._grid_params['m_min'], 
                self._grid_params['m_max'], self._grid_params['m_curv'],
                # Income
                z_grid, Pi_flat,
                # Params
                self._model_params['beta'], self._model_params['r_m'],
                self._model_params['sigma'],
                self._model_params['tax_lambda'], self._model_params['tax_tau'],
                self._model_params['tax_transfer'],
            )
            return self._ss_result

        self._ss_result = _cpp.solve_hank_steady_state(
            # Grid
            self._grid_params['Nm'], self._grid_params['m_min'], 
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            # Income
            self._income['z_grid'], self._income['Pi_flat'],
            # Params
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
        )
        return self._ss_result
    
    def compute_jacobians(self, T=None):
        """Compute block Jacobians J_C_rm and J_C_w."""
        if T is None: T = self.T
        if self._ss_result is None: self.solve_steady_state()
        
        self._jacobians = _cpp.compute_jacobians(
            # Grid
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            # Income
            self._income['z_grid'], self._income['Pi_flat'],
            # Params
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            # Steady state
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'],
            self._ss_result['E_Vm'], self._ss_result['E_V'],
            # Horizon
            T,
        )
        
        self.J_C_r = np.array(self._jacobians['J_C_rm'])
        self.J_C_y = np.array(self._jacobians['J_C_w'])
        return self._jacobians

    # --- New API Extensions ---

    def get_transition_matrix(self):
        """Export sparse transition matrix Lambda as scipy.sparse.coo_matrix."""
        if self._ss_result is None: self.solve_steady_state()
        
        rows, cols, data, size = _cpp.get_transition_matrix(
            # Grid
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            # Income
            self._income['z_grid'], self._income['Pi_flat'],
            # Params
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'], self._grid_params['m_min'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            # Policy
            self._ss_result['m_pol'], self._ss_result['a_pol']
        )
        
        return scipy.sparse.coo_matrix((data, (rows, cols)), shape=(size, size))

    def probe_policy(self, m_val, a_val, z_idx=0):
        """Interpolate policy function at specific state (m, a, z)."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.probe_policy(
            self._ss_result['c_pol'],
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            len(self._income['z_grid']),
            m_val, a_val, z_idx
        )

    def solve_ge_zlb(self, dr_star, forced_binding=None):
        """Solve GE with Zero Lower Bound constraint."""
        if self._jacobians is None: self.compute_jacobians(len(dr_star))
        if forced_binding is None: forced_binding = []
        
        return _cpp.solve_ge_zlb(
            self._jacobians['J_C_rm'], self._jacobians['J_C_w'],
            dr_star,
            self._model_params['beta'], 
            self._model_params.get('kappa', 0.1),
            self._model_params.get('phi_pi', 1.5),
            forced_binding
        )

    def analyze_inequality(self, dr_path, dZ_path):
        """Analyze inequality (winners/losers) for given paths."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.analyze_inequality(
            # Standard args...
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            self._income['z_grid'], self._income['Pi_flat'],
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'],
            self._ss_result['E_Vm'], self._ss_result['E_V'],
            # Paths
            dr_path, dZ_path
        )

    def solve_fiscal_shock(self, dG_path, dTrans_path):
        """Solve for Fiscal Shock (Gov Spending or Transfer)."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.solve_fiscal_shock(
            # Standard args...
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            self._income['z_grid'], self._income['Pi_flat'],
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'],
            self._ss_result['E_Vm'], self._ss_result['E_V'],
            # Paths
            dG_path, dTrans_path
        )

    def decompose_multiplier(self, dY_path, dTrans_path, dr_path):
        """Decompose dY into Direct (Partial Eq) and Indirect (General Eq) effects on C."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.decompose_multiplier(
            # Standard args...
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            self._income['z_grid'], self._income['Pi_flat'],
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'],
            self._ss_result['E_Vm'], self._ss_result['E_V'],
            # Inputs
            dY_path, dTrans_path, dr_path
        )

    def solve_optimal_policy(self, lambda_y, dr_star):
        """Solve for optimal monetary policy (LQR)."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.solve_optimal_policy(
            # Standard args...
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            self._income['z_grid'], self._income['Pi_flat'],
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'],
            self._ss_result['E_Vm'], self._ss_result['E_V'],
            # Opt Params
            lambda_y, dr_star
        )

    def compute_mpc_distribution(self):
        """Compute aggregate and distributional MPC statistics."""
        if self._ss_result is None: self.solve_steady_state()
        
        return _cpp.compute_mpc_distribution(
            # Standard args...
            self._grid_params['Nm'], self._grid_params['m_min'],
            self._grid_params['m_max'], self._grid_params['m_curv'],
            self._grid_params['Na'], self._grid_params['a_min'],
            self._grid_params['a_max'], self._grid_params['a_curv'],
            self._income['z_grid'], self._income['Pi_flat'],
            self._model_params['beta'], self._model_params['r_m'],
            self._model_params['r_a'], self._model_params['chi'],
            self._model_params['sigma'],
            self._model_params['tax_lambda'], self._model_params['tax_tau'],
            self._model_params['tax_transfer'],
            self._ss_result['c_pol'], self._ss_result['m_pol'],
            self._ss_result['a_pol'], self._ss_result['value'],
            self._ss_result['adjust_flag'], self._ss_result['distribution'] 
        )


class GPUBackend:
    """
    CSV-based backend that loads pre-computed Jacobians from GPU export.
    Fallback when C++ module is not available.
    """
    
    def __init__(self, path_R, path_Z, T=50):
        self.T = T
        self.J_C_r = self._load_jacobian(path_R, 'dC')
        self.J_C_y = self._load_jacobian(path_Z, 'dC')
        
    def _load_jacobian(self, path, col_name):
        import pandas as pd
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Run C++ Engine first.")
        df = pd.read_csv(path)
        vec = df[col_name].values
        if len(vec) < self.T: vec = np.pad(vec, (0, self.T - len(vec)))
        else: vec = vec[:self.T]
        return toeplitz(vec, np.zeros(self.T))


def get_backend(model_type="two_asset", T=50, params=None, fallback_paths=None):
    """
    Factory function to get the appropriate backend.
    """
    if _CPP_AVAILABLE:
        try:
            backend = CppBackend(T=T, model_type=model_type, params=params)
            backend.compute_jacobians()
            return backend
        except Exception as e:
            print(f"[WARN] C++ backend failed: {e}, falling back to CSV")
    
    if fallback_paths is None:
        fallback_paths = {'path_R': 'gpu_jacobian_R.csv', 'path_Z': 'gpu_jacobian_Z.csv'}
    return GPUBackend(fallback_paths['path_R'], fallback_paths['path_Z'], T)

def is_cpp_available():
    return _CPP_AVAILABLE
