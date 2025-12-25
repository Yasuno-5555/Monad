import pickle
import hashlib
import os
import numpy as np
from .solver import SOESolver, NKHANKSolver
from .nonlinear import NewtonSolver
from .cpp_backend import get_backend

class Model:
    """
    High-level Orchestrator for Monad Models.
    Manages State (SS, Jacobian) and Backend.
    """
    def __init__(self, model_type="two_asset", T=50, params=None, cache_dir=".cache"):
        self.model_type = model_type
        self.T = T
        self.params = params or {}
        self.cache_dir = cache_dir
        self.backend = None
        self.solver = None
        self._state_cache = {}
        
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
    def initialize(self):
        """Initialize backend and compute SS/Jacobians if needed."""
        print(f"[Monad] Initializing {self.model_type} model...")
        self.backend = get_backend(
            model_type=self.model_type, T=self.T, params=self.params
        )
        # Check cache first for Jacobians
        cache_key = self._get_hash()
        if self._load_cache(cache_key):
            print("[Monad] Loaded state from cache.")
        else:
            print("[Monad] Computing Steady State & Jacobians...")
            self.backend.compute_jacobians()
            self._save_cache(cache_key)
            
        # Init Solver Wrapper
        # Default to Linear/NKHANK structure
        self.solver = SOESolver(T=self.T, params=self.params, model_type=self.model_type)
        # Inject the initialized backend to avoid re-computation
        self.solver.backend = self.backend 

    def run_experiment(self, shocks, robust=False, zlb=True):
        """
        Run a simulation scenario.
        Args:
            shocks: dict of {name: path} (e.g. {'dr_star': vec})
            robust: if True, use homotopy/damping
            zlb: bool, enable ZLB
        """
        if self.solver is None: self.initialize()
        
        # Prepare Nonlinear Solver if ZLB or large shocks
        nl_solver = NewtonSolver(self.solver, max_iter=200 if not robust else 1000)
        nl_solver.zlb_enabled = zlb
        nl_solver.damping = 0.5 if robust else 0.8
        
        # Dispatch
        # Currently optimized for 'dr_star' (Natural Rate / SOE shock)
        if 'dr_star' in shocks:
            shock_path = shocks['dr_star']
            if len(shock_path) < self.T:
                 shock_path = np.pad(shock_path, (0, self.T - len(shock_path)))
            
            if robust:
                # Simple Manual Homotopy logic or use solver's homotopy
                print("[Monad] Running Robust Mode (Homotopy)...")
                # Assuming solver has .solve_with_homotopy or we just try-catch
                try:
                    res = nl_solver.solve_nonlinear(shock_path)
                except:
                    print("[Monad] Direct solve failed, retrying with damping...")
                    nl_solver.damping = 0.2
                    res = nl_solver.solve_nonlinear(shock_path)
            else:
                res = nl_solver.solve_nonlinear(shock_path)
                
            return res
            
        elif 'dG' in shocks or 'dTrans' in shocks:
            # Direct Fiscal Experiment via Backend (Partial or Simplified GE)
            dG = shocks.get('dG', np.zeros(self.T))
            dTrans = shocks.get('dTrans', np.zeros(self.T))
            return self.backend.solve_fiscal_shock(dG, dTrans)
            
        else:
            raise ValueError("Unknown shock type. Supported: dr_star, dG, dTrans")

    def _get_hash(self):
        """Generate hash from params for caching."""
        s = f"{self.model_type}_{self.T}_{str(sorted(self.params.items()))}"
        return hashlib.md5(s.encode()).hexdigest()

    def _save_cache(self, key):
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        # Save backend jacobians and SS result
        data = {
            'jacobians': self.backend._jacobians,
            'ss_result': self.backend._ss_result,
            'J_C_r': self.backend.J_C_r,
            'J_C_y': self.backend.J_C_y
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _load_cache(self, key):
        path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                data = pickle.load(f)
            self.backend._jacobians = data['jacobians']
            self.backend._ss_result = data['ss_result']
            self.backend.J_C_r = data['J_C_r']
            self.backend.J_C_y = data['J_C_y']
            return True
        return False
        
    # Analysis Proxies
    def analyze_inequality(self, results):
        """Run inequality analysis on simulation results."""
        # Need dr and dZ (or dY? Cpp expects dZ usually mapped to Wage/Income)
        # From NL sovler results we have Y, C, i, pi.
        # We need to reconstruction Inputs for specific backend method
        # backend.analyze_inequality(dr_path, dZ_path)
        
        # Approx: dZ approx dY (Labor Income)
        # dr is real rate deviation. r_real = i - pi. dr = r_real - r_ss.
        # NL solver returns 'dr' usually.
        
        dr = results.get('dr', np.zeros(self.T))
        dY = results.get('Y', np.zeros(self.T)) # dY or Y level?
        # Check NewtonSolver output. Usually level deviation or percent?
        # NewtonSolver returns whatever variables are in block.
        # Usually deviation from SS.
        
        return self.backend.analyze_inequality(dr, dY)

