import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.nonlinear import PiecewiseSolver
from monad.regime import registry, regime

# --- Mock Classes for Dummy Linear Model ---
# Model: y_t = rho * y_{t-1} + e_t
# T = 10
# Normal rho = 0.9
# Crisis rho = 0.5 (if y_{t-1} > 1.0)

class MockBlock:
    def __init__(self):
        self.params = {'phi_pi': 1.5}
    def get_phillips_curve(self): return np.eye(10) # Dummy
    def get_taylor_rule(self): return np.eye(10) # Dummy
    def get_fisher_equation(self): return np.eye(10) # Dummy

class MockBackend:
    def __init__(self):
        # Dummy Jacobians needed for PiecewiseSolver internals (legacy)
        # But we override _compute_jacobian_for_regime anyway for this test
        self.J_C_r = np.eye(10)
        self.J_C_y = np.eye(10)

class MockModel:
    def __init__(self):
        self.T = 10
        self.block = MockBlock()
        self.backend = MockBackend()
        
class TestSolver(PiecewiseSolver):
    """Subclass to simulate our specific Dummy Linear Model Logic"""
    
    def _derive_paths(self, Y, shock):
        # We only care about Y for regime eval
        return {'Y': Y}
        
    def _compute_jacobian_for_regime(self, regime_name):
        # Build H where H * dY = -Res
        # System: y_t - rho * y_{t-1} = e_t
        # Res = (rho * y_{t-1} + e_t) - y_t
        # dRes/dy_t = -1
        # dRes/dy_{t-1} = rho
        
        rho = 0.9 if regime_name == "normal" else 0.5
        
        # H matrix structure:
        # Col t (y_t): -1
        # Col t-1 (y_{t-1}): rho
        
        H = -1.0 * np.eye(self.T)
        
        # Add rho * y_{t-1} effect
        # Rows 1..T, Cols 0..T-1
        row_idx, col_idx = np.diag_indices(self.T)
        # Shift for lag
        # y_t depends on y_{t-1}
        # eq t: y_t - rho * y_{t-1} = 0
        # d(Eq)/dy_{t-1} = -rho. Wait, I defined Res differently above.
        
        # Let's stick to PiecewiseSolver convention: H * dY = -Residual
        # Newton: Y_new = Y_old - H^-1 * Res
        # So H should be d(Res)/dY
        
        # Res_t = (rho * Y_{t-1} + e_t) - Y_t
        # dRes_t / dY_t = -1
        # dRes_t / dY_{t-1} = rho
        
        H = -1.0 * np.eye(self.T)
        for t in range(1, self.T):
            H[t, t-1] = rho
            
        return H

    def _assemble_system(self, regime_path, Y, shock, paths):
        # Full bypass of parent logic for Mock
        # We don't call super() because parent tries to read paths['i'], paths['pi'] which we don't populate
        
        # 1. H assembly
        # Custom "Patchwork" H for this mock model
        H = np.zeros((self.T, self.T))
        
        # Base H for "normal"
        H_normal = self._compute_jacobian_for_regime("normal")
        H_crisis = self._compute_jacobian_for_regime("Crisis")
        
        unique_regimes = np.unique(regime_path)
        
        # Apply rows
        for t in range(self.T):
            r_name = regime_path[t]
            if r_name == "Crisis":
                H[t, :] = H_crisis[t, :]
            else:
                H[t, :] = H_normal[t, :]
        
        # 2. Residual
        # Res_t = rho(regime) * Y_{t-1} + e_t - Y_t
        Res = np.zeros(self.T)
        e_t = shock 
        
        for t in range(self.T):
            rho = 0.9 # Default
            if regime_path[t] == "Crisis":
                rho = 0.5
            
            y_lag = Y[t-1] if t > 0 else 0.0
            Res[t] = (rho * y_lag + e_t[t]) - Y[t]
            
        return H, Res

def test_regime_consistency():
    print("--- Test: Regime Consistency (Dummy Linear Model) ---")
    
    # 1. Define Regime
    # Crisis if y_{t-1} > 1.0 (Note: lagged dependency!)
    
    @regime(priority=1)
    def Crisis(model, paths):
        Y = paths['Y']
        # Condition: Y_{t-1} > 1.0
        # Shift Y right to get Y_{t-1}
        Y_lag = np.roll(Y, 1)
        Y_lag[0] = 0.0 
        return (Y_lag > 1.0) # Boolean mask

    # 2. Setup Solver
    model = MockModel()
    solver = TestSolver(model, damping=1.0)
    
    # 3. Shock
    # Large shock at t=0 to trigger crisis dynamics
    shock = np.zeros(10)
    shock[0] = 2.0 # Initial impulse
    
    # 4. Solve
    # Expected Path:
    # t=0: y=2.0 (Normal rho=0.9, but e_0=2.0 dominating) -> Regime Normal (y_{-1}=0)
    # t=1: y_{-1}=2.0 > 1.0 => CRISIS. rho=0.5. y_1 = 0.5 * 2.0 = 1.0.
    # t=2: y_{-1}=1.0 (not > 1.0) => NORMAL. rho=0.9. y_2 = 0.9 * 1.0 = 0.9.
    # t=3: y_{-1}=0.9 => NORMAL. y_3 = 0.9 * 0.9 = 0.81...
    
    results = solver.solve(shock)
    Y_sol = results['Y']
    Regimes_sol = results['regime_path']
    
    print("\n[Result Path]")
    print(f"Y: {Y_sol}")
    print(f"Regimes: {Regimes_sol}")
    
    # Verification
    assert Regimes_sol[1] == "Crisis", "t=1 should be Crisis (y_0=2.0 > 1.0)"
    assert Regimes_sol[2] == "normal", "t=2 should be Normal (y_1=1.0 not > 1.0)"
    
    expected_y1 = 0.5 * 2.0
    assert np.isclose(Y_sol[1], expected_y1), f"Y[1] should be {expected_y1} (rho=0.5), got {Y_sol[1]}"
    
    print("Test Logic Passed!")

if __name__ == "__main__":
    test_regime_consistency()
