import sys
import os
import numpy as np

# Add root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.monad_core import solve_hank_steady_state
from monad.welfare import WelfareAnalyzer, WelfareResult
from monad.regime import RegimeManager, ZLB_Regime
from monad.model import MonadModel

def test_welfare_api():
    print("\n[Test] Welfare API")
    # Simulate a result dictionary
    res_base = {
        'value': np.ones(100) * 10.0,
        'distribution': np.ones(100) / 100.0,
    }
    
    res_alt = {
        'value': np.ones(100) * 10.05, # Slightly better
        'distribution': np.ones(100) / 100.0,
    }
    
    w_base = WelfareResult(res_base, params={'sigma': 2.0})
    w_alt = WelfareResult(res_alt, params={'sigma': 2.0})
    
    # Base Welfare
    print(f"Base Welfare: {w_base.W}")
    
    # Compare
    # Formula check: (10.05 / 10.0)^(1/-1) - 1 => (1.005)^-1 - 1 = 0.995 - 1 = -0.005
    # Wait, W is negative for sigma=2 => u(c) = c^-1 / -1
    # If W increases (becomes less negative), that's good.
    # But 10.05 > 10.0 is WRONG for sigma=2 because Utility is usually negative.
    # Let's fix example:
    
    # u(c) = -1/c. if c=1, u=-1. W = -1/(1-beta) approx.
    # c=2 => u=-0.5. W is higher (less negative).
    
    res_base['value'] = -100.0 * np.ones(100)
    res_alt['value']  = -50.0 * np.ones(100) # Much better
    
    w_base = WelfareResult(res_base, params={'sigma': 2.0})
    w_alt = WelfareResult(res_alt, params={'sigma': 2.0})

    # lambda = (W_alt/W_base)^(-1) - 1 = (-50/-100)^-1 - 1 = (0.5)^-1 - 1 = 2-1 = 1 (100% gain)
    cev = w_alt.compare(w_base)
    print(f"CEV Test: {cev*100:.2f}% (Expected ~100%)")
    assert abs(cev - 1.0) < 1e-4

def test_regime_object():
    print("\n[Test] Regime Object")
    mgr = RegimeManager()
    mgr.add(ZLB_Regime())
    
    path_i = np.array([0.05, 0.02, -0.01, -0.02, 0.01])
    
    constrained, is_binding = mgr.evaluate(path_i, "i")
    
    print("Input: ", path_i)
    print("Output:", constrained)
    print("Bind:  ", is_binding)
    
    expected = np.array([0.05, 0.02, 0.0, 0.0, 0.01])
    assert np.allclose(constrained, expected)

def test_model_objects_mock():
    print("\n[Test] Endogenous Objects (Mock)")
    
    # Mock Model
    m = MonadModel("dummy_path")
    
    # 1. Mutate parameter
    m.object("policy_rule").mutate(phi_pi=2.0)
    
    # 2. Toggle Solver feature
    m.object("solver").toggle("zlb")
    
    print("Overrides:", m.overrides)
    
    assert m.overrides['phi_pi'] == 2.0
    assert m.overrides['solver_settings.zlb'] == True

if __name__ == "__main__":
    test_welfare_api()
    test_regime_object()
    test_model_objects_mock()
