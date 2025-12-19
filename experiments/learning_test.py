import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.model import MonadModel
from monad.policy import Rules

def main():
    print("=== Superset Strategy / Learning Engine Verification ===")
    
    # 0. Mock Run
    MonadModel.run = lambda self: {'status': 'mock_ok'}

    # 1. Setup Policy with Learning
    # Scenario: Taylor Rule where central bank learns r_star
    policy = Rules.InertialTaylor(name="cb")
    
    print("\n[Action] Configuring Policy.learns('r_star', gain=0.1)...")
    policy.learns("r_star", gain=0.1)
    
    # 2. Attach to Model
    m = MonadModel("dummy")
    m.attach(policy)
    
    # 3. Verification Points
    # A. Variable Injection
    # Expect: "cb_r_star_belief" in registered variables
    print("\n[Audit] Variable Injection:")
    injected_vars = [k for k in policy.auxiliary_vars.keys()]
    print(f"Vars: {injected_vars}")
    assert "cb_r_star_belief" in injected_vars, "Failed to inject belief variable"
    
    # B. Equation Injection (Default Dormant)
    # Expect: "eq_cb_r_star_belief" exists and equals "x = x(-1)" (Dormant)
    print("\n[Audit] Default Equation (Dormant):")
    eqs = m.injected_equations
    target_eq_key = "eq_cb_r_star_belief"
    assert target_eq_key in eqs, "Equation not injected"
    
    eq_content = eqs[target_eq_key]
    print(f"Metric: {eq_content}")
    # Very basic check for dormant structure "x = x(-1)"
    assert "(-1)" in eq_content and "+" not in eq_content.split("=")[1], \
           "Default equation is NOT dormant! It looks active."

    # C. Strategy Inspection
    # Check that Active strategy is stored internally (even if not active yet)
    print("\n[Audit] Internal Regimes:")
    regimes = policy._learning_regimes["cb_r_star_belief"]
    print(f"Active: {regimes['active']}")
    print(f"Dormant: {regimes['dormant']}")
    
    assert "0.1" in regimes['active'], "Active strategy missing gain parameter"
    
    print("\n=> Superset Strategy Verification MATCHED expectations.")

if __name__ == "__main__":
    main()
