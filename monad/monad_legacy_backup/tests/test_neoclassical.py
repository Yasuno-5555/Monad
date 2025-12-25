import sys
import os
import json
import sympy as sp

# Add parent dir to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from middleware.model_builder import ModelBuilder
from middleware.converter import convert_to_ir

def test_neoclassical_ir():
    mb = ModelBuilder(name="NeoclassicalGrowth")
    
    # Parameters
    beta = mb.add_parameter("beta", 0.96)
    gamma = mb.add_parameter("gamma", 1.0) # Log utility
    r = mb.add_parameter("r", 0.04)
    
    # Variables
    c = mb.add_variable("c", "control", "Consumption", (0.0, None))
    a = mb.add_variable("a", "state", "Assets", (0.0, 100.0))
    a_prime = mb.add_variable("a_prime", "control", "Next Period Assets", (0.0, None))
    
    # Grids
    mb.add_grid("a_grid", "log_spaced", 0.0, 100.0, 100, curvature=2.0)
    
    # EGM Equations
    # Euler: u'(c) = beta * (1+r) * u'(c_next)
    # Simple case: c = 1/u_marg implies u_marg = c^(-gamma)
    # This is just a structural test, not a solver test
    
    u_marg = c**(-gamma)
    rhs = beta * (1 + r) * u_marg # Simplified next period logic for IR test
    
    mb.add_egm_equation(u_marg, rhs)
    
    # Convert
    ir = convert_to_ir(mb)
    
    # Verify
    try:
        # Pydantic v2
        json_output = ir.model_dump_json(indent=2)
    except AttributeError:
        # Pydantic v1
        json_output = ir.json(indent=2)
    print("IR Generation Successful:")
    print(json_output)
    
    assert ir.name == "NeoclassicalGrowth"
    assert len(ir.grids) == 1
    assert ir.grids[0].kind == "log_spaced"
    assert ir.parameters['beta'] == 0.96
    
    print("\nTest PASSED")

if __name__ == "__main__":
    try:
        test_neoclassical_ir()
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
