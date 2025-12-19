import sys
import os
import traceback
sys.path.append(os.getcwd())

def test_debug():
    print("Start Debug")
    try:
        from monad.nonlinear import PiecewiseSolver
        print("Import Solver OK")
        
        # Mock objects
        class MockConfig:
             params = {'phi_pi': 1.5}
             def get_phillips_curve(self): import numpy as np; return np.eye(50)
             def get_fisher_equation(self): import numpy as np; return np.eye(50)
             def get_taylor_rule(self): import numpy as np; return np.eye(50)

        class MockBackend:
             J_C_r = None
             J_C_y = None
             def __init__(self):
                 import numpy as np
                 self.J_C_r = np.zeros((50,50))
                 self.J_C_y = np.zeros((50,50))

        class MockLinearSolver:
             T = 50
             block = MockConfig()
             backend = MockBackend()
             
        # Instantiate
        solver = PiecewiseSolver(MockLinearSolver())
        print("Instantiation OK")
        
        # Derive paths check with dict
        import numpy as np
        Y = np.zeros(50)
        shock = {"markup": 0.05}
        paths = solver._derive_paths(Y, shock)
        print("Derive Paths OK")
        print(f"Pi[0]: {paths['pi'][0]}")

    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_debug()
