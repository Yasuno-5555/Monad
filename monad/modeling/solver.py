import sympy
import numpy as np
from scipy.optimize import root
from typing import Dict, List, Tuple
from .schema import ModelSpec

class GenericStaticSolver:
    """
    Parses a ModelSpec, compiles equations using sympy/numpy, 
    and solves for static equilibrium using scipy.optimize.root.
    """
    def __init__(self, model_spec: ModelSpec):
        self.spec = model_spec
        self.var_names = list(self.spec.variables.keys())
        self.param_dict = {k: v.value for k, v in self.spec.parameters.items()}
        
        self.residual_func = None
        self.jac_func = None
        
        self._compile()
        
    def _compile(self):
        """Compile string equations into callable numpy functions."""
        print(f"[Monad] Compiling {self.spec.name}...")
        
        # 1. Define Sympy Symbols
        sym_vars = {name: sympy.Symbol(name) for name in self.var_names}
        sym_params = {name: sympy.Symbol(name) for name in self.param_dict.keys()}
        
        all_syms = {**sym_vars, **sym_params}
        
        # 2. Parse Equations
        exprs = []
        for eq in self.spec.equations:
            # Parse string "Y - (C + I + G)" -> sympy expr
            try:
                # Use standard sympy parsing
                expr = sympy.parse_expr(eq.resid, local_dict=all_syms)
                exprs.append(expr)
                print(f"  [Eq] {eq.resid} -> {expr}")
            except Exception as e:
                raise ValueError(f"Failed to parse equation: '{eq.resid}'. Error: {e}")
        
        if len(exprs) != len(self.var_names):
            raise ValueError(f"Dim mismatch: {len(exprs)} equations for {len(self.var_names)} variables.")
            
        # 3. Substitute Parameters (Static Solver treats params as constants)
        # We can either sub now (faster solve) or keep purely symbolic (cleaner for sensitivty).
        # Let's sub now for the 'solver' instance.
        # But wait, we might want to change params later?
        # Better design: Compile function f(vars, params).
        # But for 'root' we need f(vars).
        # Let's sub values for now.
        
        exprs_subbed = [e.subs([(sym_params[k], v) for k, v in self.param_dict.items()]) for e in exprs]
        
        # 4. Compile to Lambda via lambdify
        # Input order: [var1, var2, ...]
        input_vars = [sym_vars[name] for name in self.var_names]
        
        # f_np returns list of residuals
        self.residual_func = sympy.lambdify([input_vars], exprs_subbed, modules='numpy')
        
        # 5. Symbolic Jacobian (Bonus: Analytical Derivatives!)
        jac_matrix = sympy.Matrix(exprs_subbed).jacobian(input_vars)
        self.jac_func = sympy.lambdify([input_vars], jac_matrix, modules='numpy')
        
        print("[Monad] Compilation complete.")

    def solve(self, verbose=True):
        """Solve the system using C++ GenericNewtonSolver."""
        try:
            from monad import monad_core
        except ImportError:
            print("[WARN] C++ backend not available. Install with `python setup_pybind.py build_ext --inplace`")
            return None

        # Initial Guess
        x0 = np.array([self.spec.variables[n].guess for n in self.var_names], dtype=np.float64)
        
        if verbose:
            print(f"Solving {self.spec.name} via C++ Newton Engine...")
            print(f"Variables: {self.var_names}")
            print(f"Guess: {x0}")
            
        # Wrapper for C++: generic_solver expects function that takes vector and returns vector/list
        def fun(x):
            # x is Eigen::VectorXd which pybind11 converts to numpy array
            # We pass it to our compiled residual_func
            return self.residual_func(x)
            
        # Solve via C++
        # solve_static_model(function, guess, max_iter, tol, damping)
        try:
            x_sol = monad_core.solve_static_model(fun, x0, 100, 1e-6, 1.0)
        except Exception as e:
            print(f"[FAIL] C++ Solver Error: {e}")
            return None
            
        # Result dict
        res = {name: float(val) for name, val in zip(self.var_names, x_sol)}
        
        if verbose:
            print("[SUCCESS] Equilibrium found (C++).")
            for k, v in res.items():
                print(f"  {k} = {v:.4f}")
        
        return res
