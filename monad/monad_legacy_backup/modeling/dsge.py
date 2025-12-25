import sympy
import numpy as np
import re
from typing import Dict, List, Optional
from .schema import ModelSpec, VariableSpec
from monad import monad_core

class DSGEStaticSolver:
    """
    Solves a dynamic model (DSGE) over a finite horizon T as a stacked static system.
    Unrolls equations: F(X_{t-1}, X_t, X_{t+1}) = 0 for t=0...T-1.
    """
    def __init__(self, spec: ModelSpec, T: int = 20):
        self.spec = spec
        self.T = T
        self.var_names = list(self.spec.variables.keys())
        self.N = len(self.var_names)
        self.param_dict = {k: v.value for k, v in self.spec.parameters.items()}
        
        # Boundary Conditions (Defaults to 0 or spec guess)
        self.initial_state = {v: 0.0 for v in self.var_names}
        self.terminal_state = {v: 0.0 for v in self.var_names}
        
        self.residual_func = None
        self.jac_func = None
        
        self._is_compiled = False

    def set_initial_state(self, state: Dict[str, float]):
        self.initial_state.update(state)

    def set_terminal_state(self, state: Dict[str, float]):
        self.terminal_state.update(state)

    def _parse_time_index(self, expr_str: str, t: int):
        """
        Replace x[t], x[t-1], x[t+1] with specific symbols for time,
        handling boundaries.
        """
        # Regex to find variable access: name[t...]
        # Pattern: (\w+)\[(t|t-1|t\+1)\]
        # We need to robustly replace.
        
        # Strategy: Pre-process the string for THIS 't'.
        # 1. Replace 'x[t]' with Symbol('x_t')
        # 2. Replace 'x[t-1]' with Symbol('x_{t-1}') OR Value (if t=0)
        # 3. Replace 'x[t+1]' with Symbol('x_{t+1}') OR Value (if t=T-1)
        
        # Actually easier: Work with "t" as a variable in parsing logic? No, simple string sub is safer for now.
        
        processed_eq = expr_str
        
        for var in self.var_names:
            # 1. Current: var[t] -> var_{t}
            # Note: Do explicit replacements to avoid partial matches
            processed_eq = processed_eq.replace(f"{var}[t]", f"{var}_{t}")
            
            # 2. Lag: var[t-1]
            if t == 0:
                # Boundary: Initial Value
                val = self.initial_state.get(var, 0.0)
                processed_eq = processed_eq.replace(f"{var}[t-1]", str(val))
            else:
                processed_eq = processed_eq.replace(f"{var}[t-1]", f"{var}_{t-1}")
                
            # 3. Lead: var[t+1]
            if t == self.T - 1:
                # Boundary: Terminal Value
                val = self.terminal_state.get(var, 0.0)
                processed_eq = processed_eq.replace(f"{var}[t+1]", str(val))
            else:
                processed_eq = processed_eq.replace(f"{var}[t+1]", f"{var}_{t+1}")
                
        return processed_eq

    def compile(self):
        """Unroll equations and compile to numpy/C++ compatible function."""
        print(f"[Monad-DSGE] Compiling stacked system (T={self.T}, Vars={self.N*self.T})...")
        
        all_exprs = []
        
        # Define ALL symbolic variables for the unified system: x_0, ... x_{T-1}
        # Order: [y_0, pi_0, i_0, y_1, pi_1, i_1, ...]
        self.flat_vars = []
        
        sym_map = {} # Str -> Symbol
        
        # Create symbols
        for t in range(self.T):
            for var in self.var_names:
                name = f"{var}_{t}"
                s = sympy.Symbol(name)
                sym_map[name] = s
                self.flat_vars.append(s)
                
        # Define Params
        sym_params = {name: sympy.Symbol(name) for name in self.param_dict.keys()}
        
        # Unroll Equations
        for t in range(self.T):
            for eq_spec in self.spec.equations:
                # 1. Specialized string for time t
                eq_str = self._parse_time_index(eq_spec.resid, t)
                
                # 2. Parse into Sympy
                try:
                    # Combine symbol contexts
                    local_dict = {**sym_map, **sym_params}
                    
                    expr = sympy.parse_expr(eq_str, local_dict=local_dict)
                    
                    # 3. Substitute Params (Static)
                    expr = expr.subs([(sym_params[k], v) for k, v in self.param_dict.items()])
                    
                    all_exprs.append(expr)
                    
                except Exception as e:
                    print(f"Error parsing Eq at t={t}: {eq_str}")
                    raise e
                    
        print(f"[Monad-DSGE] Generated {len(all_exprs)} stacked equations.")
        
        # Compile
        # Optimization: use 'cse=True' if possible, but standard lambdify is okay for T=20
        self.residual_func = sympy.lambdify([self.flat_vars], all_exprs, modules='numpy')
        
        self._is_compiled = True
        
    def solve(self, verbose=True):
        if not self._is_compiled:
            self.compile()
            
        # Initial Guess: Flat vector
        # Use simple steady-state guess or 0
        x0 = np.zeros(self.N * self.T)
        
        # Populate with variable guesses if provided in spec
        for t in range(self.T):
            for i, var in enumerate(self.var_names):
                guess = self.spec.variables[var].guess
                x0[t * self.N + i] = guess
        
        if verbose:
            print(f"Solving stacked system via C++ Newton...")
            
        # Wrapper
        def fun(x):
            try:
                # Returns list of values
                res = self.residual_func(x)
                # Convert list to array if needed (lambdify might return list)
                return np.array(res, dtype=np.float64)
            except Exception as e:
                print(f"Eval Error: {e}")
                raise e

        # Solve via C++
        # Using Robust Damping for large systems
        try:
            x_sol = monad_core.solve_static_model(fun, x0, 200, 1e-6, 0.5)
        except Exception as e:
            print(f"[FAIL] {e}")
            return None
            
        if verbose:
            print("[SUCCESS] DSGE Solution found.")
            
        # Unpack result into dict of arrays {var: [t0...T]}
        res_dict = {}
        for i, var in enumerate(self.var_names):
            # Extract every N-th element starting at i
            # Vector structure: [v0_t0, v1_t0... v0_t1...]
            # Indices for var i: i, i+N, i+2N...
            indices = [t * self.N + i for t in range(self.T)]
            res_dict[var] = np.array([x_sol[idx] for idx in indices])
            
        return res_dict
