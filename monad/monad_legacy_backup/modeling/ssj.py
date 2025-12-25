import sympy
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from .schema import ModelSpec

class LinearSSJBuilder:
    """
    Constructs Linear State Space Matrices (A, B, C) and Block Jacobians from a Model Spec.
    Result: 
        Explicit Part: A * dX_{t+1} + B * dX_t + C * dX_{t-1} + Sum(J_block * dBlockInputs) = 0
    """
    def __init__(self, spec: ModelSpec):
        self.spec = spec
        self.var_names = list(self.spec.variables.keys())
        self.N = len(self.var_names)
        self.param_dict = {k: v.value for k, v in self.spec.parameters.items()}
        
        # Block Management
        self.block_outputs = set()
        for b in self.spec.blocks.values():
            for out in b.outputs:
                self.block_outputs.add(out)
                
        # Symbolic placeholders
        self.sym_vars_t_minus_1 = []
        self.sym_vars_t = []
        self.sym_vars_t_plus_1 = []
        self.sym_params = {name: sympy.Symbol(name) for name in self.param_dict.keys()}
        
        self.equations = [] 
        
        self._is_compiled = False

    def _parse_equation(self, eq_str: str) -> sympy.Expr:
        local_dict = {**self.sym_params}
        processed_eq = eq_str
        
        for var in self.var_names:
            local_dict[f"{var}_tm1"] = sympy.Symbol(f"{var}_tm1")
            local_dict[f"{var}_t"]   = sympy.Symbol(f"{var}_t")
            local_dict[f"{var}_tp1"] = sympy.Symbol(f"{var}_tp1")
            
            processed_eq = processed_eq.replace(f"{var}[t-1]", f"{var}_tm1")
            processed_eq = processed_eq.replace(f"{var}[t+1]", f"{var}_tp1")
            processed_eq = processed_eq.replace(f"{var}[t]", f"{var}_t")
            
        try:
            expr = sympy.parse_expr(processed_eq, local_dict=local_dict)
            return expr
        except Exception as e:
            raise ValueError(f"Failed to parse equation: {eq_str}\nError: {e}")

    def compile(self):
        print(f"[Monad-SSJ] Compiling Linear Matrices for {self.spec.name}...")
        
        self.sym_vars_t_minus_1 = [sympy.Symbol(f"{v}_tm1") for v in self.var_names]
        self.sym_vars_t         = [sympy.Symbol(f"{v}_t") for v in self.var_names]
        self.sym_vars_t_plus_1  = [sympy.Symbol(f"{v}_tp1") for v in self.var_names]
        
        self.equations = []
        for eq in self.spec.equations:
            expr = self._parse_equation(eq.resid)
            self.equations.append(expr)
            
        F_vec = sympy.Matrix(self.equations)
        
        # Explicit Jacobians w.r.t all variables
        self.jac_A_sym = F_vec.jacobian(self.sym_vars_t_plus_1)
        self.jac_B_sym = F_vec.jacobian(self.sym_vars_t)
        self.jac_C_sym = F_vec.jacobian(self.sym_vars_t_minus_1)
        
        self._is_compiled = True
        print("[Monad-SSJ] Symbolic formulation complete.")

    def get_matrices(self, ss_dict: Dict[str, float]):
        if not self._is_compiled:
            self.compile()
            
        subs_list = []
        for pname, pval in self.param_dict.items():
            subs_list.append((self.sym_params[pname], pval))
            
        for i, var in enumerate(self.var_names):
            val = ss_dict.get(var, 0.0)
            subs_list.append((self.sym_vars_t_minus_1[i], val))
            subs_list.append((self.sym_vars_t[i],         val))
            subs_list.append((self.sym_vars_t_plus_1[i],  val))
            
        def eval_matrix(sym_mat):
            mat_sub = sym_mat.subs(subs_list)
            return np.array(mat_sub.tolist(), dtype=np.float64)
            
        A = eval_matrix(self.jac_A_sym)
        B = eval_matrix(self.jac_B_sym)
        C = eval_matrix(self.jac_C_sym)
        
        return A, B, C

    def get_block_jacobians(self, T: int = 50) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Mock retrieval of Block Jacobians.
        In a real HANK, this would call the C++ kernel.
        """
        block_jacs = {}
        for b_name, b_spec in self.spec.blocks.items():
            print(f"[SSJ] Retrieving Jacobian for Block: {b_name} ({b_spec.kernel})")
            
            # This is where we would call e.g. monad_core.TwoAssetSolver.compute_jacobians()
            # For now, return random DENSE matrices to simulate Heterogeneous block
            
            bj = {}
            for out_v in b_spec.outputs:
                for in_v in b_spec.inputs:
                    # Jacobian d(out)/d(in) is T x T
                    # For HANK, this is dense.
                    # Create a "fake" structure: Diag + some off-diagonal for persistence
                    J = np.eye(T) * 0.5 
                    if out_v == "C" and in_v == "r":
                        # C drops when r rises (intertemporal sub), so negative
                        J = -0.1 * np.eye(T) 
                        # Fill lower triangle (backward looking? no, C is forward looking wrt r usually)
                        # Actually C_t depends on r_{t}, r_{t+1}... 
                        # So it's upper triangular-ish in SSJ logic?
                        J += np.triu(np.ones((T,T)), k=1) * -0.01
                    
                    bj[f"d{out_v}_d{in_v}"] = J
            
            block_jacs[b_name] = bj
            
        return block_jacs
