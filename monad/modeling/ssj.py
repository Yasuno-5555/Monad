
import numpy as np
import sympy
import scipy.sparse
import scipy.sparse.linalg
from typing import Dict, List, Optional
from monad.modeling.schema import ModelSpec
from monad.cpp_backend import get_backend

class LinearSSJSolver:
    """
    Linearized Sequence Space Jacobian Solver for Descriptor Models.
    Handles models with both Scalar Equations and Heterogeneous Blocks.
    """
    def __init__(self, spec: ModelSpec, T=50):
        self.spec = spec
        self.T = T
        self.var_names = list(self.spec.variables.keys())
        self.N = len(self.var_names)
        self.param_dict = {k: v.value for k, v in self.spec.parameters.items()}
        
        # Identify Blocks
        self.blocks = {} # name -> backend instance
        self.block_outputs = set()
        
        for b_name, b_spec in self.spec.blocks.items():
            # Init backend for each block
            # Assuming kernel string maps to model_type for get_backend
            # e.g. kernel="OneAsset" -> model_type="one_asset"
            m_type = b_spec.kernel.lower() if b_spec.kernel else "one_asset"
            if "two" in m_type: m_type = "two_asset"
            elif "one" in m_type: m_type = "one_asset"
            
            bk = get_backend(model_type=m_type, T=T, params=self.param_dict)
            self.blocks[b_name] = bk
            for out in b_spec.outputs:
                self.block_outputs.add(out)
                
        # Parse Scalar Equations
        # We need derivatives w.r.t var[t], var[t-1], var[t+1]
        self.eq_derivs = [] # List of dicts: {var_lag: expr}
        self._compile_scalar_derivatives()
        
        # Precompute Block Jacobians
        self.block_jacs = {} # (b_name, output, input) -> matrix
        self._compute_block_jacs()
        
    def _compile_scalar_derivatives(self):
        """Linearlize scalar equations around SS (symbolically)."""
        print("[SSJ] Linearizing scalar equations...")
        
        # 1. Define symbolic variables for t, t-1, t+1
        # We treat parameters as constants (subbed in)
        
        sym_vars = {}
        for v in self.var_names:
            sym_vars[f"{v}_t"] = sympy.Symbol(f"{v}_t")
            sym_vars[f"{v}_tm1"] = sympy.Symbol(f"{v}_tm1")
            sym_vars[f"{v}_tp1"] = sympy.Symbol(f"{v}_tp1")
            
        sym_params = {k: v for k, v in self.param_dict.items()}
        
        for i, eq_spec in enumerate(self.spec.equations):
            # Parse string: substitute x[t] -> x_t
            # We assume user writes nice equations.
            # Regex or simple replace.
            
            eq_str = eq_spec.resid
            for v in self.var_names:
                eq_str = eq_str.replace(f"{v}[t]", f"{v}_t")
                eq_str = eq_str.replace(f"{v}[t-1]", f"{v}_tm1")
                eq_str = eq_str.replace(f"{v}[t+1]", f"{v}_tp1")
            
            # Parse
            # Add variables to local dict
            local_dict = {**sym_vars, **sym_params} # Params as values? No, assuming simple float params
            # Actually, sym_params might be used in equation string?
            # Ideally parse with symbols, then sub params.
            
            expr = sympy.parse_expr(eq_str, local_dict=local_dict)
            
            # Sub params
            # Note: ParameterSpec values are floats
            expr = expr.subs(sym_params)
            
            # Evaluate at SS (Guess)
            # We assume Guess IS SS.
            ss_subs = {}
            for v_name, v_spec in self.spec.variables.items():
                val = v_spec.guess
                ss_subs[sym_vars[f"{v_name}_t"]] = val
                ss_subs[sym_vars[f"{v_name}_tm1"]] = val
                ss_subs[sym_vars[f"{v_name}_tp1"]] = val
                
            # Compute Gradient w.r.t each var at each time
            derivs = {}
            for v in self.var_names:
                for lag in ['tm1', 't', 'tp1']:
                    s = sym_vars[f"{v}_{lag}"]
                    diff = sympy.diff(expr, s)
                    val = float(diff.subs(ss_subs))
                    if abs(val) > 1e-9:
                        derivs[(v, lag)] = val
            
            self.eq_derivs.append(derivs)

    def _compute_block_jacs(self):
        print("[SSJ] Computing Block Jacobians...")
        for name, bk in self.blocks.items():
            # Assuming backend caches or computes efficiently
            jacs = bk.compute_jacobians()
            
            # Map internal names to spec names?
            # Spec Block says: outputs=['C', 'A'], inputs=['r', 'w']
            # Backend returns: J_C_r, J_C_w, ...
            # We need a mapping strategy.
            # For now, HARDCODED mapping for OneAsset/TwoAsset.
            # TODO: Generalize mapping in BlockSpec.
            
            b_spec = self.spec.blocks[name]
            
            # Inputs mapping:
            # Spec input 'r' -> Backend 'r_m'? or just index?
            # We'll assume simple name matching for now or robust heuristics.
            
            # Iterate inputs/outputs defined in Spec
            for out_v in b_spec.outputs:
                for in_v in b_spec.inputs:
                    # Construct key like J_C_r
                    # Try to find in backend jacobians
                    # Candidates: J_{out}_{in}, J_{out_v}_{in_v}
                    
                    # Common HANK mapping: r -> r_m, w -> w (usually z)
                    in_key = in_v
                    if in_v == 'r' and 'J_C_rm' in jacs: in_key = 'rm'
                    if (in_v == 'w' or in_v == 'Z') and 'J_C_w' in jacs: in_key = 'w'
                    
                    jac_name = f"J_{out_v}_{in_key}" # Only if out_v matches C/A
                    # Backend usually has hardcoded keys: J_C_rm, J_C_w, J_A_rm...
                    
                    # Fallback lookup
                    mat = None
                    # Try direct match
                    if f"J_{out_v}_{in_v}" in jacs:
                        mat = jacs[f"J_{out_v}_{in_v}"]
                    # Try known aliases
                    elif out_v == 'C' and in_v == 'r': mat = jacs.get('J_C_rm')
                    elif out_v == 'C' and (in_v == 'w' or in_v == 'Y'): mat = jacs.get('J_C_w')
                    elif out_v == 'A' and in_v == 'r': mat = jacs.get('J_A_rm')
                    
                    if mat is not None:
                        self.block_jacs[(name, out_v, in_v)] = np.array(mat)
                        # print(f"  Found Jacobian for {name}: d{out_v}/d{in_v}")

    def solve(self, shock_dict: Dict[str, float] = None, max_iter=None, tol=None, damping=None): # Args for compat
        """
        Solve linear system: H * dX = -H_exo * dExo
        Or better: Treat shocks as Initial State displacements or Exogenous variables.
        Here we assume `shock_dict` maps variable name -> initial shock value OR scalar shock to exog.
        For HANK, usually shock to `r` or `Z`.
        """
        if shock_dict is None: return {}
        print("[SSJ] Solving Linear System...")
        
        # 1. Build Global H matrix (Block Diagonal + Dense Blocks)
        # Size: (N*T) x (N*T)
        rows_list = []
        cols_list = []
        data_list = []
        
        # Variable Index Map: (var, t) -> col_idx
        def idx(v, t):
            return t * self.N + self.var_names.index(v)
            
        # A. Scalar Equations
        # Each equation k at time t adds a ROW to H.
        # Which row? We map Eq k to Variable k?
        # Standard convention: N equations, N variables.
        # Eq k corresponds to clearing condition for Var k? No, specific ordering.
        # We just fill rows (t*N + k).
        
        eq_idx_offset = 0
        
        # We need to distinguish:
        # - Equations defining vars (Scalar Eqs)
        # - Block definitions (Identity with Jacobian)
        
        # Actually, in ModelSpec, 'equations' list + 'blocks'.
        # Total # equations should match N.
        # If Block defines 'C', we don't include an Euler equation for 'C'.
        # The Block GIVES us the equation: C = G(...) => dC - J dr = 0.
        # So we iterate variables to assign equations?
        
        # Heuristic:
        # 1. Non-block variables MUST have a Scalar Equation in `equations`.
        # 2. Block outputs use the Block Jacobian equation.
        
        # Map var -> is_block_output
        is_block_out = {v: False for v in self.var_names}
        for out in self.block_outputs:
            if out in self.var_names: is_block_out[out] = True
            
        # We assume `spec.equations` covers the non-block variables.
        # And we generate Block equations for block variables.
        
        scalar_eq_counter = 0
        
        for t in range(self.T):
            row_base = t * self.N
            
            # 1. Scalar Equations from Spec
            for eq_def in self.eq_derivs:
                # This equation goes to... which row?
                # We need to map EQ index to Row index.
                # Let's assume the ORDER of equations in YAML matches the Non-Block variables?
                # This is fragile. 
                # Better: In YAML, usually order matters.
                # BUT, wait. If we have N variables, we need N equations per time step.
                # `spec.equations` has K equations.
                # `spec.blocks` has M outputs.
                # We expect K + M = N.
                
                # We assign rows sequentially.
                # First K rows: Scalar Eqs.
                # Next M rows: Block Eqs.
                
                # Check for boundary (t-1, t+1)
                for (var, lag), val in eq_def.items():
                    # Determine column
                    if lag == 't': c = idx(var, t)
                    elif lag == 'tm1': c = idx(var, t-1) if t > 0 else -1
                    elif lag == 'tp1': c = idx(var, t+1) if t < self.T-1 else -1
                    
                    if c >= 0:
                        rows_list.append(row_base + scalar_eq_counter)
                        cols_list.append(c)
                        data_list.append(val)
                
                scalar_eq_counter += 1
            
            # Reset counter for next t, but we need to track offset within t
            # Let's track `current_row` inside the t loop.
            current_row = row_base + len(self.spec.equations)
            
            # 2. Block Equations
            # dOut - sum(J * dIn) = 0
            for b_name, b_spec in self.spec.blocks.items():
                for out_v in b_spec.outputs:
                    if out_v not in self.var_names: continue
                    
                    # Row for this output
                    # dOut_t term: coeff 1.0
                    rows_list.append(current_row)
                    cols_list.append(idx(out_v, t))
                    data_list.append(1.0)
                    
                    # dIn terms: -J_{t, s}
                    # We need to iterate ALL 's' (time) for the input path.
                    # This implies the Block Equation is GLOBAL, not time-local.
                    # Wait, Block Jacobian is T x T.
                    # So the equation for C_t involves r_0...r_{T-1}.
                    # We handle this OUTSIDE the t loop?
                    # YES. Block rows are best handled globally.
                    
                    current_row += 1
            
            scalar_eq_counter = 0 # Reset for next time step
            
        # B. Efficient Block Filling (Global)
        block_row_start = len(self.spec.equations) # Local offset per T? 
        # No, mixing Local (t) and Global (Block) equations is messy if we iterate t.
        # Let's reorganize.
        # Rows 0..(T * K - 1): Scalar Equations (Time stacked)
        # Rows (T * K)..(T * N - 1): Block Equations (Grouped by Variable)
        
        # New Row Mapping:
        # Eq k at time t -> Row = k * T + t
        # Block Var v (at time t) -> Row = (Offset + v_idx) * T + t
        
        rows_list = [] # Reset
        cols_list = []
        data_list = []
        
        K = len(self.spec.equations)
        # 1. Scalar Equations
        for k, eq_def in enumerate(self.eq_derivs):
            for t in range(self.T):
                r = k * self.T + t
                for (var, lag), val in eq_def.items():
                    if lag == 't': c = idx(var, t)
                    elif lag == 'tm1': c = idx(var, t-1) if t > 0 else -1
                    elif lag == 'tp1': c = idx(var, t+1) if t < self.T-1 else -1
                    
                    if c >= 0:
                        rows_list.append(r)
                        cols_list.append(c)
                        data_list.append(val)

        # 2. Block Equations
        # Base row index for blocks
        block_row_base = K * self.T
        
        current_block_offset = 0
        for b_name, b_spec in self.spec.blocks.items():
            for out_v in b_spec.outputs:
                if out_v not in self.var_names: continue
                
                # Identity term: I * dOut
                for t in range(self.T):
                    r = block_row_base + current_block_offset * self.T + t
                    c = idx(out_v, t)
                    rows_list.append(r)
                    cols_list.append(c)
                    data_list.append(1.0)
                
                # Jacobian terms: -J * dIn
                for in_v in b_spec.inputs:
                    key = (b_name, out_v, in_v)
                    if key in self.block_jacs:
                        J = self.block_jacs[key] # T x T
                        # Add -J to appropriate entries
                        # row r=t corresponds to output time t
                        # col corresponds to input time s
                        # J[t, s] is effect of in_s on out_t
                        
                        # Optimization: J might be dense, 50x50=2500 entries. Fine.
                        for t in range(self.T):
                             r = block_row_base + current_block_offset * self.T + t
                             for s in range(self.T):
                                 val = -J[t, s]
                                 if abs(val) > 1e-9:
                                     c = idx(in_v, s)
                                     rows_list.append(r)
                                     cols_list.append(c)
                                     data_list.append(val)
                
                current_block_offset += 1

        # Construct H
        dim = self.N * self.T
        H = scipy.sparse.coo_matrix((data_list, (rows_list, cols_list)), shape=(dim, dim)).tocsc()
        
        # 2. Construct RHS (Shock)
        # We handle shocks as 'e_r' (exogenous variable) perturbation?
        # If 'e_r' is a variable in the system, we need an equation for it. e.g. e_r - rho*e_r(-1) = shock.
        # If the shock is to the *equation* (e.g. initial displacement), we put it in RHS.
        
        RHS = np.zeros(dim)
        
        # HANK logic: Usually shock is to 'r_star' or 'Z' equation.
        # Or initial state displacement for AR1.
        
        # We iterate over shock_dict.
        # If key 'e_r': value is the shock size at t=0? Or path?
        # Implicitly, we modify the residual of the equation governing 'e_r' at t=0.
        # Eq: e_r[t] - rho*e_r[t-1] = 0.
        # At t=0: e_r[0] - rho*e_r[-1] = 0.
        # If we shock, we say e_r[0] = val (if pure shock) or e_r[0] - ... = val.
        # Standard: Add shock to the equation residual at t=0.
        
        for k, v in shock_dict.items():
            # Find which equation governs variable 'k'.
            # Heuristic: Eq k defines Var k? Or we search eq string?
            # We assume the user shocked a *Variable* that is governed by an *Equation*.
            # Which equation? The one where it appears on LHS? 
            # Or simplified: We assume 1-to-1 mapping Eq i <-> Var i for Scalar Eqs.
            # But we don't know the mapping!
            
            # Search for equation containing the variable on LHS?
            # Let's search `eq_derivs`. If `(k, 't')` deriv is 1.0 (or close), it's a candidate.
            
            target_eq_idx = -1
            for idx_eq, deriv in enumerate(self.eq_derivs):
                if (k, 't') in deriv:
                    target_eq_idx = idx_eq
                    break
            
            if target_eq_idx >= 0:
                # Add to RHS at t=0
                # H dX = -Residual.
                # If equation is e_r - ... = 0, and we want e_r to jump,
                # we are effectively saying the equation is e_r - ... = shock.
                # So Residual = -shock. RHS = +shock?
                # Wait. H * dX + dResidual/dShock * Shock = 0
                # H * dX = - dResidual/dShock * Shock
                # If Eq is: x - shock = 0. dR/dShock = -1. RHS = -(-1)*S = S.
                # If Eq is: x = 0 (and we add shock). x - S = 0. Correct.
                
                row_idx = target_eq_idx * self.T  # t=0
                if isinstance(v, (int, float)):
                    RHS[row_idx] = v
                else:
                    # Array path
                    pass # TODO path shocks
            else:
                print(f"[SSJ] Warning: Could not find scalar equation for shock {k}")

        # 3. Solve
        print(f"[SSJ] Solving {dim}x{dim} system...")
        try:
            dX = scipy.sparse.linalg.spsolve(H, RHS)
        except Exception as e:
            print(f"[SSJ] Solver Failed: {e}")
            return {}
            
        # 4. Unpack
        res_dict = {}
        for i, var in enumerate(self.var_names):
             val = dX[i::self.N] # Stride N? No, sorting is different now.
             # We sorted by Equation then Block.
             # But columns are still ordered by idx(var, t).
             # idx(v, t) = t*N + v_idx.
             # So standard unpacking works! Only Rows were permuted.
             
             # Extract: indices [i, i+N, i+2N...]
             indices = [t * self.N + i for t in range(self.T)]
             res_dict[var] = dX[indices]
             
        return res_dict
