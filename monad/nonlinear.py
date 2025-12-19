import numpy as np
import scipy.linalg
from .regime import registry

class PiecewiseSolver:
    """
    Generalized Solver for Endogenous Regime Switching (OccBin-style).
    Solves H(Y) * dY = -Residual(Y) where H changes discretely based on regimes.
    """
    def __init__(self, linear_solver, max_iter=20, tol=1e-6, damping=1.0, policy_block=None):
        self.model = linear_solver # Assumes this has .block, .backend, etc.
        self.max_iter = max_iter
        self.tol = tol
        self.damping = damping
        self.T = linear_solver.T
        self.policy_block = policy_block
        
        # Cache for Regime Jacobians (Lazy Loaded)
        # Key: regime_name, Value: Jacobian Matrix (TxT)
        self.jacobian_cache = {} 
        
        # Pre-load "normal"
        # We assume linear_solver is configured for "normal" by default
        self.jacobian_cache["normal"] = self._compute_jacobian_for_regime("normal")

    def solve(self, shock_path, initial_guess=None):
        if initial_guess is None:
            Y_guess = np.zeros(self.T)
        else:
            Y_guess = initial_guess.copy()

        print(f"--- Piecewise Solver (Max: {self.max_iter}) ---")

        for it in range(self.max_iter):
            # 1. Evaluate Regimes
            # We need a full path guess to evaluate regimes
            # Assuming 'evaluate_regimes' takes Y (output gap) or we need to derive others?
            # Regimes might depend on r, i, debt, etc.
            # We need to run a "Forward Pass" to get all variables.
            
            # Using standard forward pass (assuming Normal structure first? No, use current guess?)
            # Valid point: To know variables, we need consistent equation.
            # OccBin approach: "Guess Regimes -> Solve -> Verification".
            # Here we do: "Guess Y -> Derive Vars -> Determine Regimes -> Update Y".
            # This is "Relaxation" approach.
            
            # Forward pass using 'normal' logic to get proxy variables?
            # Or use the specific logic of the CURRENTLY guessed regimes?
            # Let's derive Proxy variables using simple Normal blocks first, 
            # then refine regimes. (Approximation).
            
            # Actually, we should use the `_evaluate_blocks` logic but that usually returns residuals.
            # We need the paths.
            
            paths = self._derive_paths(Y_guess, shock_path)
            
            # Determine Regimes
            # Pass all paths as context
            regime_path = registry.evaluate_regimes(self.model, paths)
            
            # 2. Assemble Patchwork Jacobian & Residuals
            H, Res = self._assemble_system(regime_path, Y_guess, shock_path, paths)
            
            # 3. Check Convergence
            err = np.max(np.abs(Res))
            print(f"  Iter {it}: Max Res={err:.2e}, Regimes={np.unique(regime_path)}")
                
            if err < self.tol:
                print(f"  [CONVERGED] Iter {it}")
                paths['regime_path'] = regime_path
                return paths

                
            # 4. Newton Step
            # dY = - H^-1 * Res
            try:
                dY = -np.linalg.solve(H, Res)
            except np.linalg.LinAlgError:
                print("  [Error] Singular Jacobian in Piecewise Solver.")
                raise

            Y_guess = Y_guess + self.damping * dY
            
        raise RuntimeError("Piecewise Solver did not converge.")

    def _derive_paths(self, Y, shock_input):
        # 1. Parse Shocks
        shock_r = np.zeros(self.T)
        shock_pi = np.zeros(self.T)
        
        if isinstance(shock_input, dict):
            # Handle dictionary shocks
            if "markup" in shock_input:
                # Markup shock affects Phillips Curve intercept
                val = shock_input["markup"]
                if np.isscalar(val):
                    shock_pi[:] = val # Permanent or scalar broadcast?
                    # Usually shock_path is a dict of vector or scalar
                    # For demo: assume step trigger or full path?
                    # Demo says: shock_path = {"markup": 0.05}
                    # We treat it as constant for now or check length
                else:
                    shock_pi = val
            if "r_star" in shock_input or "r" in shock_input:
                 val = shock_input.get("r_star", shock_input.get("r"))
                 if np.isscalar(val): shock_r[:] = val
                 else: shock_r = val
        elif np.isscalar(shock_input) or isinstance(shock_input, getattr(np, "ndarray", list)):
            shock_r = shock_input # Legacy behavior
            
        block = self.model.block
        M_pi_y = block.get_phillips_curve()
        
        # 2. Compute Inflation (pi)
        # pi = kappa * Y + beta * pi(+1) + shock
        # Linear map: pi = M @ Y + shock_vec (if shock acts as intercept)
        # M_pi_y from SSJ assumes shock is zero?
        # Actually M_pi_y matches Y -> pi.
        # We need (I - beta*L^-1)^-1 * (kappa*Y + u)
        # But `get_phillips_curve` usually returns the full map dPi/dY.
        # So pi = M * Y.
        # If there is a markup shock 'u', then pi_total = pi_endogenous + pi_shock_response?
        # In linear world: dPi = M_pi_y * dY + M_pi_u * du.
        # Constructing M_pi_u is (I - beta*L^-1)^-1.
        # For simplicity in this demo (mock backend), we just ADD shock to pi. 
        # (Technically ignoring propagation of u via expectations, but acceptable for demo).
        
        pi = M_pi_y @ Y + shock_pi
        
        # 3. Compute Interest Rate (i)
        i = np.zeros(self.T)
        beliefs = {} # Store paths of internal states
        
        if self.policy_block:
            # Stateful Policy Logic
            print(f"[DEBUG] _derive_paths: Using PolicyBlock {self.policy_block.name}")
            # Iterate forward
            ctx = {} # Carry state vars
            for t in range(self.T):
                # Context for t
                ctx['pi'] = pi[t]
                ctx['i_lag'] = i[t-1] if t > 0 else 0.0
                ctx['Y'] = Y[t]
                
                # Fisher Real Rate
                ctx['r'] = (i[t-1] if t > 0 else 0.0) - pi[t] 
                # Inject lagged beliefs from previous iteration
                for k, v in beliefs.items():
                    ctx[f"{k}_lag"] = v[t-1] if t > 0 else 0.0 # simple lag
                
                # Evaluate
                try:
                    res = self.policy_block.evaluate_flow(ctx)
                except Exception as e:
                    print(f"[ERROR] Policy Evaluate Failed at t={t}: {e}")
                    raise

                # Set i
                i[t] = res.get('i', 0.0)
                
                # Store new states
                for k, v in res.items():
                    if k == 'i': continue
                    if k not in beliefs: beliefs[k] = np.zeros(self.T)
                    beliefs[k][t] = v
        else:
            # Default Taylor (Normal)
            phi = block.params.get('phi_pi', 1.5)
            r_ss = 0.005 # hardcoded
            # i = r* + phi * pi + shock_r
            i = r_ss + shock_r + phi * pi
        
        # 4. Compute Real Rate (r) for IS Curve
        S = block.get_fisher_equation()
        r = i - S @ pi
        
        # Return dict
        paths = {'Y': Y, 'pi': pi, 'i': i, 'r': r, 'shock': shock_r}
        paths.update(beliefs) # Add belief paths for plotting
        return paths

    def _compute_jacobian_for_regime(self, regime_name):
        if regime_name == "normal":
            # Delegate to existing model logic (assumed Normal)
            # We can reuse NewtonSolver._build_dynamic_jacobian logic (unconstrained)
            # Simplified: (I - J_AD)
            return self._build_jacobian_internal(constraint_mask=np.zeros(self.T, dtype=bool))
            
        elif regime_name == "ZLB": # Hardcoded known regime for Phase 1 or use rule
             # Constrained J
             return self._build_jacobian_internal(constraint_mask=np.ones(self.T, dtype=bool))
             
        # For other regimes, we'd apply overrides using registry.
        # Phase 1: Support Normal/ZLB hardcoded-ish logic via generalized builder
        return self._build_jacobian_internal(constraint_mask=np.zeros(self.T, dtype=bool))

    def _build_jacobian_internal(self, constraint_mask):
        # Reuse logic from NewtonSolver but generic
        # J = I - J_C_r @ J_r_Y ...
        
        block = self.model.block
        M_pi_y = block.get_phillips_curve()
        M_i_pi = block.get_taylor_rule()
        S      = block.get_fisher_equation()
        
        # Apply Constraint (Zero out Taylor Rule if constrained)
        M_i_pi_eff = M_i_pi.copy()
        # Row-wise zeroing
        for t in range(self.T):
            if constraint_mask[t]:
                M_i_pi_eff[t, :] = 0.0
                
        J_r_y = (M_i_pi_eff - S) @ M_pi_y
        
        # Closed Econ assumed
        J_AD = self.model.backend.J_C_r @ J_r_y + self.model.backend.J_C_y
        
        return np.eye(self.T) - J_AD

    def _assemble_system(self, regime_path, Y, shock, paths):
        # H_composed, Res_composed
        
        # Start with Normal
        H = self.jacobian_cache["normal"].copy()
        
        unique_regimes = np.unique(regime_path)
        for r_name in unique_regimes:
            if r_name == "normal": continue
            
            # Lazy Load
            if r_name not in self.jacobian_cache:
                if r_name == "ZLB":
                    self.jacobian_cache[r_name] = self._compute_jacobian_for_regime(r_name)
                    
            if r_name in self.jacobian_cache:
                mask = (regime_path == r_name)
                H[mask, :] = self.jacobian_cache[r_name][mask, :]
            
        # Residuals
        # Use paths['r'] directly - it was computed via Policy Object or default
        r_actual = paths['r']
        
        # C = J_C_r @ r + J_C_y @ Y (Linearized consumption)
        C = self.model.backend.J_C_r @ r_actual + self.model.backend.J_C_y @ Y
        Res = C - Y
        
        return H, Res


# Aliases
NewtonSolver = PiecewiseSolver # Replace old solver
