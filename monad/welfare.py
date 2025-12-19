import numpy as np
import pandas as pd

class WelfareAnalyzer:
    """
    Computes welfare metrics from Monad simulation results.
    Focuses on:
    1. Aggregate Welfare (Utilitarian)
    2. Distributional Welfare (Weighted)
    3. Consumption Equivalent Variation (CEV)
    """
    def __init__(self, sigma=2.0, beta=0.97):
        self.sigma = sigma
        self.beta = beta

    def utility(self, c, h=0.0, frisch=1.0):
        """CRRA Utility Function: u(c) - v(h). Supports scalar and array inputs."""
        c = np.atleast_1d(c)
        
        if abs(self.sigma - 1.0) < 1e-5:
            u_c = np.log(c)
        else:
            u_c = (c ** (1.0 - self.sigma)) / (1.0 - self.sigma)
            
        # Labor disutility (if h provided and non-zero)
        h = np.atleast_1d(h)
        if np.any(h > 1e-9):
            nu = 1.0 / frisch
            v_h = (h ** (1.0 + nu)) / (1.0 + nu)
            result = u_c - v_h
        else:
            result = u_c
            
        # Return scalar if input was scalar
        if result.size == 1:
            return float(result.flat[0])
        return result


    def compute_welfare_path(self, c_path, h_path=None):
        """Compute discounted sum of utility along a path."""
        T = len(c_path)
        w = 0.0
        for t in range(T):
            u_t = self.utility(c_path[t], h_path[t] if h_path is not None else 0.0)
            w += (self.beta ** t) * u_t
        return w

    def compute_value_function_welfare(self, value_func, distribution):
        """
        Compute Ex-Ante Social Welfare using the Value Function V(s) and Distribution D(s).
        W = Sum( V(s) * D(s) )
        """
        return np.dot(value_func, distribution)

    def compute_cev(self, base_welfare, alt_welfare):
        """
        Compute Consumption Equivalent Variation (lambda).
        How much consumption must be increased in Base to match Alt welfare?
        
        For Log Utility (sigma=1):
        W_alt = W_base + log(1 + lambda) / (1 - beta)
        lambda = exp( (W_alt - W_base) * (1 - beta) ) - 1
        
        For CRRA (sigma != 1):
        W = C^(1-s)/(1-s). W_new = (C(1+lam))^(1-s)/(1-s) = (1+lam)^(1-s) * W
        1 + lambda = (W_alt / W_base) ^ (1 / (1-sigma))
        """
        if abs(self.sigma - 1.0) < 1e-5:
            # Log case
            diff = (alt_welfare - base_welfare) * (1.0 - self.beta)
            return np.exp(diff) - 1.0
        else:
            # CRRA case using Value Function logic
            # Be careful with signs if sigma > 1 (Utilities are negative)
            # Formula: (W_alt / W_base)^(1/(1-sigma)) - 1
            # Check for negative W
            ratio = alt_welfare / base_welfare
            exponent = 1.0 / (1.0 - self.sigma)
            return (ratio ** exponent) - 1.0

class WelfareResult:
    """Helper wrapper for analysis."""
    def __init__(self, res_dict, params=None):
        self.res = res_dict
        self.params = params or {}
        sigma = self.params.get('sigma', 2.0)
        beta  = self.params.get('beta', 0.97)
        self.analyzer = WelfareAnalyzer(sigma, beta)
        
        # Compute baseline welfare if distribution/value available
        if 'distribution' in res_dict and 'value' in res_dict:
            self.W = self.analyzer.compute_value_function_welfare(
                np.array(res_dict['value']), np.array(res_dict['distribution'])
            )
        else:
            self.W = None

    def compare(self, other_res_wrapper):
        """Compare this result (Alternative) against another (Baseline)."""
        if self.W is None or other_res_wrapper.W is None:
            raise ValueError("Cannot compare: Missing value function or distribution.")
            
        cev = self.analyzer.compute_cev(other_res_wrapper.W, self.W)
        print(f"B. Welfare Comparison:")
        print(f"  Base W: {other_res_wrapper.W:.4f}")
        print(f"  Alt W:  {self.W:.4f}")
        print(f"  CEV (lambda): {cev*100:.4f}%")
        return cev


class WelfareAnalysis:
    """
    Phase 3: The Judge.
    Compares two simulation results and computes heterogeneous welfare metrics.
    Supports transition path welfare with expectation operator.
    """
    def __init__(self, base, alt, sigma=2.0, beta=0.97):
        """
        Args:
            base: Dict with simulation results (baseline policy)
            alt: Dict with simulation results (alternative policy)
            sigma: CRRA coefficient
            beta: Discount factor
        """
        self.base = base
        self.alt = alt
        self.sigma = sigma
        self.beta = beta
        self.analyzer = WelfareAnalyzer(sigma, beta)
        
        # Cache computed welfare
        self._v_base = None
        self._v_alt = None
        self._cev_grid = None
        
    def compute_transition_welfare(self, paths, trans_mat=None, mu_0=None):
        """
        Compute discounted lifetime utility along a transition path.
        
        Args:
            paths: Dict with 'C', 'N' arrays of shape (T, n_a, n_e)
                   or (T,) for aggregate paths
            trans_mat: Transition matrix for expectation (n_states x n_states)
                      If None, assumes identity (no state transitions)
            mu_0: Initial distribution (optional, for weighted aggregate)
        
        Returns:
            V_0: Present value of lifetime utility, shape (n_a, n_e) or scalar
        """
        C = paths.get('C', paths.get('dC', None))
        N = paths.get('N', paths.get('dN', None))
        
        if C is None:
            raise ValueError("Paths must contain 'C' or 'dC'")
            
        # Ensure proper shape
        C = np.atleast_1d(C)
        T = C.shape[0]
        
        # Handle different dimensionalities
        if C.ndim == 1:
            # Aggregate path (T,)
            V = 0.0
            for t in reversed(range(T)):
                c_t = 1.0 + C[t]  # Deviation from steady state (ss=1)
                n_t = N[t] if N is not None else 0.0
                u_t = self.analyzer.utility(c_t, n_t)
                V = u_t + self.beta * V
            return V
            
        elif C.ndim == 3:
            # Heterogeneous path (T, n_a, n_e)
            n_a, n_e = C.shape[1], C.shape[2]
            V = np.zeros((n_a, n_e))
            
            for t in reversed(range(T)):
                c_t = 1.0 + C[t]  # Shape: (n_a, n_e)
                n_t = N[t] if N is not None else np.zeros_like(c_t)
                
                # Utility at t
                u_t = self.analyzer.utility(c_t, n_t)
                
                # Expectation: E_t[V_{t+1}]
                if trans_mat is not None:
                    # trans_mat @ V: Apply expectation over productivity 'e'
                    # Assuming trans_mat is (n_e x n_e)
                    expected_V = V @ trans_mat.T  # (n_a, n_e) @ (n_e, n_e).T
                else:
                    expected_V = V  # No transition (identity)
                    
                V = u_t + self.beta * expected_V
                
            return V
        else:
            raise ValueError(f"Unexpected path shape: {C.shape}")
    
    def compute_cev_grid(self):
        """
        Compute CEV for each state (a, e).
        
        Returns:
            cev_grid: Array of CEV values, shape matching state space
        """
        if self._cev_grid is not None:
            return self._cev_grid
            
        # Compute welfare for both
        trans_mat = self.base.get('trans_mat', None)
        
        self._v_base = self.compute_transition_welfare(self.base, trans_mat)
        self._v_alt = self.compute_transition_welfare(self.alt, trans_mat)
        
        # CEV: (V_alt / V_base)^(1/(1-sigma)) - 1
        # Handle potential negative values for sigma > 1
        ratio = self._v_alt / self._v_base
        exponent = 1.0 / (1.0 - self.sigma)
        self._cev_grid = np.sign(ratio) * (np.abs(ratio) ** exponent) - 1.0
        
        return self._cev_grid
    
    def aggregate_cev(self, mu=None):
        """
        Compute aggregate CEV weighted by distribution.
        
        Args:
            mu: Distribution over states (n_a, n_e). If None, uniform.
            
        Returns:
            Scalar aggregate CEV
        """
        cev = self.compute_cev_grid()
        
        if np.isscalar(cev):
            return cev
            
        if mu is None:
            # Uniform distribution
            mu = np.ones_like(cev) / cev.size
            
        return np.sum(cev * mu)
    
    def summary(self, mu=None, quintiles=5):
        """
        Generate summary statistics by wealth quintile.
        
        Returns:
            String summary
        """
        cev = self.compute_cev_grid()
        
        if np.isscalar(cev):
            return f"Aggregate CEV: {cev*100:.2f}%"
            
        # Compute quintile averages (along asset dimension)
        n_a = cev.shape[0]
        q_size = n_a // quintiles
        
        lines = ["CEV by Wealth Quintile:"]
        for q in range(quintiles):
            start = q * q_size
            end = (q + 1) * q_size if q < quintiles - 1 else n_a
            q_cev = np.mean(cev[start:end, :])
            label = f"Q{q+1}" if q < quintiles - 1 else f"Top {100//quintiles}%"
            lines.append(f"  {label}: {q_cev*100:.2f}%")
            
        return "\n".join(lines)
    
    def plot_distributional_gains(self, ax=None, title="CEV Distribution"):
        """
        Plot heatmap of CEV across state space.
        """
        import matplotlib.pyplot as plt
        
        cev = self.compute_cev_grid()
        
        if np.isscalar(cev):
            print("Cannot plot: Aggregate path has no distribution.")
            return
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            
        # Heatmap
        im = ax.imshow(cev * 100, aspect='auto', cmap='RdYlGn', origin='lower')
        ax.set_xlabel("Productivity (e)")
        ax.set_ylabel("Assets (a)")
        ax.set_title(title)
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("CEV (%)")
        
        plt.tight_layout()
        return ax
