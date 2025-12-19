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
        """CRRA Utility Function: u(c) - v(h)"""
        if abs(self.sigma - 1.0) < 1e-5:
            u_c = np.log(c)
        else:
            u_c = (c ** (1.0 - self.sigma)) / (1.0 - self.sigma)
            
        # Labor disutility (if h provided)
        # v(h) = h^(1+nu) / (1+nu)
        if h > 1e-9:
             nu = 1.0 / frisch
             v_h = (h ** (1.0 + nu)) / (1.0 + nu)
             return u_c - v_h
        return u_c

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
