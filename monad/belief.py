"""
Belief System for Monad HANK (Phase 4)
=====================================
Defines the structure of heterogeneous beliefs (Opinion, Sentiment, Bias).

The BeliefGrid adds a 4th dimension to the state space (m, a, z, b).
b represents "Optimism Bias" in expected returns.
"""

import numpy as np

class BeliefGrid:
    """
    Represents the grid of Belief Biases (b) across households.
    
    State 'b' acts as a subjective bias on expected returns:
        E_subjective[R] = E_rational[R] * (1 + b)
        
    Attributes:
        nodes (np.ndarray): Grid points for b (e.g., [-0.02, 0.0, 0.02])
        trans_mat (np.ndarray): Transition matrix (Markov process for b)
    """
    def __init__(self, range=(-0.02, 0.02), n=3, persistence=0.9):
        """
        Args:
            range (tuple): Min and Max bias (e.g. -2% to +2%)
            n (int): Number of belief states (Keep small! 3-5 recommended)
            persistence (float): Probability of staying in current belief state
        """
        self.min_b, self.max_b = range
        self.n = n
        self.persistence = persistence
        
        # 1. Create Nodes (Linspace)
        if n == 1:
            self.nodes = np.array([0.0])
        else:
            self.nodes = np.linspace(self.min_b, self.max_b, n)
            
        # 2. Create Transition Matrix (Simple Rouwenhorst or Tauchen-like)
        # For small N with high persistence, main diagonal is 'persistence', rest distributed
        self.trans_mat = self._build_transition_matrix(n, persistence)
        
    def _build_transition_matrix(self, n, rho):
        if n == 1:
            return np.ones((1, 1))
            
        P = np.zeros((n, n))
        
        # Simple Logic: 
        # Diagonals = rho
        # Neighbors = (1-rho)/2 (or remainder)
        # Boundaries reflected
        
        for i in range(n):
            P[i, i] = rho
            remaining = 1.0 - rho
            
            if i == 0:
                P[i, i+1] = remaining
            elif i == n-1:
                P[i, i-1] = remaining
            else:
                P[i, i-1] = remaining / 2.0
                P[i, i+1] = remaining / 2.0
                
        return P

    def get_info(self):
        return {
            "n": self.n,
            "range": (self.min_b, self.max_b),
            "persistence": self.persistence
        }

    def __repr__(self):
        return f"BeliefGrid(n={self.n}, range={self.min_b:.1%}~{self.max_b:.1%}, rho={self.persistence})"
