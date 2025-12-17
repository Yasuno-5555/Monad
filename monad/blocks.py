import numpy as np

class NKBlock:
    """
    Analytical derivatives for New Keynesian blocks.
    Provides Jacobian matrices for the supply side and monetary authority.
    """
    def __init__(self, T, params=None):
        self.T = T
        if params is None:
            self.params = {'kappa': 0.1, 'beta': 0.99, 'phi_pi': 1.5}
        else:
            self.params = params

    def get_phillips_curve(self):
        """
        New Keynesian Phillips Curve: pi_t = kappa * Y_t + beta * pi_{t+1}
        Returns M_pi_y (d_pi / d_Y).
        This is an Upper Triangular Matrix (Backward Substitution).
        """
        T = self.T
        kappa = self.params['kappa']
        beta = self.params['beta']
        
        M_pi_y = np.zeros((T, T))
        for t in range(T):
            for s in range(t, T):
                M_pi_y[t, s] = kappa * (beta ** (s - t))
        return M_pi_y

    def get_taylor_rule(self):
        """
        Taylor Rule: i_t = phi_pi * pi_t
        Returns M_i_pi (d_i / d_pi).
        Simple diagonal matrix.
        """
        return np.eye(self.T) * self.params['phi_pi']

    def get_fisher_equation(self):
        """
        Fisher Equation: r_t = i_t - pi_{t+1}
        Returns shift matrix S for expected inflation (d_r / d_pi).
        Note: d_r / d_i is Identity.
        """
        S = np.zeros((self.T, self.T))
        for t in range(self.T - 1):
            S[t, t+1] = 1.0
        return S
