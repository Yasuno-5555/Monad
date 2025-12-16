import numpy as np

def rouwenhorst(rho, sigma_eps, n):
    """
    Discretize AR(1) process log(z') = rho*log(z) + eps using Rouwenhorst method.
    Returns: z_grid (exp space), Pi (transition matrix)
    """
    if n == 1:
        return np.array([1.0]), np.array([[1.0]])

    p = (1 + rho) / 2
    
    # Base case for n=2
    Pi = np.array([[p, 1-p], [1-p, p]])
    
    # Recursively build Pi for n > 2
    for i in range(2, n):
        z_curr = np.zeros((i + 1, i + 1))
        
        # Top-left block
        z_curr[:-1, :-1] += p * Pi
        # Top-right block
        z_curr[:-1, 1:] += (1 - p) * Pi
        # Bottom-left block
        z_curr[1:, :-1] += (1 - p) * Pi
        # Bottom-right block
        z_curr[1:, 1:] += p * Pi
        
        # Normalize rows (Rouwenhorst trick to keep sum=1)
        z_curr[1:-1, :] /= 2
        Pi = z_curr

    # Construct grid
    # Std dev of the AR(1) process
    sigma_z = sigma_eps / np.sqrt(1 - rho**2)
    psi = np.sqrt(n - 1) * sigma_z
    
    # Log-grid is evenly spaced between [-psi, psi]
    x_grid = np.linspace(-psi, psi, n)
    z_grid = np.exp(x_grid)
    
    # Normalize z so mean is 1.0 (optional but recommended for HANK)
    # Stationary distribution to compute mean
    # For Rouwenhorst, stationary dist is Binomial(n-1, 0.5)
    # But let's compute it numerically to be safe and general
    evals, evecs = np.linalg.eig(Pi.T)
    stat_dist = evecs[:, np.isclose(evals, 1.0)][:, 0].real
    stat_dist /= stat_dist.sum()
    
    mean_z = np.sum(z_grid * stat_dist)
    z_grid /= mean_z
    
    return z_grid, Pi
