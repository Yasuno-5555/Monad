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


def build_labor_process(rho: float, sigma_eps: float, n_emp: int,
                        u_rate: float = 0.05, 
                        replacement_rate: float = 0.4,
                        sep_rate: float = None,
                        find_rate: float = None) -> tuple:
    """
    Build income process with explicit unemployment state.
    
    The state space is: [Unemployed, Emp_1, Emp_2, ..., Emp_n]
    - Index 0 = Unemployed (receives replacement_rate * mean_wage)
    - Index 1..n = Employed states (Rouwenhorst discretization)
    
    Parameters:
    -----------
    rho : float
        AR(1) persistence for employed productivity
    sigma_eps : float
        Innovation std dev for employed productivity
    n_emp : int
        Number of employed productivity states
    u_rate : float
        Steady-state unemployment rate (default 5%)
    replacement_rate : float
        Unemployment benefit as fraction of mean wage (default 40%)
    sep_rate : float, optional
        Job separation rate per quarter. If None, calibrated from u_rate.
    find_rate : float, optional
        Job finding rate per quarter. If None, calibrated from u_rate.
    
    Returns:
    --------
    z_grid : np.ndarray
        Income grid [n_emp + 1], with z[0] = replacement_rate
    Pi : np.ndarray
        Transition matrix [n_emp + 1, n_emp + 1]
    is_unemployed : np.ndarray
        Boolean mask indicating unemployment state
    """
    # 1. Build employed Rouwenhorst process
    z_emp, Pi_emp = rouwenhorst(rho, sigma_eps, n_emp)
    
    # 2. Calibrate transition rates
    # In steady state: u_rate * find_rate = (1 - u_rate) * sep_rate
    # Given u_rate, we need one more parameter to pin down both.
    # Standard calibration: job finding rate ~ 0.45/quarter (Shimer 2005)
    if find_rate is None and sep_rate is None:
        # Assume find_rate = 0.45 (typical US calibration)
        find_rate = 0.45
        sep_rate = u_rate * find_rate / (1 - u_rate)
    elif find_rate is None:
        find_rate = sep_rate * (1 - u_rate) / u_rate
    elif sep_rate is None:
        sep_rate = u_rate * find_rate / (1 - u_rate)
    
    # 3. Build full transition matrix
    n_total = n_emp + 1  # Unemployed + Employed states
    Pi_full = np.zeros((n_total, n_total))
    
    # Unemployed -> Employed (uniform entry into employment)
    # P(U -> Ej) = find_rate * (1/n_emp)  for each employed state j
    Pi_full[0, 0] = 1 - find_rate  # Stay unemployed
    Pi_full[0, 1:] = find_rate / n_emp  # Find job (uniform across states)
    
    # Employed -> Unemployed or stay employed
    for i in range(n_emp):
        i_full = i + 1  # Offset by 1 for unemployment state
        
        # Separation shock
        Pi_full[i_full, 0] = sep_rate
        
        # Conditional on staying employed, follow Rouwenhorst
        for j in range(n_emp):
            j_full = j + 1
            Pi_full[i_full, j_full] = (1 - sep_rate) * Pi_emp[i, j]
    
    # 4. Build income grid
    # z[0] = replacement_rate (unemployment benefit)
    # z[1:] = employed productivity (already normalized to mean 1)
    z_grid = np.zeros(n_total)
    z_grid[0] = replacement_rate  # Unemployment benefit
    z_grid[1:] = z_emp  # Employed productivity
    
    # Create unemployment indicator
    is_unemployed = np.zeros(n_total, dtype=bool)
    is_unemployed[0] = True
    
    # 5. Verify transition matrix (rows sum to 1)
    row_sums = Pi_full.sum(axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums: {row_sums}"
    
    # 6. Compute stationary distribution and verify unemployment rate
    evals, evecs = np.linalg.eig(Pi_full.T)
    stat_dist = evecs[:, np.isclose(evals, 1.0)][:, 0].real
    stat_dist /= stat_dist.sum()
    
    implied_u_rate = stat_dist[0]
    print(f"[Labor Process] Calibrated: sep_rate={sep_rate:.4f}, find_rate={find_rate:.4f}")
    print(f"[Labor Process] Implied u_rate={implied_u_rate:.4f} (target={u_rate:.4f})")
    
    return z_grid, Pi_full, is_unemployed


def build_countercyclical_labor_process(rho: float, sigma_eps: float, n_emp: int,
                                         u_rate_ss: float = 0.05,
                                         u_rate_recession: float = 0.10,
                                         replacement_rate: float = 0.4) -> dict:
    """
    Build two transition matrices for normal times vs recession.
    
    This enables counter-cyclical unemployment risk where job separation
    rates increase during recessions.
    
    Returns:
    --------
    dict with:
        'z_grid': Income grid
        'Pi_normal': Transition matrix in normal times
        'Pi_recession': Transition matrix in recession
        'is_unemployed': Boolean mask
    """
    # Normal times
    z_grid, Pi_normal, is_unemp = build_labor_process(
        rho, sigma_eps, n_emp, u_rate_ss, replacement_rate
    )
    
    # Recession (higher unemployment)
    _, Pi_recession, _ = build_labor_process(
        rho, sigma_eps, n_emp, u_rate_recession, replacement_rate
    )
    
    return {
        'z_grid': z_grid,
        'Pi_normal': Pi_normal,
        'Pi_recession': Pi_recession,
        'is_unemployed': is_unemp
    }

