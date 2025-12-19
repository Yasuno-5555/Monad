"""
GMRES Solver (Jacobian-Free)
============================
Phase 5: Lightweight HANK

Solves H * dx = -Res without constructing the explicit Jacobian matrix H.
Uses Generalized Minimal Residual method (GMRES) with a Matrix-Free LinearOperator.

H * v approx (F(X + eps * v) - F(X)) / eps
"""
import numpy as np
from scipy.sparse.linalg import gmres, LinearOperator
import time

class JacobianFreeGMRES:
    """
    Solves for equilibrium transition path using Matrix-Free GMRES.
    """
    def __init__(self, model, epsilon=1e-5, tol=1e-5, max_iter=50):
        self.model = model
        self.epsilon = epsilon
        self.tol = tol
        self.max_iter = max_iter
        
    def solve(self, X_guess, shock_dict):
        """
        Solve F(X) = 0 for X.
        Uses Newton-GMRES:
          1. Compute Residual R = F(X_k)
          2. Solve J * dX = -R using GMRES (Matrix-Free)
          3. Update X_{k+1} = X_k + dX
        """
        X = X_guess.copy()
        
        print(f"Jacobian-Free GMRES Solver (eps={self.epsilon})")
        
        for it in range(self.max_iter):
            # 1. Evaluate Baseline Residual
            # F(X)
            res_base = self.model.evaluate_residual(X, shock_dict)
            err = np.max(np.abs(res_base))
            
            print(f"  Iter {it}: Max Res = {err:.2e}")
            if err < self.tol:
                print("  [CONVERGED]")
                return X, res_base
            
            # 2. Define Linear Operator (JVP)
            # J * v
            size = X.size
            
            def matvec(v):
                # J * v approx (F(X + eps*v) - F(X)) / eps
                # 1. Perturb
                X_perturbed = X + self.epsilon * v.reshape(X.shape)
                
                # 2. Evaluate
                res_perturbed = self.model.evaluate_residual(X_perturbed, shock_dict)
                
                # 3. Differencing
                jvp = (res_perturbed - res_base) / self.epsilon
                return jvp.ravel()
            
            op = LinearOperator((size, size), matvec=matvec, dtype=float)
            
            # 3. Call GMRES
            # Solve J * dX = -Res
            b = -res_base.ravel()
            
            # Use callback to monitor GMRES inner iterations
            def callback(rk):
                pass 
                # print(f"    gmres err: {np.linalg.norm(rk):.2e}")

            dX_flat, info = gmres(op, b, rtol=1e-1, maxiter=20, callback=callback)
            
            if info != 0:
                print(f"    GMRES failed/stopped (info={info})")
                
            # 4. Update
            dX = dX_flat.reshape(X.shape)
            X += dX
            
        return X, res_base
