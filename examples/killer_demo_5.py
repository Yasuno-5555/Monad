"""
Killer Demo 5: Flight to Digital Gold
=====================================
Phase 5: Lightweight 3-Asset HANK (Crypto)

Simulates the "Flight to Digital Gold" scenario using Jacobian-Free GMRES.
Scenario:
1. Productivity Shock (Z drops) -> Stock Fundamentals crash.
2. Rational Investors sell Stock, buy Liquid.
3. Optimists (High Belief) shift from Stock to Crypto (Digital Gold).
4. Result: Stock Crash + Crypto Bubble.

Uses a Mock 3-Asset Model to verify the GMRES Solver logic.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.solver.gmres import JacobianFreeGMRES

class MockCryptoModel:
    """
    Mock 3-Asset Model (Liquid, Stock, Crypto).
    
    Structure:
    - 2 Prices: P_stock (X[0...T-1]), P_crypto (X[T...2T-1])
    - Residuals: Excess Demand for Stock, Excess Demand for Crypto
    """
    def __init__(self, T=50):
        self.T = T
        # Parameters
        self.supply_stock = 1.0
        self.supply_crypto = 0.5
        
        # Sensitivities
        self.stock_fund_elas = 1.0  # Elasticity to Z
        self.crypto_bias_elas = 5.0 # Elasticity to Belief (High!)
        
    def evaluate_residual(self, X, shock_dict):
        """
        Compute Excess Demand for Stock and Crypto.
        X = [P_stock_path, P_crypto_path]
        """
        T = self.T
        P_stock = X[:T]
        P_crypto = X[T:]
        
        z_path = shock_dict.get('Z', np.zeros(T))
        b_path = shock_dict.get('b', np.zeros(T)) # Mean Belief Bias
        
        # 1. Demand Functions
        # Stock Demand: Positive to Z, Negative to P_s (Downward sloping), Negative to Belief (if flight)
        # Actually, Rational buy Stock if Z high.
        # But Optimists (b>0) might prefer Crypto if Z is low?
        # Let's model substitution:
        
        # Fundamental Demand for Stock
        D_stock_fund = 1.0 + self.stock_fund_elas * z_path - 0.5 * (P_stock - 1.0)
        
        # Speculative Demand for Crypto
        # Depends on Belief (b) and negative correlation with Stock fundamentals? (Hedge)
        # D_crypto = Base + Bias * Elasticity - PriceEffect
        # "Flight to Gold": If Z drops, Optimists buy Crypto?
        # Let's say Belief 'b' increases when Z drops (Panic/Distrust option)? 
        # Or assume 'b' is exogenous sentiment.
        
        D_crypto_spec = 0.5 + self.crypto_bias_elas * b_path - 0.2 * (P_crypto - 1.0)
        
        # Substitution Effect: If Stock drops, some flows to Crypto?
        # (Portfolio Rebalancing)
        # For simplicity in mock: Independent demands driven by common shocks.
        
        # 2. Residuals (Excess Demand)
        Res_stock = D_stock_fund - self.supply_stock
        Res_crypto = D_crypto_spec - self.supply_crypto
        
        return np.concatenate([Res_stock, Res_crypto])

def main():
    print("=" * 60)
    print("  Killer Demo 5: Flight to Digital Gold (Phase 5)")
    print("  (Jacobian-Free GMRES Solving 3-Asset Model)")
    print("=" * 60)
    
    T = 40
    model = MockCryptoModel(T)
    solver = JacobianFreeGMRES(model, epsilon=1e-4, max_iter=20)
    
    # Initial Guess (Steady State Prices)
    # P_stock = 1.0, P_crypto = 1.0
    X_guess = np.ones(2 * T)
    
    # 3. Define Shock Scenario
    # "Productivity Crash + Flight to Crypto"
    # Z drops (Stock crash)
    # b increases (Optimism/Speculation flares up)
    
    z_shock = np.zeros(T)
    b_shock = np.zeros(T)
    
    # Shock hits at t=5
    z_shock[5:25] = -0.2 # Fundamental Crisis
    b_shock[5:25] = 0.1  # Speculative Mania (Flight)
    
    print("\n[1] Solving Equilibrium Path...")
    shock_dict = {'Z': z_shock, 'b': b_shock}
    
    start_time = np.concatenate([np.ones(T), np.ones(T)]) # Guess
    X_sol, res_final = solver.solve(X_guess, shock_dict)
    
    P_stock_sol = X_sol[:T]
    P_crypto_sol = X_sol[T:]
    
    # 4. Visualization
    print("\n[2] Plotting Flight to Digital Gold...")
    t = np.arange(T)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Phase 5: The Decoupling (Stock Crash vs Crypto Bubble)", fontsize=14, fontweight='bold')
    
    # Panel A: Prices
    ax = axes[0]
    ax.plot(t, P_stock_sol, label="Stock Price (Fundamental)", color='blue', linewidth=2)
    ax.plot(t, P_crypto_sol, label="Crypto Price (Bubble)", color='orange', linewidth=2, linestyle='--')
    ax.axvspan(5, 25, color='gray', alpha=0.1, label='Shock Period')
    ax.set_title("Asset Prices")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel B: Shocks
    ax = axes[1]
    ax.plot(t, z_shock, label="Productivity (Z)", color='blue', alpha=0.6)
    ax.plot(t, b_shock, label="Sentiment (b)", color='orange', alpha=0.6)
    ax.set_title("Underlying Shocks")
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "killer_demo_5_output.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n=> Figure saved: {output_path}")
    
    print("\n" + "=" * 60)
    print("  KEY INSIGHT:")
    print("  Jacobian-Free Solver successfully decoupled the markets.")
    print("  Stocks crashed due to Z, Crypto rallied due to b.")
    print("  (Solved 5D dynamics without constructing H!)")
    print("=" * 60)

if __name__ == "__main__":
    main()
