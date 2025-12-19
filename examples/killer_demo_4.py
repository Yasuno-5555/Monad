"""
Killer Demo 4: Asset Bubbles & Crashes
======================================
Phase 4: The Tensor Engine (Belief-Distribution HANK)

Simulates a "Sentiment Shock" where belief distribution shifts.
- Rational HANK: No bubbles, asset prices reflect fundamentals.
- Belief HANK: Optimists drive asset prices above fundamental value.

Mocking the GPU Tensor Engine with NumPy for logical verification.
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.belief import BeliefGrid

class MockTensorEngine:
    """
    Simulates the 4D (m, a, z, b) dynamics.
    Logic:
    1. Households have beliefs b in [-0.02, 0.02].
    2. Optimists (b > 0) demand more assets (a).
    3. Sentiment Shock shifts distribution of b.
    """
    def __init__(self, belief_grid):
        self.bg = belief_grid
        self.n_b = belief_grid.n
        
        # Mock Distribution: Uniform over b initially
        # Shape: (n_b,) for simplicity (aggregating over a, m, z)
        self.dist_b = np.ones(self.n_b) / self.n_b
        
        # Asset Demand Function (Mock)
        # Demand(a) = Base + Elasticity * Belief
        # Optimists buy more.
        self.asset_demand_base = 1.0
        self.elasticity = 5.0 # High elasticity -> Bubble
        
    def step(self, sentiment_shock=0.0):
        """
        Evolve distribution and compute aggregate asset demand.
        sentiment_shock: Shift in mean belief
        """
        # 1. Update Distribution (Transition)
        # Apply transition matrix
        self.dist_b = self.dist_b @ self.bg.trans_mat
        
        # Apply Shock (Shift mass towards optimistic states)
        if sentiment_shock > 0:
            # Shift mass to right (higher b)
            shifted = np.zeros_like(self.dist_b)
            shift_strength = 0.2
            for i in range(self.n_b):
                if i < self.n_b - 1:
                    move = self.dist_b[i] * shift_strength
                    shifted[i] -= move
                    shifted[i+1] += move
            self.dist_b += shifted
        elif sentiment_shock < 0:
            # Shift mass to left (lower b) - CRASH
            shifted = np.zeros_like(self.dist_b)
            shift_strength = 0.4 # Panic is faster than greed
            for i in range(self.n_b):
                if i > 0:
                    move = self.dist_b[i] * shift_strength
                    shifted[i] -= move
                    shifted[i-1] += move
            self.dist_b += shifted
            
        # 2. Compute Aggregate Demand
        # Demand_i = 1.0 + 5.0 * b_i
        demands = self.asset_demand_base + self.elasticity * self.bg.nodes
        agg_demand = np.sum(demands * self.dist_b)
        
        return agg_demand

def main():
    print("=" * 60)
    print("  Killer Demo 4: Asset Bubbles & Crashes")
    print("  (Simulating Belief Dynamics on Tensor Engine)")
    print("=" * 60)
    
    # 1. Setup Belief Grid
    # Range +/- 2%, 5 points
    bg = BeliefGrid(range=(-0.02, 0.02), n=5, persistence=0.9)
    print(f"\n[1] Belief Grid: {bg}")
    print(f"    Nodes: {bg.nodes}")
    
    engine = MockTensorEngine(bg)
    
    # 2. Simulation Loop
    T = 60
    prices = []
    beliefs_mean = []
    
    print("\n[2] Simulating Timeline...")
    print("    t=0-10:  Steady State")
    print("    t=10-30: Irrational Exuberance (Sentiment Shock)")
    print("    t=30-40: The peak")
    print("    t=40:    The CRASH (Panic)")
    
    for t in range(T):
        shock = 0.0
        if 10 <= t < 30:
            shock = 1.0 # Pushing towards optimism
        elif t == 40:
            shock = -1.0 # Panic trigger
            
        demand = engine.step(shock)
        
        # Price equilibrium (Asset Supply fixed at 1.0)
        # Demand = Supply -> Price adjustment?
        # Simple definition: Price ~ Demand (for mock)
        price = demand 
        
        prices.append(price)
        mean_b = np.sum(engine.dist_b * bg.nodes)
        beliefs_mean.append(mean_b)
        
    # 3. Visualization
    print("\n[3] Plotting Bubble...")
    t_axis = np.arange(T)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Price
    color = 'tab:blue'
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Asset Price (Bubble)', color=color, fontweight='bold')
    ax1.plot(t_axis, prices, color=color, linewidth=3, label='Asset Price')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, alpha=0.3)
    
    # Highlight Phases
    ax1.axvspan(10, 30, color='yellow', alpha=0.1, label='Exuberance Phase')
    ax1.axvspan(40, 50, color='red', alpha=0.1, label='Crash Phase')
    
    # Mean Belief
    ax2 = ax1.twinx()
    color = 'tab:purple'
    ax2.set_ylabel('Mean Sentiment (Optimism)', color=color, fontweight='bold')
    ax2.plot(t_axis, beliefs_mean, color=color, linestyle='--', linewidth=2, label='Mean Belief')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title("Anatomy of a Bubble: Belief-Distribution Dynamics", fontsize=14)
    fig.tight_layout()
    
    output_path = os.path.join(os.path.dirname(__file__), "killer_demo_4_output.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n=> Figure saved: {output_path}")
    
    # Insight
    print("\n" + "=" * 60)
    print("  KEY INSIGHT:")
    print("  Prices detach from fundamentals due to shifting mass")
    print("  in the Belief Distribution (The 'Tensor' shifting).")
    print("=" * 60)

if __name__ == "__main__":
    main()
