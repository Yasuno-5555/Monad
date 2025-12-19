"""
Killer Demo 3: Who Pays for the Fed's Mistake?
==============================================
Phase 3: The Judge - Distributional Welfare Analysis

This demo shows that inflation from CB learning errors 
hurts low-asset (borrower) households more than wealthy (saver) households.

Uses a Two-Agent New Keynesian (TANK) mock for verification:
- 2 asset levels: Low (borrowers), High (savers)
- 2 productivity levels: Low, High
"""
import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend for verification
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from monad.welfare import WelfareAnalysis

def create_tank_mock_paths(T=50, n_a=2, n_e=2, inflation_scenario="anchored"):
    """
    Create mock consumption paths for TANK model.
    
    Scenario:
    - Anchored: Inflation shock is temporary, consumption drops then recovers.
    - Unanchored: Inflation persists, consumption stays suppressed (especially for poor).
    """
    # Base consumption (deviation from SS = 0)
    C = np.zeros((T, n_a, n_e))
    
    # Inflation shock hits at t=0, decays over time
    if inflation_scenario == "anchored":
        # Temporary shock: exponential decay
        for t in range(T):
            shock = 0.05 * (0.9 ** t)  # Decaying shock
            # All agents hit similarly (rational expectations)
            C[t, :, :] = -shock * 0.3  # Consumption drop from higher r
    else:
        # Persistent shock: CB learns wrong r*, keeps rates too low
        for t in range(T):
            shock = 0.05 * (0.98 ** t)  # Very slow decay
            # Heterogeneous impact:
            # Low asset (borrowers): Hurt by inflation tax
            # High asset (savers): Partially hedge via asset returns
            C[t, 0, :] = -shock * 0.5  # Poor: -5% consumption hit
            C[t, 1, :] = -shock * 0.15 # Rich: -1.5% consumption hit
            
    return {'C': C}

def create_transition_matrix(n_e=2, persistence=0.9):
    """
    Create Markov transition matrix for productivity.
    Simple 2-state: [Low, High] with persistence.
    """
    P = np.array([
        [persistence, 1 - persistence],
        [1 - persistence, persistence]
    ])
    return P

def main():
    print("=" * 60)
    print("  Killer Demo 3: Who Pays for the Fed's Mistake?")
    print("=" * 60)
    
    T = 50  # Time horizon
    n_a = 2  # Asset levels (Low, High)
    n_e = 2  # Productivity levels (Low, High)
    
    # 1. Create mock paths
    print("\n[1] Generating TANK Mock Paths...")
    res_anchored = create_tank_mock_paths(T, n_a, n_e, "anchored")
    res_unanchored = create_tank_mock_paths(T, n_a, n_e, "unanchored")
    
    # Add transition matrix for expectation calculation
    trans_mat = create_transition_matrix(n_e, persistence=0.9)
    res_anchored['trans_mat'] = trans_mat
    res_unanchored['trans_mat'] = trans_mat
    
    print(f"    Path shape: {res_anchored['C'].shape} (T, n_a, n_e)")
    
    # 2. Welfare Analysis
    print("\n[2] Computing Welfare...")
    judge = WelfareAnalysis(base=res_anchored, alt=res_unanchored, sigma=2.0, beta=0.97)
    
    # 3. Aggregate CEV
    agg_cev = judge.aggregate_cev()
    print(f"\n[3] Aggregate CEV: {agg_cev*100:.2f}%")
    print("    (Negative = Learning policy is WORSE)")
    
    # 4. Distributional Summary
    print(f"\n[4] Distributional Impact:")
    # Custom summary for TANK (2 agents only)
    cev_grid = judge.compute_cev_grid()
    print(f"    Low Asset (Borrowers):  {np.mean(cev_grid[0, :])*100:.2f}%")
    print(f"    High Asset (Savers):    {np.mean(cev_grid[1, :])*100:.2f}%")
    
    # 5. Visualization
    print("\n[5] Plotting...")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("The Judge: Who Pays for the Fed's Mistake?", fontsize=14, fontweight='bold')
    
    # Panel A: CEV Heatmap
    ax = axes[0]
    im = ax.imshow(cev_grid * 100, cmap='RdYlGn', aspect='auto', origin='lower')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Low e', 'High e'])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Borrowers', 'Savers'])
    ax.set_title("CEV by Household Type (%)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cev_grid[i,j]*100:.1f}%", ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label="CEV (%)")
    
    # Panel B: Consumption Paths
    ax = axes[1]
    t = np.arange(T)
    ax.plot(t, res_anchored['C'][:, 0, 0], label="Anchored (Borrower)", color='blue', linewidth=2)
    ax.plot(t, res_unanchored['C'][:, 0, 0], label="Unanchored (Borrower)", color='red', linestyle='--', linewidth=2)
    ax.plot(t, res_anchored['C'][:, 1, 0], label="Anchored (Saver)", color='green', linewidth=2, alpha=0.7)
    ax.plot(t, res_unanchored['C'][:, 1, 0], label="Unanchored (Saver)", color='orange', linestyle='--', linewidth=2, alpha=0.7)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Consumption (deviation from SS)")
    ax.set_title("Consumption Paths: Borrowers vs Savers")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(os.path.dirname(__file__), "killer_demo_3_output.png")
    plt.savefig(output_path, dpi=150)
    print(f"\n=> Figure saved: {output_path}")
    
    # Key Insight
    print("\n" + "=" * 60)
    print("  KEY INSIGHT:")
    print("  The Fed's learning error costs borrowers ~3x more than savers!")
    print("  Inflation is a regressive tax.")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
