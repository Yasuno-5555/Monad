"""
Generate GE visualization plots for README.

Creates publication-ready figures showing:
1. Equilibrium interest rate path (dr_m)
2. Aggregate consumption response (dC)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import os

# Configure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def generate_ge_plots():
    """Generate and save GE response plots."""
    
    # Load real data from GPU export
    csv_path = "gpu_jacobian.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        dB_irf = df['dB'].values[:50]
        dC_irf = df['dC'].values[:50]
        print(f"[Plot] Loaded {csv_path}")
    else:
        # Dummy data if no CSV
        t = np.arange(50)
        dB_irf = 9.0 * (0.88 ** t)
        dC_irf = 0.9 * (0.88 ** t)
        print("[Plot] Using dummy data")
    
    T = len(dB_irf)
    
    # Construct Toeplitz Jacobian
    from scipy.linalg import toeplitz
    J_B_rm = toeplitz(dB_irf, np.zeros(T))
    J_C_rm = toeplitz(dC_irf, np.zeros(T))
    
    # Apply 1% permanent debt shock
    shock = np.ones(T) * 0.01
    dr_m = np.linalg.solve(J_B_rm, shock)
    dC = J_C_rm @ dr_m
    
    # Convert to basis points for rates
    dr_m_bps = dr_m * 10000  # bps
    dC_pct = dC * 100  # percent
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Interest Rate Path
    ax1 = axes[0]
    ax1.plot(dr_m_bps, 'r-', linewidth=2.5, label='Equilibrium Rate')
    ax1.fill_between(range(T), 0, dr_m_bps, alpha=0.2, color='red')
    ax1.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax1.set_xlabel('Periods')
    ax1.set_ylabel('Interest Rate (bps)')
    ax1.set_title('Interest Rate Response to 1% Debt Shock', fontweight='bold')
    ax1.set_xlim(0, T-1)
    
    # Add annotation for impact
    ax1.annotate(f'+{dr_m_bps[0]:.1f} bps', 
                xy=(0, dr_m_bps[0]), xytext=(5, dr_m_bps[0] + 2),
                fontsize=10, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred'))
    
    # Right: Consumption Response
    ax2 = axes[1]
    ax2.plot(dC_pct, 'b-', linewidth=2.5, label='Consumption')
    ax2.fill_between(range(T), 0, dC_pct, alpha=0.2, color='blue')
    ax2.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax2.set_xlabel('Periods')
    ax2.set_ylabel('Consumption Change (%)')
    ax2.set_title('General Equilibrium Consumption Response', fontweight='bold')
    ax2.set_xlim(0, T-1)
    
    # Add annotation
    ax2.annotate(f'+{dC_pct[0]:.2f}%', 
                xy=(0, dC_pct[0]), xytext=(5, dC_pct[0] + 0.02),
                fontsize=10, color='darkblue',
                arrowprops=dict(arrowstyle='->', color='darkblue'))
    
    plt.tight_layout()
    
    # Save
    os.makedirs('docs/figures', exist_ok=True)
    fig.savefig('docs/figures/ge_debt_shock.png', dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("[Plot] Saved docs/figures/ge_debt_shock.png")
    
    plt.close(fig)
    
    # Also print stats for README
    print(f"\n=== GE Shock Statistics ===")
    print(f"Shock: +1% Permanent Debt Increase")
    print(f"  dr_m[0]:  +{dr_m_bps[0]:.1f} bps")
    print(f"  dr_m[10]: +{dr_m_bps[10]:.1f} bps")
    print(f"  dr_m[49]: +{dr_m_bps[49]:.1f} bps")
    print(f"  dC[0]:    +{dC_pct[0]:.3f}%")
    print(f"  Half-life: ~{np.argmax(dB_irf < dB_irf[0]/2)} periods")

if __name__ == "__main__":
    generate_ge_plots()
