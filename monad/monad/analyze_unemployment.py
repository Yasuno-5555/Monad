"""
v1.7 Unemployment Analysis: "The Pain of Unemployment Check"
Verifies that the unemployment state correctly affects:
1. Consumption gap between employed and unemployed
2. Hand-to-Mouth rate among unemployed
3. Precautionary savings effects
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_unemployment_steady_state(result_dir: str = "."):
    """Analyze steady state with unemployment for distributional effects."""
    
    ss_path = Path(result_dir) / "steady_state.csv"
    if not ss_path.exists():
        print("[Error] steady_state.csv not found. Run model.solve() first.")
        return None
    
    df = pd.read_csv(ss_path)
    print(f"Loaded {len(df)} rows from steady_state.csv")
    print(f"Columns: {df.columns.tolist()}")
    
    # The CSV structure is: asset, z_idx, consumption, next_asset, distribution
    # Check if we have z_idx column
    z_col = None
    for col in ['z_idx', 'income_idx', 'z']:
        if col in df.columns:
            z_col = col
            break
    
    if z_col is None:
        print("[Warning] No income state column found. Inferring from structure...")
        # Infer: if we have n_assets * n_z rows, z alternates
        config_path = Path(result_dir) / "model_config.json"
        if config_path.exists():
            import json
            config = json.load(open(config_path))
            n_z = config['income_process']['n_z']
            z_grid = config['income_process']['z_grid']
            n_a = len(df) // n_z
            
            # Add z_idx column
            df['z_idx'] = np.repeat(np.arange(n_z), n_a)
            df['z_val'] = np.repeat(z_grid, n_a)
            z_col = 'z_idx'
            print(f"Inferred structure: {n_a} assets × {n_z} income states")
    
    # Analysis
    print("\n" + "="*60)
    print("=== v1.7 Unemployment Analysis ===")
    print("="*60)
    
    # 1. Consumption Gap
    unemployed = df[df[z_col] == 0]
    employed = df[df[z_col] > 0]
    
    # Weighted mean by distribution mass
    if 'distribution' in df.columns:
        C_u = np.average(unemployed['consumption'], weights=unemployed['distribution'])
        C_e = np.average(employed['consumption'], weights=employed['distribution'])
        D_u = unemployed['distribution'].sum()
        D_e = employed['distribution'].sum()
    else:
        C_u = unemployed['consumption'].mean()
        C_e = employed['consumption'].mean()
        D_u = len(unemployed) / len(df)
        D_e = len(employed) / len(df)
    
    print(f"\n1. CONSUMPTION GAP")
    print(f"   Unemployed (z=0) mean consumption: {C_u:.4f}")
    print(f"   Employed (z>0) mean consumption:   {C_e:.4f}")
    print(f"   Consumption drop upon unemployment: {100 * (1 - C_u/C_e):.2f}%")
    
    # 2. Hand-to-Mouth Rate
    # HtM = asset < threshold (e.g., 1 month of consumption)
    htm_threshold = C_u * 0.25  # 1/4 of quarterly consumption ≈ 1 month
    
    if 'distribution' in df.columns:
        htm_u = unemployed[unemployed['asset'] < htm_threshold]['distribution'].sum() / D_u
        htm_e = employed[employed['asset'] < htm_threshold]['distribution'].sum() / D_e
    else:
        htm_u = len(unemployed[unemployed['asset'] < htm_threshold]) / len(unemployed)
        htm_e = len(employed[employed['asset'] < htm_threshold]) / len(employed)
    
    print(f"\n2. HAND-TO-MOUTH RATE (asset < {htm_threshold:.4f})")
    print(f"   Among Unemployed: {100 * htm_u:.2f}%")
    print(f"   Among Employed:   {100 * htm_e:.2f}%")
    
    # 3. Population Distribution
    print(f"\n3. POPULATION DISTRIBUTION")
    print(f"   Unemployed share: {100 * D_u:.2f}%")
    print(f"   Employed share:   {100 * D_e:.2f}%")
    
    # 4. Asset Distribution by Employment Status
    if 'distribution' in df.columns:
        A_u = np.average(unemployed['asset'], weights=unemployed['distribution'])
        A_e = np.average(employed['asset'], weights=employed['distribution'])
    else:
        A_u = unemployed['asset'].mean()
        A_e = employed['asset'].mean()
    
    print(f"\n4. MEAN ASSETS")
    print(f"   Unemployed: {A_u:.4f}")
    print(f"   Employed:   {A_e:.4f}")
    print(f"   Asset gap:  {100 * (1 - A_u/A_e):.2f}% lower for unemployed")
    
    # 5. MPC Approximation (rough: 1 - a'/a ratio for poor)
    if 'next_asset' in df.columns:
        poor = df[df['asset'] < df['asset'].quantile(0.25)]
        if len(poor) > 0 and 'distribution' in df.columns:
            mpc_approx = 1 - (poor['next_asset'] / (poor['asset'] + poor['consumption'])).mean()
            print(f"\n5. MPC APPROXIMATION (Bottom 25%)")
            print(f"   Approximate MPC: {mpc_approx:.3f}")
    
    print("\n" + "="*60)
    
    # Return summary dict
    return {
        'C_unemployed': C_u,
        'C_employed': C_e,
        'consumption_gap_pct': 100 * (1 - C_u/C_e),
        'htm_unemployed_pct': 100 * htm_u,
        'htm_employed_pct': 100 * htm_e,
        'unemployment_rate': 100 * D_u,
        'A_unemployed': A_u,
        'A_employed': A_e
    }


def plot_consumption_by_employment(result_dir: str = "."):
    """Plot consumption distribution by employment status."""
    df = pd.read_csv(Path(result_dir) / "steady_state.csv")
    
    # Infer z_idx if needed (same logic as above)
    if 'z_idx' not in df.columns:
        import json
        config = json.load(open(Path(result_dir) / "model_config.json"))
        n_z = config['income_process']['n_z']
        n_a = len(df) // n_z
        df['z_idx'] = np.repeat(np.arange(n_z), n_a)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Consumption by z
    ax1 = axes[0]
    for z in df['z_idx'].unique():
        subset = df[df['z_idx'] == z]
        label = "Unemployed" if z == 0 else f"Employed (z={z})"
        color = 'red' if z == 0 else plt.cm.Blues(0.3 + 0.15*z)
        ax1.plot(subset['asset'], subset['consumption'], label=label, color=color, 
                 linewidth=2 if z == 0 else 1)
    
    ax1.set_xlabel("Assets")
    ax1.set_ylabel("Consumption")
    ax1.set_title("Consumption Policy by Employment Status")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Distribution
    ax2 = axes[1]
    if 'distribution' in df.columns:
        unemployed = df[df['z_idx'] == 0]
        employed = df[df['z_idx'] > 0]
        
        ax2.bar(unemployed['asset'], unemployed['distribution'], width=0.5, alpha=0.7, 
                color='red', label='Unemployed')
        # Aggregate employed
        emp_agg = employed.groupby('asset')['distribution'].sum()
        ax2.bar(emp_agg.index, emp_agg.values, width=0.5, alpha=0.5, 
                color='blue', label='Employed (all)')
        
        ax2.set_xlabel("Assets")
        ax2.set_ylabel("Distribution Mass")
        ax2.set_title("Wealth Distribution by Employment")
        ax2.legend()
        ax2.set_xlim(0, 10)  # Focus on lower wealth
    
    plt.tight_layout()
    plt.savefig("unemployment_analysis.png", dpi=150)
    print("[Analysis] Saved unemployment_analysis.png")
    return fig


if __name__ == "__main__":
    summary = analyze_unemployment_steady_state(".")
    if summary:
        print("\n=== Summary Dict ===")
        for k, v in summary.items():
            print(f"  {k}: {v:.2f}")
        
        plot_consumption_by_employment(".")
