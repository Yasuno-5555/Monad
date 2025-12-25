import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def visualize_two_asset():
    # 1. Load Data
    print("Loading data...")
    if not os.path.exists("policy_2asset.csv"):
        print("Error: policy_2asset.csv not found.")
        return
    if not os.path.exists("dist_2asset.csv"):
        print("Error: dist_2asset.csv not found.")
        return

    df_pol = pd.read_csv("policy_2asset.csv")
    df_dist = pd.read_csv("dist_2asset.csv")

    # Filter for a specific income state (e.g., Low Income vs High Income)
    # For simplicity, if multiple z exist, we might just look at the first one or aggregate.
    # Let's check unique z values
    z_vals = df_pol['z_val'].unique()
    print(f"Income states found: {z_vals}")
    
    # We will plot for the first z state found (usually lowest income)
    z_target = z_vals[0]
    print(f"Visualizing for z_val = {z_target}")
    
    df_pol_z = df_pol[df_pol['z_val'] == z_target]
    df_dist_z = df_dist[df_dist['z_val'] == z_target]
    
    # Pivot for Heatmaps (m on X-axis, a on Y-axis)
    # m_val vs a_val
    pivot_adjust = df_pol_z.pivot_table(index='a_val', columns='m_val', values='adjust_flag')
    pivot_dist = df_dist_z.pivot_table(index='a_val', columns='m_val', values='mass', fill_value=0)

    # 2. Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # --- Plot A: The Inaction Region (Policy) ---
    # adjust_flag = 0 (No Adjust) vs 1 (Adjust)
    sns.heatmap(pivot_adjust, ax=axes[0], cmap="Blues", cbar=False)
    axes[0].set_title(f"Adjustment Policy Function (z={z_target})\n(White=Inaction, Blue=Adjust)")
    axes[0].set_ylabel("Illiquid Asset (a)")
    axes[0].set_xlabel("Liquid Asset (m)")
    axes[0].invert_yaxis() # Usually low 'a' at bottom
    
    # Reduce tick frequency safely
    n_ticks_x = len(axes[0].get_xticks())
    if n_ticks_x > 10:
        step = max(1, n_ticks_x // 5)
        # Apply locator instead of manual list slicing to avoid mismatch
        import matplotlib.ticker as ticker
        axes[0].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axes[0].yaxis.set_major_locator(ticker.MaxNLocator(5))
        axes[1].xaxis.set_major_locator(ticker.MaxNLocator(5))
        axes[1].yaxis.set_major_locator(ticker.MaxNLocator(5))

    plt.tight_layout()
    plt.savefig("two_asset_diagnostic.png")
    print("Saved 'two_asset_diagnostic.png'. Check the Inaction Region!")

if __name__ == "__main__":
    visualize_two_asset()
