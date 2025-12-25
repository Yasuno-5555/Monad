import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_inequality_analysis():
    # Style settings for publication quality
    sns.set_theme(style="whitegrid", context="talk")
    # plt.rcParams['font.family'] = 'sans-serif' # or 'serif' for LaTeX look

    # 1. Load Data
    try:
        df_irf = pd.read_csv("irf_groups.csv")
        df_heat = pd.read_csv("heatmap_sensitivity.csv")
    except FileNotFoundError:
        print("Error: CSV files not found. Please export 'irf_groups.csv' and 'heatmap_sensitivity.csv' from C++.")
        return

    # --- Figure 1: The Winners & Losers (Time Series) ---
    fig1, ax1 = plt.subplots(figsize=(12, 7))
    
    # Plot lines with distinct styles
    ax1.plot(df_irf['time'], df_irf['debtors'], label='Debtors (a < 0)', color='#d62728', linewidth=3, linestyle='-')
    ax1.plot(df_irf['time'], df_irf['bottom50'], label='Bottom 50% Wealth', color='#ff7f0e', linewidth=2.5, linestyle='--')
    ax1.plot(df_irf['time'], df_irf['aggregate'], label='Aggregate Economy', color='black', linewidth=1.5, alpha=0.6)
    ax1.plot(df_irf['time'], df_irf['top10'], label='Top 10% Wealth', color='#1f77b4', linewidth=2.5, linestyle='-.')

    ax1.set_title("Consumption Response to 1% Rate Hike (Monetary Shock)", pad=20, fontsize=18, fontweight='bold')
    ax1.set_ylabel("% Change in Consumption", fontsize=14)
    ax1.set_xlabel("Quarters", fontsize=14)
    ax1.axhline(0, color='black', linewidth=1)
    ax1.legend(frameon=True, fancybox=True, framealpha=0.9)
    ax1.set_xlim(0, 20) # Show first 5 years
    
    # Add annotation arrow for the Fisher Channel effect
    # (Assuming debtors drop the most)
    if not df_irf.empty:
      min_val = df_irf['debtors'].min()
      min_time = df_irf['debtors'].idxmin()
      ax1.annotate('Fisher Channel Impact\n(Debt Service Burden)', 
                  xy=(min_time, min_val), xytext=(min_time+5, min_val-0.005),
                  arrowprops=dict(facecolor='black', shrink=0.05))

    plt.tight_layout()
    plt.savefig("inequality_winners_losers.png", dpi=300)
    print("Saved 'inequality_winners_losers.png'")

    # --- Figure 2: The Sensitivity Heatmap (Who reacts most?) ---
    # Pivot data: X=m, Y=a, Z=dC/dr
    pivot_heat = df_heat.pivot_table(index='a_val', columns='m_val', values='dC_dr')
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Use a diverging colormap (Red = Cut Consumption, Blue = Increase)
    # Center at 0 to distinguish winners and losers
    vmax = np.abs(pivot_heat.values).max() if not pivot_heat.empty else 1.0
    sns.heatmap(pivot_heat, ax=ax2, cmap="vlag_r", center=0, 
                cbar_kws={'label': 'Sensitivity dC/dr'},
                xticklabels=5, yticklabels=5) # Sparsify ticks

    ax2.set_title("Consumption Sensitivity Surface (t=0)", pad=20, fontsize=18, fontweight='bold')
    ax2.set_xlabel("Liquid Asset (m)")
    ax2.set_ylabel("Illiquid Asset (a)")
    ax2.invert_yaxis() # Standard economics convention: low 'a' at bottom
    
    # Highlight the HtM region
    if not pivot_heat.empty:
       ax2.add_patch(plt.Rectangle((0, pivot_heat.shape[0]-5), 5, 5, fill=False, edgecolor='red', lw=2, linestyle='--'))
       ax2.text(2, pivot_heat.shape[0]-6, "Hand-to-Mouth\nHotspot", color='red', fontsize=10, ha='left')

    plt.tight_layout()
    plt.savefig("inequality_heatmap.png", dpi=300)
    print("Saved 'inequality_heatmap.png'")

if __name__ == "__main__":
    plot_inequality_analysis()
