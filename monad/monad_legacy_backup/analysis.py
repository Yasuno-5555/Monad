"""
Monad Studio: Inequality Analysis Module
Visualizes distributional dynamics from HANK model output.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class MonadAnalysis:
    """Analyze and visualize HANK model inequality dynamics."""
    
    def __init__(self, result_dir: str = "."):
        self.result_dir = Path(result_dir)
        self.inequality_df = None
        self.transition_df = None
        self._load_data()
    
    def _load_data(self):
        """Load CSV output from C++ engine."""
        ineq_path = self.result_dir / "inequality_path.csv"
        trans_path = self.result_dir / "transition_nk.csv"
        
        if ineq_path.exists():
            self.inequality_df = pd.read_csv(ineq_path)
            print(f"[Analysis] Loaded {ineq_path}")
        else:
            print(f"[Warning] {ineq_path} not found")
            
        if trans_path.exists():
            self.transition_df = pd.read_csv(trans_path)
            print(f"[Analysis] Loaded {trans_path}")
        else:
            print(f"[Warning] {trans_path} not found")
    
    def plot_winners_losers(self, save_path: str = None) -> plt.Figure:
        """
        Plot consumption response: Top 10% vs Bottom 50%.
        
        The key insight: Under comprehensive taxation (Phase 2),
        the wealthy (Top 10%) should see dampened consumption gains.
        """
        if self.inequality_df is None:
            raise ValueError("No inequality data loaded")
        
        df = self.inequality_df
        
        # Compute percentage change from steady state (t=0)
        C_top10_pct = (df['C_top10'] - df['C_top10'].iloc[0]) / df['C_top10'].iloc[0] * 100
        C_bot50_pct = (df['C_bottom50'] - df['C_bottom50'].iloc[0]) / df['C_bottom50'].iloc[0] * 100
        C_total_pct = (df['C_total'] - df['C_total'].iloc[0]) / df['C_total'].iloc[0] * 100
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(df['period'], C_top10_pct, label="Top 10% (Wealthy)", color="crimson", linewidth=2)
        ax.plot(df['period'], C_bot50_pct, label="Bottom 50% (Poor)", color="royalblue", linewidth=2)
        ax.plot(df['period'], C_total_pct, label="Aggregate", color="gray", linestyle="--", alpha=0.7)
        
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Quarters", fontsize=12)
        ax.set_ylabel("% Change from Steady State", fontsize=12)
        ax.set_title("Distributional Consumption Response to Monetary Shock", fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[Analysis] Saved plot to {save_path}")
        
        return fig
    
    def plot_gini_dynamics(self, save_path: str = None) -> plt.Figure:
        """Plot Gini coefficient and Top 10% wealth share dynamics."""
        if self.inequality_df is None:
            raise ValueError("No inequality data loaded")
        
        df = self.inequality_df
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Gini coefficient
        ax1 = axes[0]
        ax1.plot(df['period'], df['A_gini'], color='purple', linewidth=2)
        ax1.set_xlabel("Quarters")
        ax1.set_ylabel("Asset Gini Coefficient")
        ax1.set_title("Wealth Inequality Dynamics")
        ax1.grid(True, alpha=0.3)
        
        # Top 10% share
        ax2 = axes[1]
        ax2.plot(df['period'], df['wealth_top10_share'] * 100, color='darkred', linewidth=2)
        ax2.set_xlabel("Quarters")
        ax2.set_ylabel("Top 10% Wealth Share (%)")
        ax2.set_title("Concentration at the Top")
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
            print(f"[Analysis] Saved plot to {save_path}")
        
        return fig
    
    def plot_transition_path(self, save_path: str = None) -> plt.Figure:
        """Plot interest rate path from transition solution."""
        if self.transition_df is None:
            raise ValueError("No transition data loaded")
        
        df = self.transition_df
        
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.plot(df['period'], df['dr'] * 100, color='green', linewidth=2)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlabel("Quarters", fontsize=12)
        ax.set_ylabel("Deviation from Steady State (bps)", fontsize=12)
        ax.set_title("Interest Rate Response (dr)", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150)
        
        return fig
    
    def summary(self) -> dict:
        """Return summary statistics of the transition."""
        if self.inequality_df is None:
            return {}
        
        df = self.inequality_df
        
        # Peak responses
        C_top10_chg = (df['C_top10'] - df['C_top10'].iloc[0]) / df['C_top10'].iloc[0] * 100
        C_bot50_chg = (df['C_bottom50'] - df['C_bottom50'].iloc[0]) / df['C_bottom50'].iloc[0] * 100
        
        return {
            "peak_C_top10_pct": C_top10_chg.max(),
            "peak_C_bottom50_pct": C_bot50_chg.max(),
            "gini_ss": df['A_gini'].iloc[0],
            "gini_peak": df['A_gini'].max(),
            "top10_share_ss": df['wealth_top10_share'].iloc[0] * 100,
        }


def compare_phases(phase1_dir: str, phase2_dir: str, save_path: str = None):
    """
    Compare Phase 1 (labor tax only) vs Phase 2 (comprehensive tax).
    
    This visualization demonstrates the stabilization mechanism:
    Phase 2 should show dampened Top 10% consumption response.
    """
    try:
        phase1 = MonadAnalysis(phase1_dir)
        phase2 = MonadAnalysis(phase2_dir)
    except Exception as e:
        print(f"[Error] Could not load both phases: {e}")
        return None
    
    if phase1.inequality_df is None or phase2.inequality_df is None:
        print("[Error] Missing inequality data for comparison")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Top 10% comparison
    ax1 = axes[0]
    df1, df2 = phase1.inequality_df, phase2.inequality_df
    
    C_top10_p1 = (df1['C_top10'] - df1['C_top10'].iloc[0]) / df1['C_top10'].iloc[0] * 100
    C_top10_p2 = (df2['C_top10'] - df2['C_top10'].iloc[0]) / df2['C_top10'].iloc[0] * 100
    
    ax1.plot(df1['period'], C_top10_p1, label="Phase 1 (Labor Tax)", color="red", linestyle="--")
    ax1.plot(df2['period'], C_top10_p2, label="Phase 2 (Comprehensive)", color="darkred", linewidth=2)
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel("Quarters")
    ax1.set_ylabel("% Change")
    ax1.set_title("Top 10% Consumption: Phase 1 vs Phase 2")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom 50% comparison
    ax2 = axes[1]
    C_bot50_p1 = (df1['C_bottom50'] - df1['C_bottom50'].iloc[0]) / df1['C_bottom50'].iloc[0] * 100
    C_bot50_p2 = (df2['C_bottom50'] - df2['C_bottom50'].iloc[0]) / df2['C_bottom50'].iloc[0] * 100
    
    ax2.plot(df1['period'], C_bot50_p1, label="Phase 1 (Labor Tax)", color="blue", linestyle="--")
    ax2.plot(df2['period'], C_bot50_p2, label="Phase 2 (Comprehensive)", color="darkblue", linewidth=2)
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_xlabel("Quarters")
    ax2.set_ylabel("% Change")
    ax2.set_title("Bottom 50% Consumption: Phase 1 vs Phase 2")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[Analysis] Saved comparison to {save_path}")
    
    return fig


if __name__ == "__main__":
    # Quick test
    analysis = MonadAnalysis(".")
    
    if analysis.inequality_df is not None:
        print("\n=== Summary ===")
        for k, v in analysis.summary().items():
            print(f"  {k}: {v:.4f}")
        
        analysis.plot_winners_losers("winners_losers.png")
        analysis.plot_gini_dynamics("gini_dynamics.png")
        print("\n[Done] Plots saved.")
    else:
        print("[Info] Run MonadEngine first to generate data.")
