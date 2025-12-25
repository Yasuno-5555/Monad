
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class MonadPlotter:
    def __init__(self, style="whitegrid"):
        sns.set_theme(style=style)
        
    def plot_macro_responses(self, runner, vars=None, title="Macroeconomic Responses"):
        """Plot Grid of Macro IRFs comparing scenarios"""
        if vars is None:
            vars = ["dY", "dC", "dN", "dpi", "dw", "dreal_r"]
            
        n_vars = len(vars)
        cols = 3
        rows = (n_vars + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_vars > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, var in enumerate(vars):
            ax = axes[i]
            df = runner.get_comparison_df(var)
            if not df.empty:
                # Scale by 100 for percentage
                (df * 100).plot(ax=ax, linewidth=2)
                ax.set_title(var + " (%)")
                ax.grid(True, alpha=0.3)
                ax.set_xlabel("Quarters")
            else:
                ax.text(0.5, 0.5, f"No Data for {var}", ha='center')
        
        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        return fig

    def plot_inequality_wedge(self, runner, metric="C_bottom50"):
        """Plot inequality metric comparison"""
        # Need to handle inequality extraction in runner first or direct access
        # Assuming runner.results[label]["inequality"] exists
        
        data = {}
        for label, res in runner.results.items():
            if "inequality" in res:
                df = res["inequality"]
                if metric in df.columns:
                    data[label] = df[metric]
        
        df_comp = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        if not df_comp.empty:
            df_comp.plot(ax=ax, linewidth=2)
            ax.set_title(f"Inequality Dynamics: {metric}")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
        return fig
    
    def plot_fiscal_decomposition(self, runner):
        """Plot Debt and Tax paths"""
        return self.plot_macro_responses(runner, vars=["dB", "dT"], title="Fiscal Dynamics")
