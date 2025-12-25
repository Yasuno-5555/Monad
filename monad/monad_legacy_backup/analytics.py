import matplotlib.pyplot as plt
import numpy as np

def plot_irf(results, variables=None, title="Impulse Response", show=True):
    """
    Plot Impulse Response Functions.
    Args:
        results: dict of time series
        variables: list of keys to plot (default: Y, C, i, pi)
    """
    if variables is None:
        variables = ['Y', 'C_agg', 'i', 'pi']
        
    T = len(results[variables[0]])
    t = np.arange(T)
    
    fig, axes = plt.subplots(len(variables), 1, figsize=(8, 2*len(variables)), sharex=True)
    if len(variables) == 1: axes = [axes]
    
    for ax, var in zip(axes, variables):
        if var in results:
            data = results[var]
            # Convert to percent if small?
            if np.max(np.abs(data)) < 0.2: # Heuristic
                data = data * 100
                ylabel = "% Dev"
            else:
                ylabel = "Level"
                
            ax.plot(t, data, lw=2)
            ax.axhline(0, color='k', lw=0.5, ls='--')
            ax.set_ylabel(f"{var} ({ylabel})")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{var} not found", ha='center')
            
    axes[-1].set_xlabel("Quarters")
    fig.suptitle(title)
    plt.tight_layout()
    if show: plt.show()
    return fig

def plot_multiplier_decomposition(decomp_res, title="Multiplier Decomposition"):
    """
    Plot stacked bar chart of Direct vs Indirect effects.
    decomp_res: {direct: vec, indirect: vec}
    """
    direct = decomp_res['direct']
    indirect = decomp_res['indirect']
    T = len(direct)
    t = np.arange(T)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(t, direct, label='Direct (PE)', alpha=0.7)
    ax.bar(t, indirect, bottom=direct, label='Indirect (GE)', alpha=0.7)
    
    total = direct + indirect
    ax.plot(t, total, 'k--', label='Total Effect')
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def compare_scenarios(scenario_dict, variable='Y', title="Scenario Comparison"):
    """
    Compare a single variable across multiple scenarios.
    scenario_dict: { 'Scenario Name': result_dict, ... }
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for name, res in scenario_dict.items():
        if variable in res:
            data = res[variable]
            # Heuristic scaling
            if np.max(np.abs(data)) < 0.2: data = data * 100
            ax.plot(data, label=name, lw=2)
            
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"{title}: {variable}")
    ax.set_ylabel("% Deviation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_inequality_heatmap(heatmap_vec, grid, title="Consumption Response Heatmap"):
    """
    Plot heatmap of consumption response on (m, a) plane.
    Needs grid object to reshape.
    """
    # This requires reconstructing 2D shape from flat vector
    # nm = grid.N_m, na = grid.N_a.
    # Assuming z aggregated or specific z?
    # Usually heatmaps are shown for a specific income state or averaged.
    # For now, just placeholder or simple reshaping if dimensions known.
    pass
