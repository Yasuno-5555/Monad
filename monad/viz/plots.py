import matplotlib.pyplot as plt
import numpy as np

def plot_impulse_responses(results, title="Impulse Responses"):
    """Standard 3-panel plot for Y, pi, r"""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Output
    ax[0].plot(results['dY'] * 100, color='#1f77b4', lw=2)
    ax[0].set_title("Output Gap ($Y$)")
    ax[0].set_ylabel("% Deviation")
    
    # Inflation
    ax[1].plot(results['dpi'] * 100, color='#d62728', lw=2)
    ax[1].set_title("Inflation ($\pi$)")
    
    # Rate
    ax[2].plot(results['dr'] * 10000, color='#2ca02c', lw=2)
    ax[2].set_title("Real Interest Rate ($r$)")
    ax[2].set_ylabel("Basis Points")
    
    for a in ax:
        a.grid(True, alpha=0.3)
        a.axhline(0, color='k', ls=':', lw=1)
        a.set_xlabel("Quarters")
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig("impulse_responses.png")
    print("Impulse response plot saved to impulse_responses.png")
    # plt.show()

def plot_decomposition(solver, results):
    """
    Decomposes consumption response into Direct vs Indirect effects.
    Requires solver instance to access Jacobians.
    """
    dY = results['dY']
    dr = results['dr']
    
    # Calculate components
    dC_direct   = solver.backend.J_C_r @ dr
    dC_indirect = solver.backend.J_C_y @ dY
    dC_total    = dC_direct + dC_indirect
    
    t = np.arange(len(dY))
    
    plt.figure(figsize=(10, 6))
    plt.bar(t, dC_direct, label='Direct (Intertemporal Subst.)', color='#1f77b4', alpha=0.7)
    plt.bar(t, dC_indirect, bottom=dC_direct, label='Indirect (Income Effect)', color='#d62728', alpha=0.7)
    plt.plot(t, dC_total, label='Total Response', color='k', lw=2, ls='--')
    
    plt.title("Consumption Decomposition: Direct vs Indirect Channels")
    plt.ylabel("% Deviation")
    plt.xlabel("Quarters")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("decomposition_refactored.png")
    print("Decomposition plot saved to decomposition_refactored.png")
    # plt.show()
