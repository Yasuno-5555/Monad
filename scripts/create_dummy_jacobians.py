import numpy as np
import os

def create_dummy():
    T = 50
    # Create economically valid mock Jacobians
    # J_C_r: dC/dr - Lower triangular (causal), negative (consumption falls when r rises)
    # Decay factor: impulse response decays over time
    J_C_r = np.zeros((T, T))
    for i in range(T):
        for j in range(i+1):
            decay = 0.85 ** (i - j)
            J_C_r[i, j] = -0.3 * decay # Negative, decaying
    
    # J_C_y: dC/dr - Output feedback (if Y rises, C rises due to income effect)
    # Smaller effect, diagonal dominant
    J_C_y = np.zeros((T, T))
    for i in range(T):
        J_C_y[i, i] = 0.6 # MPC out of income
        if i > 0:
            J_C_y[i, i-1] = 0.1 # Lag effect

    # Save as gpu_jacobian_R.csv (J_C_r) and gpu_jacobian_Z.csv (J_C_y, simplified naming)
    # Note: MonadModel expects these two files.
    
    wd = os.getcwd()
    print(f"Creating files in {wd}")
    np.savetxt("gpu_jacobian_R.csv", J_C_r, delimiter=",")
    np.savetxt("gpu_jacobian_Z.csv", J_C_y, delimiter=",")
    print("Done.")

if __name__ == "__main__":
    create_dummy()
