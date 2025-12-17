import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def verify_and_plot():
    print("--- Monad v4.0 Jacobian Verification ---")
    
    # 1. Load Data
    try:
        if not os.path.exists("gpu_jacobian_R.csv"):
            print("[ERROR] gpu_jacobian_R.csv not found.")
            return
        if not os.path.exists("gpu_jacobian_Z.csv"):
            print("[ERROR] gpu_jacobian_Z.csv not found.")
            return
            
        df_R = pd.read_csv("gpu_jacobian_R.csv")
        df_Z = pd.read_csv("gpu_jacobian_Z.csv")
    except Exception as e:
        print(f"[ERROR] Failed to read CSV files: {e}")
        return

    # 2. Key Metrics Extraction
    # dC/dr at t=0 (Substitution Effect)
    if len(df_R) > 0:
        dc_dr_0 = df_R['dC'].iloc[0]
        check_r = 'OK' if dc_dr_0 < 0 else 'FAIL'
        print(f"Metrics Check:")
        print(f"  dC/dr (t=0): {dc_dr_0:.6f} [{check_r}] -> Should be Negative")
    else:
        print("[ERROR] gpu_jacobian_R.csv is empty")

    # dC/dY at t=0 (Marginal Propensity to Consume - MPC)
    if len(df_Z) > 0:
        dc_dy_0 = df_Z['dC'].iloc[0]
        check_z = 'OK' if dc_dy_0 > 0 else 'FAIL'
        print(f"  dC/dY (t=0): {dc_dy_0:.6f} [{check_z}] -> Should be Positive (MPC)")
    else:
        print("[ERROR] gpu_jacobian_Z.csv is empty")

    # 3. Visualization
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Interest Rate Jacobian (d/dr)
    if len(df_R) > 0:
        ax[0].plot(df_R['dC'], label='Consumption (dC)', color='blue')
        ax[0].plot(df_R['dB'], label='Liquid Assets (dB)', color='green', linestyle='--')
        ax[0].set_title(f"Interest Rate Jacobian (J_r)\nSubstitution Effect dominates if dC < 0")
        ax[0].axhline(0, color='black', linewidth=0.8)
        ax[0].legend()
        ax[0].grid(True, alpha=0.3)

    # Plot 2: Income Jacobian (d/dY)
    if len(df_Z) > 0:
        ax[1].plot(df_Z['dC'], label='Consumption (dC)', color='red')
        ax[1].plot(df_Z['dB'], label='Liquid Assets (dB)', color='orange', linestyle='--')
        ax[1].set_title(f"Income Jacobian (J_Y)\nKeynesian Multiplier Base if dC > 0\nMPC = {dc_dy_0:.3f}")
        ax[1].axhline(0, color='black', linewidth=0.8)
        ax[1].legend()
        ax[1].grid(True, alpha=0.3)

    plt.suptitle("Monad v4.0: GPU Jacobian Validation", fontsize=16)
    plt.tight_layout()
    plt.savefig("jacobian_verification.png")
    print("Plot saved to jacobian_verification.png")
    # plt.show() # Non-interactive mode for agent

if __name__ == "__main__":
    verify_and_plot()
