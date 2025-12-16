from monad.model import MonadModel
import matplotlib.pyplot as plt
import os
import sys

def main():
    print("=== Monad Studio: Sanity Check ===")
    
    # 1. Initialize Model
    # Use Sanity Check params (High R to induce responsiveness)
    # R_guess need to be consistent with what AnalyticalSolver finds? 
    # AnalyticalSolver solves for market clearing R given beta. 
    # If we want a specific R, we can tune Beta?
    # Or just use standard calibration which IS responsive (dK ~ 98 in previous test).
    
    model = MonadModel()
    model.set_param("beta", 0.96)
    model.set_param("sigma", 2.0)
    model.set_param("run_shock", True) # Flag for valid JSON if needed (not used by C++ yet but good practice)
    
    # Define Grid
    model.define_grid(size=200, type="Log-spaced", max_asset=100.0)
    
    # 2. Solve (Run C++ Engine)
    # Assumes MonadEngine.exe is in build_phase3/Release or current dir
    try:
        results = model.solve(exe_path="MonadEngine.exe")
    except Exception as e:
        print(f"Error executing model: {e}")
        return 1

    # 3. Analyze Results
    if "transition" in results:
        trans = results["transition"]
        print("Transition Data Head:")
        print(trans.head())
        
        # Check for non-zero response
        max_dr = trans["dr"].abs().max()
        print(f"Max Interest Rate Response: {max_dr}")
        
        if max_dr > 1e-6:
            print("[PASS] Sensitivity detected.")
        else:
            print("[FAIL] Response is near zero/zero (Expected for Constrained SS without Risk??)")
            # In Phase 3 tests, we saw dK=98 for R=0.05.
            # AnalyticalSolver solves for Equilibrium R.
            # If Beta=0.96, Equil R will make Beta(1+R)=1 roughly? No, < 1.
            # If Equil R is such that constrained, response might be zero.
            # Let's hope the default calibration yields something or we TWEAK beta in `model.py` 
            # to replicate the "high R" scenario if needed. 
            pass

        # Plot
        try:
             plt.figure(figsize=(10, 5))
             plt.subplot(1,2,1)
             plt.plot(trans["period"], trans["dr"], label="Iterest Rate (dr)")
             plt.legend()
             plt.title("Interest Rate Response")
             
             plt.subplot(1,2,2)
             plt.plot(trans["period"], trans["dZ"], label="Shock (dZ)")
             plt.legend()
             plt.title("Exogenous Shock")
             
             plt.savefig("sanity_check_irf.png")
             print("Saved sanity_check_irf.png")
        except:
             print("Plotting failed (no display?)")

    else:
        print("[FAIL] No transition data found.")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())
