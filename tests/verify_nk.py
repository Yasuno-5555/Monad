from monad.model import MonadModel
import pandas as pd
import subprocess

print("--- Verifying NK Transition (v1.4) ---")

model = MonadModel()
model.set_param("beta", 0.985)
model.set_param("sigma", 2.0)
model.set_param("alpha", 0.33)
model.set_param("A", 1.0)
model.set_param("r_guess", 0.015)
model.set_risk(0.96, 0.20, 7)
model.define_grid(size=200, type="Log-spaced", max_asset=200.0)

# Run solve
try:
    model.solve()
except:
    pass

# Run engine manually to see stdout
exe_path = "MonadEngine.exe"
print(f"Running {exe_path}...")
result = subprocess.run([exe_path, "model_config.json"], capture_output=True, text=True)

# Parse output for IRF confirmation
nk_ran = False
for line in result.stdout.splitlines():
    if "Solving NK Transition Path" in line:
        nk_ran = True
    if "[SSJ]" in line or "[Debug]" in line:
        print(line)

print(f"NK Transition Ran: {nk_ran}")

# Check CSV
try:
    df = pd.read_csv("transition_nk.csv")
    print(df.head(10))
    print("\nTail:")
    print(df.tail(5))
    
    # Check if dY (dr column) is non-zero and decays
    max_resp = df["dr"].abs().max()
    print(f"Max Output Gap Response: {max_resp}")
    
except Exception as e:
    print(f"Error reading CSV: {e}")
