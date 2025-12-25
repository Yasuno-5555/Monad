from monad.model import MonadModel
import pandas as pd
import subprocess

print("--- Verifying Redistribution (v1.6) ---")

model = MonadModel()
model.set_param("beta", 0.985)
model.set_param("sigma", 2.0)
model.set_param("alpha", 0.33)
model.set_param("r_guess", 0.015)
model.set_risk(0.96, 0.20, 7)
model.define_grid(size=200, type="Log-spaced", max_asset=200.0)

# Set Fiscal Params: Progressive Tax
# tau = 0.15 (Progressive)
model.set_fiscal(lambda_=0.9, tau=0.15, transfer=0.0) 

# Run solve (Steady State)
try:
    print("Solving Steady State with Progressive Tax...")
    model.solve()
except:
    pass

# Run engine manually to see stdout for Transition
exe_path = "MonadEngine.exe"
print(f"Running {exe_path}...")
result = subprocess.run([exe_path, "model_config.json"], capture_output=True, text=True)

for line in result.stdout.splitlines():
    if "[SSJ]" in line or "[Debug]" in line:
        print(line)

# Check CSV
try:
    df = pd.read_csv("transition_nk.csv")
    print("\nTail of Transition Path:")
    print(df.tail(10))
    
    # Check Max Response
    max_resp = df["dr"].abs().max()
    print(f"Max Output Gap Response: {max_resp}")
    
except Exception as e:
    print(f"Error reading CSV: {e}")
