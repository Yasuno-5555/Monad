from monad.model import MonadModel
import subprocess

print("--- Verifying SSJ Partials (2D) ---")

model = MonadModel()
model.set_param("beta", 0.985)
model.set_param("sigma", 2.0)
model.set_param("alpha", 0.33)
model.set_param("A", 1.0)
model.set_param("r_guess", 0.015)
model.set_risk(0.96, 0.20, 7)
model.define_grid(size=200, type="Log-spaced", max_asset=200.0)

# Run solve (generated config and runs engine)
# We catch output from the tool log
try:
    model.solve()
except:
    pass

# We don't need to run it again, but if we want to capture cleanly:


# Run engine manually to see stdout
exe_path = "MonadEngine.exe"
print(f"Running {exe_path}...")
result = subprocess.run([exe_path, "model_config.json"], capture_output=True, text=True)

print("--- Engine Output ---")
for line in result.stdout.splitlines():
    if "[SSJ]" in line:
        print(line)

print("--- End Output ---")
if result.returncode != 0:
    print("Engine failed!")
    print(result.stderr)
