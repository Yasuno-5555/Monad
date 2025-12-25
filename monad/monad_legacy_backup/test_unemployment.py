"""Test v1.7 unemployment income process"""
from monad.model import MonadModel

# Configure model with unemployment
model = MonadModel("v1.7_Unemployment_Test")
model.set_risk(rho=0.9, sigma_eps=0.2, n_z=5)  # 5 employed states
model.set_unemployment(u_rate=0.05, replacement_rate=0.4)  # Enable unemployment
model.set_fiscal(tau=0.15)  # Progressive tax
model.define_grid(size=200)

print("=== v1.7 Unemployment Test ===")
print(f"Unemployment Rate: 5%")
print(f"Replacement Rate: 40%")
print(f"Total income states: 6 (1 unemployed + 5 employed)")

# Solve
results = model.solve()

# Check results
if "steady_state" in results:
    ss = results["steady_state"]
    print(f"\nSteady State: {len(ss)} rows")
    print(f"z_idx range: {ss['z_idx'].min()} to {ss['z_idx'].max()}")
    
    # Check z_idx=0 (unemployed)
    unemployed = ss[ss['z_idx'] == 0]
    employed = ss[ss['z_idx'] > 0]
    
    print(f"\nUnemployed consumption (mean): {unemployed['consumption'].mean():.4f}")
    print(f"Employed consumption (mean): {employed['consumption'].mean():.4f}")
    
print("\n=== Test Complete ===")
