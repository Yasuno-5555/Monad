"""
Canonical Example 3: Taylor Principle Violation
Demonstrates the "Consultant" role of Monad: diagnosing bad policy.
"""
from monad import Monad

# 1. Setup: Dovish central bank (phi_pi < 1.0 violates Taylor Principle)
m = Monad("us_normal").setup(phi_pi=0.8)

# 2. Attempt to Solve
# This should result in Indeterminacy or Instability.
try:
    res = m.shock("monetary", -0.01).solve()
    
    # Check what the Consultant thinks
    diagnosis = res.determinacy()
    print(f"Status: {diagnosis['status']}") 
    print(f"Note:   {diagnosis['notes']}")

except Exception as e:
    print(f"Thinking Failed: {e}")
