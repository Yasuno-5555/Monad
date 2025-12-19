"""
Canonical Example 2: HANK with ZLB
Demonstrates automatic nonlinear solver selection for binding constraints.
"""
from monad import Monad

# 1. Setup: Deep recession requiring ZLB logic
# We lower the Natural Rate (r*) deep enough to hit the bound.
m = Monad("presets/japan_zlb")

# 2. Solve with ZLB constraint
# The facade automatically wraps the Linear solver with Newton.
res = (
    m.shock("r_star", -0.02) # Extra negative shock
     .solve(zlb=True)
)

# 3. Verify
print(f"Determinacy: {res.determinacy()['status']}")
# Nominal rate 'i' should flatline at 0.0
res.plot(["i", "Y", "pi"], title="Liquidity Trap Scenario")
