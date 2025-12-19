"""
Canonical Example 1: Basic Linear New Keynesian
Demonstrates the simplest "thought process": shock -> linear solve -> plot.
"""
from monad import Monad

# 1. Initialize the Agent with a standard preset
m = Monad("us_normal")

# 2. Define the Thought Process
# "What happens if monetary policy tightens by 100bps?"
res = (
    m.shock("monetary", -0.01)
     .solve(nonlinear=False)
)

# 3. Insight
print(f"Impact on Output Gap: {res.data['Y'][0]:.4%}")
res.plot(["Y", "pi", "r"], title="Monetary Contraction (Linear)")
