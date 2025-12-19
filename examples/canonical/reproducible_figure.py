"""
Canonical Example 4: Reproducible Science
Demonstrates how to freeze thoughts for publication.
"""
from monad import Monad
import os

# 1. Generate Result
# We use a specific parameter set we want to preserve.
m = Monad("us_normal")
res = (
    m.setup(kappa=0.05) # Flatter Phillips Curve
     .shock("monetary", -0.01)
     .solve()
)

# 2. Export Evidence
# This creates 'figure_1.csv' AND 'figure_1.meta.json'
res.export("figure_1.csv")

# 3. Verification (for the skeptical)
if os.path.exists("figure_1.meta.json"):
    print("Evidence secured: Metadata sidecar generated.")
    # In a real paper, you would commit this json with your plot.
