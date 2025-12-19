# Monad Studio

Monad is a thinking library for macroeconomic models.

It unifies the workflow of setting up, shocking, solving, and comparing heterogeneous agent models (HANK) and their linear equivalents (RANK/NK).

This is largely AI-generated code. I cannot fully verify the results myself. Use at your own risk and please validate before citing.

## Why Monad?

*   **Thought-Speed Interaction**: No boilerplate. Just `Setup` -> `Shock` -> `Solve`.
*   **Auto-Solver Selection**: Monad automatically chooses between Linear SSJ, Newton, or Piecewise solvers based on your simulation context (e.g., ZLB).
*   **Credibility**: Every result carries a reproducibility fingerprint and metadata sidecar.

## The 10-Line "Killer" Demo

Evaluate policy counterfactuals and their welfare implications in seconds:

```python
from monad.model import MonadModel
from monad.welfare import WelfareResult

# 1. Base Model
m = MonadModel("bin/MonadEngine")
base = m.run()

# 2. Counterfactual: Weak Policy + ZLB Crisis
alt = (
    m.object("policy_rule").mutate(phi=0.8)
     .object("solver").toggle("zlb")
     .run()
)

# 3. Decision: Is the new policy better? (Consumption Equivalent Variation)
WelfareResult(alt).compare(WelfareResult(base))
```

This single block demonstrates:
1.  **Fluent Mutation**: `mutate()`, `toggle()`
2.  **Structural Change**: Switching regimes (ZLB) and parameters on the fly.
3.  **Normative Analysis**: Directly computing welfare losses.

## Quick Start

### Installation

```bash
# From PyPI (when published)
pip install monad-econ

# From source (development)
git clone https://github.com/monad-econ/monad.git
cd monad
pip install -e ".[dev]"
```

### Basic Usage

Run a standard US calibration with a monetary shock:

```python
from monad import Monad

# The Agent
m = Monad("us_normal")

# The Thought
res = (
    m.shock("monetary", -0.01)
     .solve()
)

# The Insight
res.plot("Y")
res.export("figure_1.csv") # Generates figure_1.meta.json automatically
```

With Command Line Interface:
```bash
python -m monad run us_normal --shock monetary:-0.01 --export result.csv
```

## Canonical Examples
See `examples/canonical/` for clean, copy-pasteable patterns.

| File | Concept |
|------|---------|
| [nk_basic.py](examples/canonical/nk_basic.py) | **Linear Thinking**, standard New Keynesian logic. |
| [hank_zlb.py](examples/canonical/hank_zlb.py) | **Nonlinear Thinking**, finding a way out of a Liquidity Trap. |
| [determinacy_fail.py](examples/canonical/determinacy_fail.py) | **Diagnosis**, identifying when policies violate stability. |
| [reproducible_figure.py](examples/canonical/reproducible_figure.py) | **Science**, exporting evidence with fingerprints. |

## Policy-as-Object (Phase 2)

Monad treats **policy rules as first-class objects** with internal state and learning capabilities.

```python
from monad.policy import Rules

# Create a Central Bank with Learning
fed = Rules.InertialTaylor(phi_pi=1.5, rho=0.8, name="Fed")
fed.learns("r_star", gain=0.05)  # CB learns the natural rate

# Attach to Model and Solve
m.attach(fed)
res = m.run(diff_inputs={"markup": 0.05})

# Inspect CB's Belief Path
print(res['Fed_r_star_belief'])
```

**Killer Demo 2** (`examples/killer_demo_2.py`) demonstrates the 1970s "Great Inflation" logic: how a Central Bank's misperception of `r*` can turn a temporary shock into persistent inflation.


Monad acts as a consultant, diagnosing the stability of your model.

### Determinacy Status

| Status | Meaning |
|--------|---------|
| `UNIQUE` | A stable, unique equilibrium was found. (Blanchard-Kahn satisfied or Newton converged) |
| `INDETERMINATE` | Multiple equilibria possible. (e.g., Passive Fiscal/Monetary mix) |
| `UNSTABLE` | No stable equilibrium. (e.g., Taylor Principle violation) |
| `UNKNOWN` | Solver completed but diagnostics are inconclusive. |

### Citing
```python
print(res.cite())
```

## Reproducibility
Every `export()` generates a sidecar file (`.meta.json`) containing:
*   Full parameter snapshot (Immutable)
*   Exact shock definitions
*   Solver selection logic used
*   `fingerprint` (SHA-256 hash)

---
*Monad Studio Engine v4.0*
