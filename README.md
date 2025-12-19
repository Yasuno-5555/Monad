# Monad Studio

Monad is a thinking library for macroeconomic models.

It unifies the workflow of setting up, shocking, solving, and comparing heterogeneous agent models (HANK) and their linear equivalents (RANK/NK).

## Why Monad?

*   **Thought-Speed Interaction**: No boilerplate. Just `Setup` -> `Shock` -> `Solve`.
*   **Auto-Solver Selection**: Monad automatically chooses between Linear SSJ, Newton, or Piecewise solvers based on your simulation context (e.g., ZLB).
*   **Credibility**: Every result carries a reproducibility fingerprint and metadata sidecar.

## Quick Start
```bash
pip install monad  # (Hypothetical)
```

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

## Credibility & Determinacy

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
