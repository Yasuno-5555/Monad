# Monad Engine v4.0 Usage Guide

## Workflow Overview

The Monad Engine v4.0 workflow consists of two phases:
1.  **Engine Phase (C++)**: High-performance computation of Micro-Jacobians ($J_{C,r}, J_{C,Y}$).
2.  **Lab Phase (Python)**: General Equilibrium solving and Policy Analysis using `monad` package.

---

## 1. Engine Phase (C++)

First, you must build and run the C++ core to generate the Jacobian data.

### 1.1 Parameters
Modify `test_model.json` to configure the economic environment:
- `beta`: Discount factor
- `r_m`, `r_a`: Steady-state interest rates
- `chi`: Adjustment cost

### 1.2 Build
Use CMake to compile the project.

```bash
mkdir build_phase3
cd build_phase3
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### 1.3 Run (Generate Data)
Execute the solver relative to the project root (to ensure JSON/CSV paths are correct).

```powershell
.\Release\MonadTwoAssetCUDA.exe
```

**Output Files:** (Generated in project root)
- `gpu_jacobian_R.csv`: Consumption response to Interest Rate shock.
- `gpu_jacobian_Z.csv`: Consumption response to Income shock (MPC).

---

## 2. Lab Phase (Python)

Once the CSVs are generated, use the **Monad Lab** to conduct research.

### 2.1 Basic Experiment
Run the monetary shock verification script:

```bash
python experiments/02_monetary_shock.py
```
This solves the NK-HANK General Equilibrium and plots Impulse Response Functions (IRFs).

### 2.2 Advanced Experiment (Forward Guidance)
Test the effect of future policy announcements:

```bash
python experiments/03_forward_guidance.py
```
This compares immediate rate cuts vs. future promises, verifying the "Anticipation Effect".

---

## 3. Customizing the Lab

You can create your own experiments by using the `monad` package.

### Example: Custom Fiscal Shock
Create a new file `experiments/my_fiscal_shock.py`:

```python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from monad.solver import NKHANKSolver

# Initialize Solver
solver = NKHANKSolver(
    path_R="../gpu_jacobian_R.csv",
    path_Z="../gpu_jacobian_Z.csv"
)

# Define Shock (e.g., Government Spending increases demand Y directly?)
# Note: Currently solver handles Monetary Policy (r).
# For Fiscal Policy, extend the DAG in monad/solver.py.
```

### Changing NK Parameters
You can tweak the New Keynesian block parameters when initializing the solver:

```python
params = {
    'kappa': 0.05,  # Stickier prices
    'beta': 0.99,
    'phi_pi': 1.5
}
solver = NKHANKSolver(..., params=params)
```
