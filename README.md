# Monad Engine v4.0: NK-HANK Base

**The "Monad Lab" Release**

> *December 2025 - Full Heterogeneous Agent New Keynesian (HANK) Model with GPU Acceleration and Python Research Platform*

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)]()

## Overview

**Monad Engine** is a high-performance computational framework for solving Two-Asset HANK models. 
Version 4.0 integrates a **GPU-accelerated micro-foundation** with a **modular Python laboratory**, enabling complex macroeconomic experiments (like Forward Guidance and Fiscal Multipliers) in milliseconds.

### Key Features v4.0

1.  **GPU Income Jacobian ($J_{C,Y}$)**:
    *   Computes the sensitivity of consumption to income shocks directly on the GPU.
    *   Captures the "Indirect Effect" (Keynesian Multiplier) crucial for HANK dynamics.
    *   Jacobian computation time: **< 100ms** (RTX 4090).

2.  **Monad Lab (`monad/`)**:
    *   New modular Python package for policy research.
    *   Clean separation of Data (`core.py`), Theory (`blocks.py`), and Solution (`solver.py`).
    *   Plug-and-play New Keynesian blocks (NKPC, Taylor Rule, Fisher Equation).

3.  **Advanced Experiments**:
    *   **Consumption Decomposition**: Visualizing Direct (Substitution) vs Indirect (Income) channels.
    *   **Forward Guidance**: Verified "Anticipation Effects" of future policy announcements.

---

## Architecture

![Monad v4.0 Architecture](https://mermaid.ink/img/pako:eNp1ksFu2zAMhl9F0GkF7NBDd9htwIbtMOw6DNiKogiyM9WOLUWK5LhB3n3Ukr_FsBwSiQ_5-1M-qYqV0JJqvO7hVbGl8eZop-D94TBU-4-Hg9r_ePio33_tP-uPj6P6sP90_KjV98PhE4-vT0cNp4-Hj_pQjzoN-6-q_n_gQhVLCbXW4M15j4dChb2C0aLz4C00GkZ4M8Z4c-dMgeKdwYUzBYwGjHcGjDeG94Y03hTeG1J4Y0jhTSGFN2M03ozReNM5YzTeNM5UaLy5c6Zy463lzeXG28qby423lTeVG28tbyo33lrepL9eX268vby53Hh7eXP519vy5nLjbeXN5cbbypvLjbeVN5cbb8dvLjfext9cbrxdv7mn4P_6H_8X_73_1j9P_61_nv4z_63__v8r_g7_R_4O_4P_1j9v_61_3v4z_63__v87_gH_R_4B_4P_1v_A_2_4n_zPhP_R_0z8H_3PhP_R_0z8H_3PhP-z_4H4n_0PxP_sfyD-Z_8D8T_7H4j_2f9A_M_-B-J_9j8Q_7P_gfjf_Q_E_-5_IP53_wPxv_sfiP_d_0D87_4H4n_3PxD_u_-B-N_9D8T_7n8g_nf_A_G_-x-I_93_QPzv_gfjf_c_GP-7_8H43_0PjH93A7F_gQyMfw==)

```mermaid
graph TD
    subgraph GPU ["GPU Engine (C++/CUDA)"]
        K[Kernels] -->|Compute| J1[Jacobian J_Cr]
        K -->|Compute| J2[Jacobian J_CY]
        J1 --> CSV1[gpu_jacobian_R.csv]
        J2 --> CSV2[gpu_jacobian_Z.csv]
    end

    subgraph Lab ["Monad Lab (Python)"]
        CSV1 & CSV2 --> Core[monad/core.py]
        Core -->|Toeplitz Matrix| Solver[monad/solver.py]
        Blocks[monad/blocks.py] -->|NKPC / Taylor| Solver
        Solver -->|General Equilibrium| Exp[experiments/]
        Exp --> Viz[monad/plots.py]
    end
```

---

## Quick Start

### 1. Build the Engine (C++)
The C++ core must be built to generate the Jacobian matrices.

```bash
mkdir build_phase3
cd build_phase3
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### 2. Generate Data (Run Engine)
Execute the solver to produce the micro-Jacobians (`gpu_jacobian_R.csv`, `gpu_jacobian_Z.csv`).

```powershell
.\Release\MonadTwoAssetCUDA.exe
```

### 3. Run Experiments (Python)
Now use the **Monad Lab** to solve for General Equilibrium and analyze results.

```bash
# Experiment 1: Monetary Policy Shock (+25bps)
python experiments/02_monetary_shock.py

# Experiment 2: Forward Guidance (Rate Cut Promise)
python experiments/03_forward_guidance.py
```

---

## Directory Structure

```text
Monad/
├── src/                  # C++/CUDA GPU Core
│   ├── gpu/              # Custom CUDA Kernels (Dual EGM)
│   ├── solver/           # TwoAssetSolver logic
│   └── main_two_asset.cpp
├── monad/                # Python Research Package
│   ├── core.py           # Backend Data Loader
│   ├── blocks.py         # New Keynesian Equations
│   ├── solver.py         # GE Solver (DAG Assembly)
│   └── plots.py          # Visualization Tools
├── experiments/          # Research Scripts
│   ├── 02_monetary_shock.py
│   └── 03_forward_guidance.py
├── docs/                 # Documentation
└── gpu_jacobian_*.csv    # Interface Data
```

---

## Results

### Consumption Decomposition
The "Money Plot" showing how the **Indirect Income Effect** (Red) amplifies the **Direct Substitution Effect** (Blue).

![Decomposition](decomposition_refactored.png)

### Forward Guidance
Validating the **Anticipation Effect**: Consumption rises at $t=0$ in response to a rate cut announced for $t=4$.

![Forward Guidance](forward_guidance.png)

---

## License

MIT License

---

**Monad Engine Development Team**
*v4.0 Final Release - December 2025*
