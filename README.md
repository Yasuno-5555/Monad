# Monad Engine v6.0 "Izanagi"

> **Global Nonlinear Solver:** Newton–Raphson on Sequence Space  
> **Constraints:** Zero Lower Bound (ZLB) & Consumption–Export Disconnect  
> **Result:** Successful simulation of an *Export-led Recession* under Liquidity Trap conditions  
>
> **Important:** A CUDA-capable GPU is required to generate Jacobian matrices.  
> CPU-only environments can analyze **pre-generated (cached) results only**.


---

# Monad Engine v4.0 – The "Monad Lab" Upgrade  
*(Previous versions below)*

**Monad Lab Release**  
*December 2025 – GPU-Accelerated HANK Engine with Python Research Environment*

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://isocpp.org/std/the-standard)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![Status: Stable](https://img.shields.io/badge/Status-Stable-green.svg)]()

---

## Overview

**Monad Engine** is a high-performance computational framework for solving
**Heterogeneous Agent New Keynesian (HANK)** models using
**sequence-space Newton methods**.

The system combines:

- a **C++/CUDA GPU backend** for microeconomic Jacobian generation, and  
- a **modular Python research layer ("Monad Lab")** for nonlinear general equilibrium analysis.

The design emphasizes **clarity of model assumptions**, **reproducibility**, and
**fast experimentation** with advanced macroeconomic scenarios
(e.g. ZLB episodes, forward guidance, open-economy mechanisms).

---

## Core Design Philosophy

- The **C++ engine** is responsible for *high-cost microeconomic computations*
  (e.g. household Jacobians).
- The **Python layer** assembles macroeconomic blocks and solves
  nonlinear general equilibrium systems.
- Any component involving **GUI or interactive visualization**
  is considered *experimental* and **not part of the core engine**.

---

## GPU Requirement (Important)

⚠ **A CUDA-capable GPU is required.**

The current workflow assumes that Jacobian matrices are generated
using the GPU backend.

Specifically, the following files must exist:

- `gpu_jacobian_R.csv`
- `gpu_jacobian_Z.csv`

Without these files:

- the Python solvers will not run, and  
- GUI tools (if used) will fail to execute.

**CPU-only execution is not supported**, except for analysis of
previously generated results.

---

## Architecture Overview

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
        Core -->|Toeplitz Assembly| Solver[monad/solver.py]
        Blocks[monad/blocks.py] -->|NKPC / Taylor / Fisher| Solver
        Solver -->|Nonlinear GE| Exp[experiments/]
        Exp --> Viz[monad/plots.py]
    end
