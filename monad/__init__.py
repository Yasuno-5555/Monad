"""
Monad: A Thinking Library for Macroeconomic Models
===================================================

The Monad Trilogy:
- Phase 1 (Engine): Regime-Endogenous Switching
- Phase 2 (Brain): Policy-as-Object with Learning
- Phase 3 (Judge): Distributional Welfare Analysis

Quick Start:
    from monad import Monad
    m = Monad("us_normal")
    res = m.shock("monetary", -0.01).solve()
    res.plot("Y")
"""

__version__ = "2.0.0"
__author__ = "Monad Team"

# Core API
from .facade import Monad, Study, DeterminacyStatus
from .model import MonadModel

# Phase 2: Policy Objects
from .policy import PolicyBlock, Rules

# Phase 3: Welfare Analysis
from .welfare import WelfareAnalysis, WelfareAnalyzer, WelfareResult

# Phase 4: Belief (The Tensor)
from .belief import BeliefGrid

# Phase 5: Jacobian-Free Solver
from .solver import NKHANKSolver, SOESolver, JacobianFreeGMRES
from .nonlinear import PiecewiseSolver

# Regime System (Advanced)
from .regime import regime, registry

__all__ = [
    # Core
    "Monad",
    "MonadModel",
    "Study",
    "DeterminacyStatus",
    # Policy
    "PolicyBlock",
    "Rules",
    # Welfare
    "WelfareAnalysis",
    "WelfareAnalyzer", 
    "WelfareResult",
    # Belief
    "BeliefGrid",
    # Solvers
    "NKHANKSolver",
    "SOESolver",
    "JacobianFreeGMRES",
    "PiecewiseSolver",
    # Regime
    "regime",
    "registry",
]
