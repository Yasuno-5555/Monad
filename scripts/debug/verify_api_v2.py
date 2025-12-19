"""
Verification Script for Monad v2.0.0 API
Checks that all key components are importable from the top-level package.
"""
try:
    print("Testing Imports...")
    from monad import (
        # Core
        Monad, MonadModel, Study, DeterminacyStatus,
        # Policy (Phase 2)
        PolicyBlock, Rules,
        # Welfare (Phase 3)
        WelfareAnalysis, WelfareAnalyzer, WelfareResult,
        # Belief (Phase 4)
        BeliefGrid,
        # Solver (Phase 5)
        JacobianFreeGMRES,
        NKHANKSolver, SOESolver,
        # Regime (Phase 1)
        regime, registry,
        # Info
        __version__
    )
    print(f"✅ Success! Monad v{__version__} is ready.")
    print("   Imported: Monad, PolicyBlock, WelfareAnalysis, BeliefGrid, JacobianFreeGMRES")

except ImportError as e:
    print(f"❌ Import Failed: {e}")
    exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)
