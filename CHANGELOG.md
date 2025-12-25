# Changelog

All notable changes to this project will be documented in this file.

## [4.0.0] - 2025-12-26

### Added
- **Three-Asset Model**: Extended state space to $(m, a, h, z)$ with housing/physical capital.
- **Economic Complexity**: 
  - **Wealth Tax**: Non-linear progressive tax on total assets.
  - **Housing Frictions**: Fixed and quadratic adjustment costs for illiquid assets.
- **Monte Carlo Simulation**: 
  - Single-GPU simulation of 10,000 agents.
  - `curand` integration for stochastic income shocks.
  - Automated portfolio composition analysis.

### Changed
- **Full GPU Deployment**: 
  - Optimized CUDA kernels for 4D grids ($80 \times 80 \times 40 \times 5$).
  - Trilinear interpolation for policy functions.
  - Achieved ~200ms per iteration on RTX 3070.

## [3.0.0] - 2025-12-24

### Added
- **GPU Backend**: 
  - CUDA implementation of Bellman/Expectations solvers.
  - `__restrict__` and `__ldg()` optimizations for memory throughput.
  - Shared memory caching for grid arrays.
  - Parallel binary search for policy interpolation.

### Changed
- **Build System**: 
  - Stabilized `Zigen` integration with absolute paths.
  - Isolated `MonadCore` library for testing.

## [2.0.0] - 2024-12-16

### Added
- **Two-Asset HANK Model**: Full implementation with liquid and illiquid assets
- **Sequence Space Jacobian (SSJ)**: Auclert et al. (2021) methodology
  - `JacobianBuilder3D`: Dual number automatic differentiation
  - `FakeNewsAggregator`: Distribution perturbation calculation
  - `SsjSolver3D`: Block Jacobian construction
- **General Equilibrium Solver**: Market clearing with multiplier effects
- **Inequality Analyzer**: Group-specific consumption responses
  - Top 10% / Bottom 50% / Debtors decomposition
  - Spatial sensitivity heatmaps
- **Python Visualization Suite**: Publication-quality figures

### Changed
- Refactored grid system to `MultiDimGrid` for 3D state space
- Improved sparse matrix construction for distribution dynamics

## [1.8.0] - 2024-12-14

### Added
- Analysis suite with macro, fiscal, and inequality panels
- Unemployment support in income process

## [1.7.0] - 2024-12-12

### Added
- Wage Phillips Curve
- Unemployment dynamics

## [1.6.0] - 2024-12-10

### Added
- Progressive taxation
- Capital taxation

## [1.5.0] - 2024-12-08

### Added
- Fiscal policy block
- Contemporaneous fiscal rules

## [1.4.0] - 2024-12-06

### Added
- New Keynesian blocks
- Price stickiness

## [1.0.0] - 2024-12-01

### Added
- Initial release
- Basic EGM solver for Aiyagari model
- Distribution aggregator
