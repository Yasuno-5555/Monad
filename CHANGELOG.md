# Changelog

All notable changes to this project will be documented in this file.

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
