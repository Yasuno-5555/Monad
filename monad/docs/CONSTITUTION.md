# MonadLab Project Constitution

This document codifies the implicit knowledge and philosophy governing the development of MonadLab. All future features and refactoring must adhere to these five core principles.

## 1. The Glass Box Rule (Transparency)
**"No Black Boxes."**
*   **Rule**: Any feature that produces an **Output** must also provide the **Decomposition** of that output.
*   **Example**: If Welfare Analysis shows a gain for borrowers, `.decompose()` must allow the user to see if it's due to inflation tax or wage increases.
*   **Reason**: In economic modeling, accountability and explainability are more important than raw speed.

## 2. The Democratization Rule (Accessibility)
**"8GB VRAM Limit."**
*   **Rule**: Features must run on consumer hardware (e.g., gaming PCs with 8GB VRAM).
*   **Constraint**: Do not implement features that require supercomputing clusters. Use techniques like Float32 precision or Sparse Solvers (GMRES) to fit within constraints.
*   **Reason**: Economic analysis must be liberated from exclusive institutions. Anyone should be able to verify FRB policy from their home.

## 3. The 10-Line Rule (Usability)
**"The Fluent Interface."**
*   **Rule**: No matter how complex the simulation, the user code must fit within **10 lines** of fluent, readable logic.
*   **Pattern**: Avoid configuration hell. Use method chaining: `m.shock(...).solve().plot()`.
*   **Reason**: Do not interrupt the User's "speed of thought". Maximize time spent thinking about economics, not code.

## 4. The Distribution-First Rule (Philosophy)
**"Who Pays?"**
*   **Rule**: Aggregate outputs (GDP, Inflation) are insufficient. Always provide **Distribution** and **Inequality** perspectives.
*   **Requirement**: Default plots must include heatmaps, quintile analysis, or distributional impacts.
*   **Reason**: Monad is HANK (Heterogeneous Agent New Keynesian), not RANK. Our purpose is to reveal "who wins and who loses" behind the aggregates.

## 5. The Immutability Rule (Reproducibility)
**"Scientific Integrity."**
*   **Rule**: Results are frozen in time with their context.
*   **Requirement**: parameters, Git Hash, and Random Seeds must be inextricably linked to the output. No overwriting of past results without explicit intent.
*   **Reason**: A result is only scientific if it can be perfectly reproduced 1 year later by a stranger (or yourself).
