#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "AnalyticalSolver.hpp"

namespace py = pybind11;

// Wrapper function to handle Python <-> C++ interface
// - Converts std::vector<double> to UnifiedGrid
// - Handles reference parameter r_guess by returning it
double solve_steady_state_py(const std::vector<double>& nodes, double r_guess, double beta, double sigma) {
    UnifiedGrid grid(nodes);
    double r = r_guess;
    AnalyticalSolver::solve_steady_state(grid, r, beta, sigma);
    return r;
}

PYBIND11_MODULE(monad_core, m) {
    m.doc() = "Monad Engine Core Module"; // Optional module docstring

    m.def("solve_steady_state", &solve_steady_state_py, "Solve steady state interest rate",
          py::arg("grid_nodes"), py::arg("r_guess"), py::arg("beta"), py::arg("sigma"));
          
    // Optional: Expose GridGenerator for convenience if needed
    // But currently user only asked for AnalyticalSolver::solve_steady_state
}
