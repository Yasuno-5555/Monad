#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include "ssj/SsjSolver3D.hpp"
#include "FiscalExperiment.hpp"
#include "OptimalPolicy.hpp"
#include "MicroAnalyzer.hpp"
#include "InequalityAnalyzer.hpp"

namespace py = pybind11;

void init_minimal(py::module_ &m) {
    m.doc() = "Minimal Monad Core";
}

PYBIND11_MODULE(monad_core, m) {
    init_minimal(m);
}
