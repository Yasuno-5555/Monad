#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

// #include "AnalyticalSolver.hpp"
// #include "Engine.hpp"
// #include "grid/MultiDimGrid.hpp"
// #include "kernel/TwoAssetKernel.hpp"
// #include "solver/TwoAssetSolver.hpp"
// #include "solver/OneAssetSolver.hpp"
// #include "aggregator/DistributionAggregator3D.hpp"
// #include "ssj/SsjSolver3D.hpp"
// #include "ssj/GeneralEquilibrium.hpp"
// #include "ssj/SparseMatrixBuilder.hpp"
// #include "InequalityAnalyzer.hpp"
// #include "MicroAnalyzer.hpp"
// #include "FiscalExperiment.hpp"
// #include "OptimalPolicy.hpp"
// #include "Params.hpp"
// #include "solver/GenericNewtonSolver.hpp"

namespace py = pybind11;

PYBIND11_MODULE(monad_core, m) {
    m.doc() = "Monad Engine Core - Debugging";
}
