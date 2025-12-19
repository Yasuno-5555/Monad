#include <pybind11/pybind11.h>
#include <Eigen/Dense>

namespace py = pybind11;

int add(int i, int j) {
    Eigen::Vector2d v;
    v << i, j;
    return (int)v.sum();
}

PYBIND11_MODULE(monad_core_test, m) {
    m.def("add", &add);
}
