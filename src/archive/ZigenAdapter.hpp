#pragma once
/**
 * @file ZigenAdapter.hpp
 * @brief Bridge layer between MonadLab and Zigen library
 *
 * Provides type aliases and wrappers for Zigen functionality.
 * This adapter serves as a Facade, allowing future backend changes
 * (e.g., to Intel MKL) without affecting the rest of the codebase.
 *
 * Note: This file should NOT be included in CUDA compilation units.
 */

#ifndef __CUDACC__

#include <Zigen/Matrix.hpp>
#include <Zigen/Solvers.hpp>
#include <Zigen/Autodiff.hpp>
#include <vector>

namespace Monad {
namespace ZigenBridge {

// =============================================================================
// Type Aliases
// =============================================================================

/// Dynamic-sized matrix of doubles (OpenMP-accelerated)
using MatD = ::Zigen::Matrix<double, ::Zigen::Dynamic, ::Zigen::Dynamic>;

/// Fixed-size matrices for common sizes
using Mat3d = ::Zigen::Matrix<double, 3, 3>;
using Mat4d = ::Zigen::Matrix<double, 4, 4>;

/// Scalar dual number for forward-mode automatic differentiation
using Duald = ::Zigen::Autodiff::Dual<double>;
using Dualf = ::Zigen::Autodiff::Dual<float>;

// =============================================================================
// Multi-Variable Automatic Differentiation (ZigenDualV)
// =============================================================================

/**
 * @brief Dual number with vector-valued gradient for Jacobian computation.
 */
struct ZigenDualV {
    double val;
    std::vector<double> grad;

    ZigenDualV() : val(0.0) {}
    ZigenDualV(double v) : val(v) {}
    ZigenDualV(double v, size_t index, size_t total_vars) : val(v) {
        grad.assign(total_vars, 0.0);
        if (index < total_vars) grad[index] = 1.0;
    }
    ZigenDualV(double v, const std::vector<double>& g) : val(v), grad(g) {}

    // Helper to ensure gradient vectors have the same size during operations
    static std::vector<double> add_grads(const std::vector<double>& a, const std::vector<double>& b) {
        size_t n = std::max(a.size(), b.size());
        std::vector<double> res(n, 0.0);
        for(size_t i=0; i<a.size(); ++i) res[i] += a[i];
        for(size_t i=0; i<b.size(); ++i) res[i] += b[i];
        return res;
    }

    ZigenDualV operator+(const ZigenDualV& rhs) const {
        return {val + rhs.val, add_grads(grad, rhs.grad)};
    }

    ZigenDualV operator-(const ZigenDualV& rhs) const {
        size_t n = std::max(grad.size(), rhs.grad.size());
        std::vector<double> res(n, 0.0);
        for(size_t i=0; i<grad.size(); ++i) res[i] += grad[i];
        for(size_t i=0; i<rhs.grad.size(); ++i) res[i] -= rhs.grad[i];
        return {val - rhs.val, res};
    }

    ZigenDualV operator*(const ZigenDualV& rhs) const {
        size_t n = std::max(grad.size(), rhs.grad.size());
        std::vector<double> res(n, 0.0);
        for(size_t i=0; i<grad.size(); ++i) res[i] += grad[i] * rhs.val;
        for(size_t i=0; i<rhs.grad.size(); ++i) res[i] += val * rhs.grad[i];
        return {val * rhs.val, res};
    }

    ZigenDualV operator/(const ZigenDualV& rhs) const {
        double v2 = rhs.val * rhs.val;
        size_t n = std::max(grad.size(), rhs.grad.size());
        std::vector<double> res(n, 0.0);
        for(size_t i=0; i<grad.size(); ++i) res[i] += (grad[i] * rhs.val) / v2;
        for(size_t i=0; i<rhs.grad.size(); ++i) res[i] -= (val * rhs.grad[i]) / v2;
        return {val / rhs.val, res};
    }

    // Scalar operations (delegating to Dual-Dual logic)
    ZigenDualV operator+(double rhs) const { return *this + ZigenDualV(rhs); }
    ZigenDualV operator-(double rhs) const { return *this - ZigenDualV(rhs); }
    ZigenDualV operator*(double rhs) const { return *this * ZigenDualV(rhs); }
    ZigenDualV operator/(double rhs) const { return *this / ZigenDualV(rhs); }

    friend ZigenDualV operator+(double lhs, const ZigenDualV& rhs) { return ZigenDualV(lhs) + rhs; }
    friend ZigenDualV operator-(double lhs, const ZigenDualV& rhs) { return ZigenDualV(lhs) - rhs; }
    friend ZigenDualV operator*(double lhs, const ZigenDualV& rhs) { return ZigenDualV(lhs) * rhs; }
    friend ZigenDualV operator/(double lhs, const ZigenDualV& rhs) { return ZigenDualV(lhs) / rhs; }

    // Math functions for AD
    friend ZigenDualV pow(const ZigenDualV& x, double n) {
        double v = std::pow(x.val, n);
        std::vector<double> res_grad(x.grad.size());
        double deriv = n * std::pow(x.val, n - 1.0);
        for(size_t i=0; i<x.grad.size(); ++i) res_grad[i] = x.grad[i] * deriv;
        return {v, res_grad};
    }

    friend ZigenDualV abs(const ZigenDualV& x) {
        double v = std::abs(x.val);
        double s = (x.val >= 0) ? 1.0 : -1.0;
        std::vector<double> res_grad(x.grad.size());
        for(size_t i=0; i<x.grad.size(); ++i) res_grad[i] = x.grad[i] * s;
        return {v, res_grad};
    }
};

// =============================================================================
// Solver Wrappers
// =============================================================================

/**
 * @brief Solve sparse symmetric positive definite system using Conjugate Gradient
 *
 * @tparam MatrixType Matrix type supporting operator* with vector
 * @param A Symmetric positive definite matrix
 * @param b Right-hand side vector
 * @param tol Convergence tolerance (default: 1e-10)
 * @param maxIter Maximum iterations (default: 1000)
 * @return SolverResult containing solution and convergence info
 */
template <typename MatrixType>
auto solve_cg(const MatrixType& A, const std::vector<double>& b,
              double tol = 1e-10, size_t maxIter = 1000) {
    return ::Zigen::Solvers::cg(A, b, tol, maxIter);
}

/**
 * @brief Solve general sparse system using BiCGSTAB
 *
 * Suitable for non-symmetric matrices.
 *
 * @tparam MatrixType Matrix type supporting operator* with vector
 * @param A General square matrix
 * @param b Right-hand side vector
 * @param tol Convergence tolerance (default: 1e-10)
 * @param maxIter Maximum iterations (default: 1000)
 * @return SolverResult containing solution and convergence info
 */
template <typename MatrixType>
auto solve_bicgstab(const MatrixType& A, const std::vector<double>& b,
                    double tol = 1e-10, size_t maxIter = 1000) {
    return ::Zigen::Solvers::bicgstab(A, b, tol, maxIter);
}

// =============================================================================
// Matrix Utilities
// =============================================================================

/**
 * @brief Create Zigen matrix from nested std::vector (with copy)
 *
 * Note: For Phase 1, this performs a copy. Future optimization could
 * use zero-copy Map if Zigen adds that feature.
 */
inline MatD from_vector(const std::vector<std::vector<double>>& data) {
    if (data.empty()) return MatD(0, 0);

    size_t rows = data.size();
    size_t cols = data[0].size();
    MatD mat(rows, cols);

    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            mat(i, j) = data[i][j];
        }
    }
    return mat;
}

/**
 * @brief Create Zigen column vector from std::vector
 */
inline MatD from_vector(const std::vector<double>& data) {
    size_t n = data.size();
    MatD vec(n, 1);
    for (size_t i = 0; i < n; ++i) {
        vec(i, 0) = data[i];
    }
    return vec;
}

/**
 * @brief Convert Zigen matrix to nested std::vector
 */
inline std::vector<std::vector<double>> to_vector_2d(const MatD& mat) {
    std::vector<std::vector<double>> result(mat.rows());
    for (size_t i = 0; i < mat.rows(); ++i) {
        result[i].resize(mat.cols());
        for (size_t j = 0; j < mat.cols(); ++j) {
            result[i][j] = mat(i, j);
        }
    }
    return result;
}

/**
 * @brief Convert Zigen column vector to std::vector
 */
inline std::vector<double> to_vector(const MatD& mat) {
    std::vector<double> result(mat.rows());
    for (size_t i = 0; i < mat.rows(); ++i) {
        result[i] = mat(i, 0);
    }
    return result;
}

} // namespace ZigenBridge
} // namespace Monad

#endif // __CUDACC__
