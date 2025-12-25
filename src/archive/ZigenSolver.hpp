#pragma once
/**
 * @file ZigenSolver.hpp
 * @brief High-level linear solver interface using Zigen backend
 *
 * Provides convenient wrappers for common linear algebra operations:
 * - LU decomposition for general systems
 * - Cholesky decomposition for SPD systems (faster for OLS, etc.)
 * - Iterative solvers for large sparse systems
 *
 * Note: This file should NOT be included in CUDA compilation units.
 */

#ifndef __CUDACC__

#include "../ZigenAdapter.hpp"
#include <stdexcept>

namespace Monad {

class ZigenSolver {
public:
    /**
     * @brief Solve linear system Ax = b using LU decomposition
     *
     * General-purpose solver for any square non-singular matrix.
     *
     * @param A Square matrix (n x n) as nested vectors
     * @param b Right-hand side vector (n)
     * @return Solution vector x
     * @throws std::runtime_error if matrix is singular
     */
    static std::vector<double> solve_linear(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b
    ) {
        if (A.empty() || A.size() != b.size()) {
            throw std::runtime_error("ZigenSolver: dimension mismatch");
        }

        size_t n = A.size();
        ZigenBridge::MatD mat(n, n);
        ZigenBridge::MatD rhs(n, 1);

        // Copy data to Zigen matrices
        for (size_t i = 0; i < n; ++i) {
            rhs(i, 0) = b[i];
            for (size_t j = 0; j < n; ++j) {
                mat(i, j) = A[i][j];
            }
        }

        // Solve using LU decomposition
        auto result = mat.solve(rhs);

        // Extract solution
        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = result(i, 0);
        }
        return x;
    }

    /**
     * @brief Solve SPD system Ax = b using Cholesky decomposition
     *
     * Faster than LU for symmetric positive definite matrices.
     * Ideal for least squares (X'X * beta = X'y) and covariance problems.
     *
     * @param A Symmetric positive definite matrix (n x n)
     * @param b Right-hand side vector (n)
     * @return Solution vector x
     * @throws std::runtime_error if matrix is not SPD
     */
    static std::vector<double> solve_spd(
        const std::vector<std::vector<double>>& A,
        const std::vector<double>& b
    ) {
        if (A.empty() || A.size() != b.size()) {
            throw std::runtime_error("ZigenSolver: dimension mismatch");
        }

        size_t n = A.size();
        ZigenBridge::MatD mat(n, n);
        ZigenBridge::MatD rhs(n, 1);

        for (size_t i = 0; i < n; ++i) {
            rhs(i, 0) = b[i];
            for (size_t j = 0; j < n; ++j) {
                mat(i, j) = A[i][j];
            }
        }

        // Solve using Cholesky (LLT) decomposition
        auto result = mat.solve_llt(rhs);

        std::vector<double> x(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = result(i, 0);
        }
        return x;
    }

    /**
     * @brief Solve linear system using Zigen Matrix directly
     *
     * More efficient when already working with Zigen matrices.
     *
     * @param A Zigen matrix
     * @param b Zigen column vector
     * @return Solution as Zigen column vector
     */
    static ZigenBridge::MatD solve(
        const ZigenBridge::MatD& A,
        const ZigenBridge::MatD& b
    ) {
        return A.solve(b);
    }

    /**
     * @brief Solve SPD system using Zigen Matrix directly
     */
    static ZigenBridge::MatD solve_llt(
        const ZigenBridge::MatD& A,
        const ZigenBridge::MatD& b
    ) {
        return A.solve_llt(b);
    }

    /**
     * @brief Compute matrix inverse
     *
     * @param A Square matrix
     * @return Inverse matrix A^(-1)
     */
    static std::vector<std::vector<double>> inverse(
        const std::vector<std::vector<double>>& A
    ) {
        if (A.empty()) {
            throw std::runtime_error("ZigenSolver: empty matrix");
        }

        size_t n = A.size();
        ZigenBridge::MatD mat(n, n);

        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                mat(i, j) = A[i][j];
            }
        }

        auto inv = mat.inverse();
        return ZigenBridge::to_vector_2d(inv);
    }

    /**
     * @brief Matrix multiplication C = A * B
     *
     * Uses OpenMP-accelerated GEMM when available.
     */
    static std::vector<std::vector<double>> matmul(
        const std::vector<std::vector<double>>& A,
        const std::vector<std::vector<double>>& B
    ) {
        auto matA = ZigenBridge::from_vector(A);
        auto matB = ZigenBridge::from_vector(B);
        auto matC = matA * matB;
        return ZigenBridge::to_vector_2d(matC);
    }
};

} // namespace Monad

#endif // __CUDACC__
