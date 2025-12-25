#pragma once
/**
 * @file ZigenNewtonSolver.hpp
 * @brief Multi-variable Newton-Raphson solver using Zigen AD and linear solvers
 *
 * This solver finds the root x such that f(x) = 0 for vector-valued functions.
 * It leverages Zigen's automatic differentiation for accurate Jacobian 
 * computation and Zigen's optimized linear algebra for the Newton step.
 */

#ifndef __CUDACC__

#include "../ZigenAdapter.hpp"
#include "ZigenSolver.hpp"
#include <functional>
#include <iostream>
#include <cmath>
#include <iomanip>

namespace Monad {

class ZigenNewtonSolver {
public:
    using Vector = std::vector<double>;
    
    /**
     * @brief Objective function type for the Newton solver.
     * 
     * Takes a vector of ZigenDualV and returns a vector of ZigenDualV.
     * This allows the solver to compute the value and Jacobian in one pass.
     */
    using ObjectiveFunc = std::function<std::vector<ZigenBridge::ZigenDualV>(const std::vector<ZigenBridge::ZigenDualV>&)>;

    struct SolverParams {
        double tol = 1e-8;
        int max_iter = 50;
        bool verbose = true;
    };

    /**
     * @brief Solve f(x) = 0 using Newton-Raphson method
     * 
     * @param func Objective function f(x)
     * @param initial_guess Initial estimate for x
     * @param params Solver parameters (tol, max_iter, verbose)
     * @return Solution vector x
     */
    static Vector solve(ObjectiveFunc func, Vector initial_guess, SolverParams params = {}) {
        Vector x = initial_guess;
        size_t n = x.size();

        if (params.verbose) {
            std::cout << "[NewtonSolver] Starting solver loop (n=" << n << ")" << std::endl;
        }

        for (int iter = 0; iter < params.max_iter; ++iter) {
            // 1. Prepare input vector with vector-gradient AD
            std::vector<ZigenBridge::ZigenDualV> x_dual(n);
            for (size_t i = 0; i < n; ++i) {
                x_dual[i] = ZigenBridge::ZigenDualV(x[i], i, n);
            }

            // 2. Evaluate function and compute Jacobian
            auto y_dual = func(x_dual);
            
            if (y_dual.size() != n) {
                throw std::runtime_error("NewtonSolver: dimension mismatch between x and f(x)");
            }

            Vector residual(n);
            std::vector<std::vector<double>> J(n, Vector(n));
            
            double sum_sq_err = 0.0;
            for (size_t i = 0; i < n; ++i) {
                residual[i] = y_dual[i].val;
                sum_sq_err += residual[i] * residual[i];
                for (size_t j = 0; j < n; ++j) {
                    J[i][j] = y_dual[i].grad[j];
                }
            }
            
            double norm = std::sqrt(sum_sq_err);
            if (params.verbose) {
                std::cout << "  Iter " << std::setw(2) << iter 
                          << " | Norm: " << std::scientific << std::setprecision(4) << norm << std::endl;
            }

            // Convergence check
            if (norm < params.tol) {
                if (params.verbose) std::cout << "  Converged!" << std::endl;
                return x;
            }

            // 3. Solve J * delta = -residual
            Vector neg_residual(n);
            for (size_t i = 0; i < n; ++i) neg_residual[i] = -residual[i];

            Vector delta;
            try {
                delta = ZigenSolver::solve_linear(J, neg_residual);
            } catch (const std::exception& e) {
                std::cerr << "  [ERROR] NewtonSolver: Linear solve failed at iter " << iter 
                          << " (" << e.what() << ")" << std::endl;
                return x; // Return current best guess
            }

            // 4. Update x
            for (size_t i = 0; i < n; ++i) {
                x[i] += delta[i];
            }
        }
        
        if (params.verbose) {
            std::cerr << "  [WARNING] NewtonSolver: Failed to converge within " 
                      << params.max_iter << " iterations." << std::endl;
        }
        return x;
    }
};

} // namespace Monad

#endif // __CUDACC__
