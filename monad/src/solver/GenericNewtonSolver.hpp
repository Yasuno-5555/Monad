#pragma once
#include <functional>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

namespace Monad {

// Function pointer type: F(x) -> residuals
using ResidualFunc = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;

class GenericNewtonSolver {
public:
    struct Options {
        int max_iter = 100;
        double tol = 1e-6;
        double fd_eps = 1e-6;          // Finite Difference Step
        double damping_factor = 1.0;    // Initial damping
        int max_backtracking = 10;      // Max backtracking steps
        bool verbose = true;
    };

    static Eigen::VectorXd solve(ResidualFunc F, Eigen::VectorXd x0, Options opts = Options()) {
        Eigen::VectorXd x = x0;
        
        if (opts.verbose) {
            std::cout << "--- C++ GenericNewtonSolver ---" << std::endl;
            std::cout << "Dim: " << x.size() << ", Max Iter: " << opts.max_iter << std::endl;
        }

        for (int iter = 0; iter < opts.max_iter; ++iter) {
            // 1. Evaluate Function
            Eigen::VectorXd f_val = F(x);
            double error = f_val.norm();
            
            if (opts.verbose) {
                std::cout << "  Iter " << std::setw(3) << iter 
                          << " | Error: " << std::scientific << error 
                          << " | x[0]: " << (x.size() > 0 ? x[0] : 0.0) << std::endl;
            }

            if (error < opts.tol) {
                if (opts.verbose) std::cout << "[CONVERGED] Solution found." << std::endl;
                return x;
            }

            // 2. Compute Finite Difference Jacobian
            Eigen::MatrixXd J = compute_jacobian(F, x, f_val, opts.fd_eps);

            // 3. Newton Step: J * dx = -f
            // Use ColPivHouseholderQR for robustness against singular matrices
            Eigen::VectorXd dx = J.colPivHouseholderQr().solve(-f_val);

            // 4. Backtracking Line Search
            double lambda = opts.damping_factor;
            bool improved = false;
            
            for(int bt = 0; bt < opts.max_backtracking; ++bt) {
                Eigen::VectorXd x_new = x + lambda * dx;
                Eigen::VectorXd f_new = F(x_new);
                
                if (f_new.norm() < error) {
                    x = x_new;
                    improved = true;
                    break;
                }
                lambda *= 0.5;
            }
            
            if (!improved) {
                // If backtracking fails, take the step anyway or terminate?
                // Usually take small step or trust region. For now, take smallest damped step or break.
                if (opts.verbose) std::cout << "[WARN] Line search failed. Taking small step." << std::endl;
                x = x + lambda * dx;
            }
        }
        
        if (opts.verbose) std::cout << "[FAIL] Max iterations reached." << std::endl;
        return x;
    }

private:
    static Eigen::MatrixXd compute_jacobian(
        ResidualFunc F, 
        const Eigen::VectorXd& x, 
        const Eigen::VectorXd& f_base,
        double eps
    ) {
        int n = x.size();
        int m = f_base.size(); // residuals
        Eigen::MatrixXd J(m, n);
        
        Eigen::VectorXd x_pert = x;
        
        for(int j=0; j<n; ++j) {
            double orig = x_pert[j];
            double h = eps * std::max(1.0, std::abs(orig)); // Relative step size
            
            x_pert[j] = orig + h;
            Eigen::VectorXd f_plus = F(x_pert);
            
            // Central Difference? No, Forward is cheaper (N+1 evals vs 2N)
            // J.col(j) = (f_plus - f_base) / h;
            
            // Let's use Forward difference reusing f_base
            for(int i=0; i<m; ++i) {
                J(i, j) = (f_plus[i] - f_base[i]) / h;
            }
            
            x_pert[j] = orig; // Restore
        }
        
        return J;
    }
};

} // namespace Monad
