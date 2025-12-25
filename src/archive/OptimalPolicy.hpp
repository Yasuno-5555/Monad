#pragma once
#include <Eigen/Dense>
#include <iostream>
#include "ssj/SsjSolver3D.hpp"
#include "blocks/SimpleBlocks.hpp"

namespace Monad {

class OptimalPolicy {
    SsjSolver3D& solver;
    int T;
    
public:
    OptimalPolicy(SsjSolver3D& s, int horizon) : solver(s), T(horizon) {}
    
    // Optimal Linear-Quadratic Regulator
    // Minimize Loss = Sum beta^t * (pi_t^2 + lambda_y * y_t^2)
    // Subject to linearized economy:
    //   Ay * y = Ar * r + A_shock * shock
    //   pi = kappa * y + beta * pi(+1)
    
    // Actually, we can express the whole economy as a linear system H * [Y, pi, r]' = Shock
    // And optimize over r.
    
    // Target: Find path of r that minimizes loss.
    // Inputs:
    //   lambda_y: weight on output gap stabilization
    //   shock_path: underlying shock (e.g. natural rate shock r*)
    
    struct LQRResult {
        Eigen::VectorXd r_opt;
        Eigen::VectorXd y_opt;
        Eigen::VectorXd pi_opt;
        double loss;
    };
    
    LQRResult solve_optimal_policy(double lambda_y, const Eigen::VectorXd& dr_star) {
        // 1. Get Jacobians
        auto J = solver.compute_block_jacobians(T);
        Eigen::MatrixXd J_C_r = J["C"]["rm"];
        Eigen::MatrixXd J_C_y = J["C"]["w"];
        
        // 2. Construct IS Curve constraint
        // Y = C = J_C_y * Y + J_C_r * (r - r*)  (Assuming shock is r*)
        // (I - J_C_y) * Y - J_C_r * r = -J_C_r * r*
        // Let A_is = (I - J_C_y), B_is = -J_C_r
        // A_is * Y + B_is * r = B_is * r*
        
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(T, T);
        Eigen::MatrixXd A_is = I - J_C_y;
        Eigen::MatrixXd B_is = J_C_r; // Wait, eq is: A_is * Y = J_C_r * (r - r*) -> A_is * Y - J_C_r * r = - J_C_r * r*
        // Let's stick to standard form: Y = M_Yr * r + M_Ye * shock
        // Y = (I - J_C_y)^-1 * J_C_r * (r - r*)
        
        Eigen::MatrixXd M_Yr = A_is.colPivHouseholderQr().solve(J_C_r);
        Eigen::VectorXd Y_shock = M_Yr * (-dr_star); // Effect of r* shock if r=0
        
        // 3. Construct Phillips Curve constraint
        // pi = kappa * Y + beta * pi(+1)
        // Vector form: pi = M_pi_Y * Y
        double beta_val = 0.99; // Should come from params, hardcoded for now
        double kappa_val = 0.1;
        Eigen::MatrixXd M_pi_Y = SimpleBlocks::build_nkpc_jacobian(T, beta_val, kappa_val);
        
        // 4. Optimization Problem
        // Min r  TotalLoss
        // Loss = pi' * pi + lambda_y * Y' * Y
        // Substituting relations:
        // Y  = M_Yr * r + Y_shock
        // pi = M_pi_Y * Y = M_pi_Y * (M_Yr * r + Y_shock)
        
        // Let K = M_pi_Y * M_Yr (Effect of r on pi)
        // Let C = M_pi_Y * Y_shock (Effect of shock on pi)
        // pi = K * r + C
        // Y  = M_Yr * r + Y_shock
        
        // Loss = (Kr + C)'(Kr + C) + lambda_y * (M_Yr r + Y_shock)'(M_Yr r + Y_shock)
        //      = r'K'Kr + 2C'Kr + C'C + lambda_y(r'M'Mr + 2Y_shock'Mr + Y_shock'Y_shock)
        
        // dLoss/dr = 2K'K r + 2K'C + 2*lambda_y*M'M r + 2*lambda_y*M'Y_shock = 0
        // (K'K + lambda_y M'M) r = - (K'C + lambda_y M'Y_shock)
        
        Eigen::MatrixXd K = M_pi_Y * M_Yr;
        Eigen::VectorXd C = M_pi_Y * Y_shock;
        
        Eigen::MatrixXd Hessian = K.transpose() * K + lambda_y * M_Yr.transpose() * M_Yr;
        Eigen::VectorXd GradientBase = K.transpose() * C + lambda_y * M_Yr.transpose() * Y_shock;
        
        // Solve for optimal r
        // Add minimal regularization to Hessian to ensure invertibility
        Hessian += 1e-6 * Eigen::MatrixXd::Identity(T, T);
        
        Eigen::VectorXd r_opt = Hessian.colPivHouseholderQr().solve(-GradientBase);
        
        // Recover Y and pi
        Eigen::VectorXd Y_opt = M_Yr * r_opt + Y_shock;
        Eigen::VectorXd pi_opt = K * r_opt + C;
        
        double loss = pi_opt.dot(pi_opt) + lambda_y * Y_opt.dot(Y_opt);
        
        return {r_opt, Y_opt, pi_opt, loss};
    }
};

} // namespace Monad
