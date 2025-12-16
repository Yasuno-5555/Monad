#pragma once
#include <Eigen/Dense>
#include <vector>

namespace Monad {

class SimpleBlocks {
public:
    using Mat = Eigen::MatrixXd;

    // 1. New Keynesian Phillips Curve
    // pi_t = beta * pi_{t+1} + kappa * mc_t
    // Form: A * pi = kappa * mc  => pi = A^{-1} * kappa * mc
    // Where A = I - beta * L^{-1} (Backward shift? No, Forward shift)
    // L^{-1} x_t = x_{t+1}
    static Mat build_nkpc_jacobian(int T, double beta, double kappa) {
        // Construct matrix M such that pi = M * mc
        // M = (I - beta * L^{-1})^{-1} * kappa
        
        Mat A = Mat::Identity(T, T);
        
        // Add beta * L^{-1} (Super-diagonal)
        // pi_t - beta * pi_{t+1} = ...
        // Row t has 1 at t, -beta at t+1
        for(int t=0; t<T-1; ++t) {
            A(t, t+1) = -beta;
        }
        
        // Invert A to get influence of mc on pi
        Mat invA = A.inverse(); 
        
        return invA * kappa; // Jacobian J_pi_mc
    }

    // 2. Taylor Rule
    // i_t = phi_pi * pi_t
    static Mat build_taylor_jacobian(int T, double phi_pi) {
        return Mat::Identity(T, T) * phi_pi;
    }

    // 3. Fisher Equation
    // r_t = i_t - pi_{t+1}
    // Need J_r_i (Identity) and J_r_pi (Negative Shift)
    struct FisherJacobians {
        Mat J_r_i;
        Mat J_r_pi;
    };

    static FisherJacobians build_fisher_jacobians(int T) {
        FisherJacobians out;
        out.J_r_i = Mat::Identity(T, T);
        
        // J_r_pi: r_t depends on -pi_{t+1}
        // Row t, Col t+1 is -1
        out.J_r_pi = Mat::Zero(T, T);
        for(int t=0; t<T-1; ++t) {
            out.J_r_pi(t, t+1) = -1.0;
        }
        
        return out;
    }
};

} // namespace Monad
