#pragma once
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>

namespace Monad {

class FiscalBlock {
public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    // Debt Dynamics:
    // dB_t = (1 + r_ss) * dB_{t-1} + B_ss * dr_t + dG_t - dT_t
    // Fiscal Rule:
    // dT_t = phi * dB_{t-1}
    
    // Updated Rule (v1.5.1):
    // dT_t = phi * dB_{t-1} + psi * B_ss * dr_t
    // psi: Pass-through of direct interest costs (0 = None, 1 = Full)
    
    static Mat build_tax_response_matrix(int T, double r_ss, double B_ss, double phi, double psi) {
        // 1. Express system for dB (Standard AR1 dynamics, assuming dT cancels interest shock if psi=1?)
        // Let's derive dB carefully with the new rule.
        // dB_t = (1+r)dB_{t-1} + B*dr - dT
        // dT = phi*dB_{t-1} + psi*B*dr
        // Substitute:
        // dB_t = (1+r - phi)dB_{t-1} + (1 - psi)B*dr
        
        double rho_B = 1.0 + r_ss - phi;
        
        // Build Lower Triangular L_inv for Debt accumulation
        Mat L_inv = Mat::Zero(T, T);
        for(int c=0; c<T; ++c) {
            for(int r=c; r<T; ++r) {
                L_inv(r, c) = std::pow(rho_B, r - c);
            }
        }
        
        // Source term for Debt is now (1 - psi) * B_ss * dr
        // M_B_r = L_inv * (1 - psi) * B_ss
        Mat M_B_r = L_inv * ((1.0 - psi) * B_ss);
        
        // 2. Compute dT
        // dT = M_T_B * dB + J_T_r_direct * dr
        
        // Lagged response M_T_B (Matrix of phi lags)
        Mat M_T_B = Mat::Zero(T, T);
        for(int t=1; t<T; ++t) {
            M_T_B(t, t-1) = phi;
        }
        
        // Direct response J_T_r_direct (Diagonal psi * B_ss)
        Mat J_T_r_direct = Mat::Identity(T, T) * (psi * B_ss);
        
        // Total M_T_r
        // dT = M_T_B * (M_B_r * dr) + J_T_r_direct * dr
        //    = (M_T_B * M_B_r + J_T_r_direct) * dr
        
        return M_T_B * M_B_r + J_T_r_direct;
    }
};

} // namespace Monad
