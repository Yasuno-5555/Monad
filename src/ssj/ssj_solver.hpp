#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include "jacobian_builder.hpp"
#include "aggregator.hpp"

namespace Monad {

class SsjSolver {
public:
    using Mat = Eigen::MatrixXd;
    using Vec = Eigen::VectorXd;

    // Solve for the general equilibrium price path dr
    // given an exogenous shock path dZ (e.g., productivity shock)
    static Vec solve_linear_transition(
        int T,
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const Eigen::SparseMatrix<double>& Lambda_ss,
        const std::vector<double>& a_ss,
        const JacobianBuilder::PolicyPartials& partials,
        const Vec& dZ // Exogenous shock vector (e.g., shift in labor demand)
    ) {
        std::cout << "[Monad::SSJ] Building General Equilibrium Jacobian..." << std::endl;

        // 1. Build Household Jacobian Matrix (J_ha)
        // We have the impulse response to a shock at t=0: IRF_0
        // Because of time invariance, the response to a shock at t=s is just IRF_0 shifted by s.
        // J_ha is a Toeplitz-like matrix (columns are shifted IRFs).
        
        // Get the "base" impulse response (column 0)
        // Note: build_asset_impulse_response returns dK vector
        Vec irf_K = JacobianAggregator::build_asset_impulse_response(
            T, grid, D_ss, Lambda_ss, a_ss, partials.da_dr
        );

        Mat J_ha = Mat::Zero(T, T);
        for (int col = 0; col < T; ++col) {
            for (int row = col; row < T; ++row) {
                // J[row, col] is effect on t=row from shock at t=col
                // = effect on t=(row-col) from shock at t=0
                J_ha(row, col) = irf_K(row - col);
            }
        }

        // 2. Build Firm Jacobian Matrix (J_firm)
        // Simple case: K_dem = (r / alpha*A)^(1/(alpha-1))
        // Linearized: dK_dem = (dK/dr)_ss * dr
        // This is diagonal because firms are static (in this simple model).
        
        // Analytical derivative of K_dem w.r.t r (Scalar)
        double alpha = 0.33; 
        double A = 1.0; 
        double r_ss = 0.03; // Approximate, should be passed in ideally
        // Re-calibrating dK/dr slightly based on actual r_ss would be better but this is sufficient for MVP
        
        double term = r_ss / (alpha * A);
        double K_dem_ss = std::pow(term, 1.0 / (alpha - 1.0));
        
        // dK_dem/dr = 1/(alpha-1) * K_dem_ss/r_ss
        double dKdr_firm_scalar = (1.0/(alpha - 1.0)) * (K_dem_ss / r_ss);

        Mat J_firm = Mat::Identity(T, T) * dKdr_firm_scalar;

        // 3. Total Equilibrium Jacobian (H)
        // Market Clearing: K_sup(r) - K_dem(r) = 0
        // Total J = J_ha - J_firm
        // We want to solve J * dr + dZ = 0  =>  J * dr = -dZ
        Mat H = J_ha - J_firm;

        // 4. Solve Linear System
        std::cout << "[Monad::SSJ] Solving linear system (Size " << T << ")..." << std::endl;
        
        // Use PartialPivLU for general square matrix
        Vec dr = H.partialPivLu().solve(-dZ); // Note: solve for -dZ

        return dr;
    }
};

} // namespace Monad
