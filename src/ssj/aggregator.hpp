#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include "../UnifiedGrid.hpp"
#include "jacobian_builder.hpp" 

namespace Monad {

class JacobianAggregator {
public:
    using Vec = Eigen::VectorXd;
    using SparseMat = Eigen::SparseMatrix<double>;

    // -----------------------------------------------------------------------
    // Core Logic: Fake News Algorithm (Forward Propagation)
    // -----------------------------------------------------------------------
    // 金利 r が t=0 でショックを受けたときの、総資本 K のインパルス応答を計算する
    // Output: Vector of size T (dK_0, dK_1, ..., dK_T-1)
    static Vec build_asset_impulse_response(
        int T,
        const UnifiedGrid& grid,
        const Vec& D_ss,           // Steady State Distribution
        const SparseMat& Lambda_ss,// Steady State Transition Matrix
        const std::vector<double>& a_ss, // Steady State Asset Policy
        const Vec& da_dr           // Sensitivity of asset policy (from Partials)
    ) {
        Vec dK = Vec::Zero(T);

        // 1. Direct Effect on K at t=0?
        // K_0 is determined by D_0 (which is fixed at D_ss).
        // K_t = sum(D_t * a).
        // So shock at t=0 does NOT affect K_0.
        dK[0] = 0.0; 

        // 2. Fake News: Distribution Shift at t=1
        // dD_1 = (dLambda_r)^T * D_ss
        Vec dD = compute_distribution_shift(grid, D_ss, a_ss, da_dr);

        // 3. Propagation
        // dD_t = Lambda_ss^T * dD_{t-1}
        for (int t = 1; t < T; ++t) {
            // Calculate Aggregate Capital Change: dK_t = sum(dD_t * grid)
            double dk_val = 0.0;
            for(int i=0; i<grid.size; ++i) dk_val += dD[i] * grid.nodes[i];
            dK[t] = dk_val;

            // Propagate distribution to next period
            if (t < T - 1) {
                dD = Lambda_ss.transpose() * dD;
            }
        }

        return dK;
    }

    // 金利 r が t=0 でショックを受けたときの、総消費 C のインパルス応答
    static Vec build_consumption_impulse_response(
        int T,
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const SparseMat& Lambda_ss,
        const std::vector<double>& a_ss,
        const Vec& c_ss,           // Steady State Consumption Policy
        const Vec& da_dr,          // To compute distribution shift
        const Vec& dc_dr           // Direct sensitivity of consumption
    ) {
        Vec dC = Vec::Zero(T);

        // 1. Direct Effect at t=0
        // C_0 = sum(D_ss * c_new) => dC_0 = sum(D_ss * dc_dr)
        dC[0] = D_ss.dot(dc_dr);

        // 2. Fake News: Distribution Shift at t=1
        Vec dD = compute_distribution_shift(grid, D_ss, a_ss, da_dr);

        // 3. Propagation
        for (int t = 1; t < T; ++t) {
            // dC_t = sum(dD_t * c_ss)
            // (Note: We assume r_t is back to steady state for t>0, so we use c_ss)
            dC[t] = dD.dot(c_ss);

            if (t < T - 1) {
                dD = Lambda_ss.transpose() * dD;
            }
        }

        return dC;
    }

private:
    // Helper: Compute dD_1 = (dLambda)^T * D_ss
    // Using perturbation of weights technique
    static Vec compute_distribution_shift(
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const std::vector<double>& a_ss,
        const Vec& da
    ) {
        int n = grid.size;
        Vec dD = Vec::Zero(n);

        // For each grid point i, agents move to a_ss[i] + da[i]
        for (int i = 0; i < n; ++i) {
            double mass = D_ss[i];
            double shift = da[i]; 
            double a_dest = a_ss[i];

            if (mass < 1e-16) continue; // Skip empty bins
            if (std::abs(shift) < 1e-16) continue;

            // Find bracket [grid[j], grid[j+1]] for the ORIGINAL destination a_ss
            auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_dest);
            int j = 0;
            if (it == grid.nodes.begin()) j = 0;
            else if (it == grid.nodes.end()) j = n - 2;
            else j = (int)std::distance(grid.nodes.begin(), it) - 1;
            
            if (j < 0) j = 0;
            if (j >= n - 1) j = n - 2;

            double dx = grid.nodes[j+1] - grid.nodes[j];
            
            // Perturbation of linear interpolation weights
            // New weight on right (j+1) = Old weight + da/dx
            // So mass moving to j+1 changes by: mass * (da/dx)
            // Mass moving to j   changes by: -mass * (da/dx)
            
            double d_mass_right = mass * (shift / dx);
            
            // Apply shift to distribution density
            // Using safe indexing
            if (j >= 0 && j < n) dD[j] -= d_mass_right;
            if (j+1 >= 0 && j+1 < n) dD[j+1] += d_mass_right;
        }
        
        return dD; 
    }
};

} // namespace Monad
