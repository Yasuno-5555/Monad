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
        const Vec& da_dr,          // Sensitivity of asset policy (from Partials)
        const IncomeProcess& income // For Z transitions
    ) {
        Vec dK = Vec::Zero(T);

        // 1. Direct Effect on K at t=0 derived from D_0=D_ss
        dK[0] = 0.0; 

        // 2. Fake News: Distribution Shift at t=1
        Vec dD = compute_distribution_shift(grid, D_ss, a_ss, da_dr, income);

        // 3. Propagation
        int max_size = grid.size * income.n_z;
        
        for (int t = 1; t < T; ++t) {
            // Calculate Aggregate Capital Change: dK_t = sum(dD_t * grid)
            double dk_val = 0.0;
            // Iterate all states (ia, iz)
            for(int i=0; i<max_size; ++i) {
                int ia_idx = i % grid.size;
                dk_val += dD[i] * grid.nodes[ia_idx];
            }
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
        const Vec& dc_dr,          // Direct sensitivity of consumption
        const IncomeProcess& income
    ) {
        Vec dC = Vec::Zero(T);

        // 1. Direct Effect at t=0
        // C_0 = sum(D_ss * c_new) => dC_0 = sum(D_ss * dc_dr)
        dC[0] = D_ss.dot(dc_dr);

        // 2. Fake News: Distribution Shift at t=1
        Vec dD = compute_distribution_shift(grid, D_ss, a_ss, da_dr, income);

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

    // Helper: Compute dD_1 = (dLambda)^T * D_ss
    // Using perturbation of weights technique (2D)
    static Vec compute_distribution_shift(
        const UnifiedGrid& grid,
        const Vec& D_ss,
        const std::vector<double>& a_ss,
        const Vec& da,
        const IncomeProcess& income // Needed for Z transition
    ) {
        int na = grid.size;
        int nz = income.n_z;
        int size = na * nz; 
        Vec dD = Vec::Zero(size);

        // For each grid point i (flattened), agents move to a_ss[i] + da[i]
        // i = iz * Na + ia
        for (int i = 0; i < size; ++i) {
            double mass = D_ss[i];
            double shift = da[i]; 
            double a_dest = a_ss[i];

            if (mass < 1e-16) continue; // Skip empty bins
            if (std::abs(shift) < 1e-16) continue;

            // Find bracket [grid[j], grid[j+1]] for the ORIGINAL destination a_ss
            auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_dest);
            int p = 0;
            if (it == grid.nodes.begin()) p = 0;
            else if (it == grid.nodes.end()) p = na - 2;
            else p = (int)std::distance(grid.nodes.begin(), it) - 1;
            
            if (p < 0) p = 0;
            if (p >= na - 1) p = na - 2;

            double dx = grid.nodes[p+1] - grid.nodes[p];
            
            // Perturbation of linear interpolation weights: da/dx
            double d_weight_right = shift / dx;
            
            // This shift in asset weight applies to ALL future income branches
            int iz_current = i / na;
            
            for(int iz_next = 0; iz_next < nz; ++iz_next) {
                double prob = income.prob(iz_current, iz_next);
                if (prob < 1e-12) continue;
                
                double d_mass_flow = mass * prob * d_weight_right;
                
                // Destination indices
                int col_left = iz_next * na + p;
                int col_right = iz_next * na + (p+1);
                
                // Mass moves from Left node to Right node?
                // Weight_right increases by d_weight_right
                // Mass(Right) += Mass * Prob * d_weight
                // Mass(Left)  -= Mass * Prob * d_weight
                
                if (col_left >= 0 && col_left < size) dD[col_left] -= d_mass_flow;
                if (col_right >= 0 && col_right < size) dD[col_right] += d_mass_flow;
            }
        }
        
        return dD; 
    }
};

} // namespace Monad
