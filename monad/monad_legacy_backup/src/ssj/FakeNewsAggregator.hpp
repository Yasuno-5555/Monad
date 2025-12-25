#pragma once
#include <Eigen/Sparse>
#include <vector>
#include <map>
#include <iostream>
#include "../grid/MultiDimGrid.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

typedef Eigen::SparseMatrix<double> SparseMat;
typedef Eigen::Triplet<double> Triplet;

// Fake News Aggregator for 3D State Space
// Calculates the distribution impulse response dD_t to shocks
class FakeNewsAggregator {
    const MultiDimGrid& grid;
    const IncomeProcess& income;
    const SparseMat Lambda_ss; // Steady State Transition

public:
    FakeNewsAggregator(const MultiDimGrid& g, const IncomeProcess& inc, const SparseMat& lambda)
        : grid(g), income(inc), Lambda_ss(lambda) {}

    // Compute dD / dX via Fake News Algorithm
    // dD_t = Lambda_ss * dD_{t-1} + dLambda_{t-1} * D_{ss}
    // dLambda_{t-1} comes from shock at t-1 affecting policies
    // "Fake News" matrix F_s,j = curlyJ_{s, j} (Auclert 2021)
    
    // We compute the "Fake News Matrix" F for a single input variable (e.g. r_m)
    // Input 'partials' is vector of ∂m'/∂r, ∂a'/∂r at each state i
    std::vector<double> compute_fake_news_vector(const std::vector<double>& dm_dr,
                                                 const std::vector<double>& da_dr,
                                                 const TwoAssetPolicy& pol_ss,
                                                 const std::vector<double>& D_ss) {
        
        // F = dLambda * D_ss
        // dLambda is the perturbation of transition matrix
        // Instead of building dLambda (sparse), we directly compute dLambda * D_ss
        // dLambda_{ji} is change in prob(i->j)
        // (dLambda * D)_j = sum_i dLambda_{ji} * D_i
        
        // Logic:
        // For each state i (FROM), we have mass D_ss[i].
        // This mass is distributed to neighbors based on weights w_L(m'), w_H(m'), ...
        // Change in policy dm' changes weights dw.
        // d(Weight * D) = dWeight * D
        
        std::vector<double> F(grid.total_size, 0.0);

        for (int iz = 0; iz < grid.n_z; ++iz) {
            for (int ia = 0; ia < grid.N_a; ++ia) {
                for (int im = 0; im < grid.N_m; ++im) {
                    
                    int idx_from = grid.idx(im, ia, iz);
                    double mass = D_ss[idx_from];
                    if (mass < 1e-12) continue;

                    // 1. Get SS Policy
                    double m_next = pol_ss.m_pol[idx_from];
                    double a_next = pol_ss.a_pol[idx_from];

                    // 2. Get Derivatives
                    double dmp = dm_dr[idx_from]; // ∂m'/∂r
                    double dap = da_dr[idx_from]; // ∂a'/∂r

                    // 3. Compute Weight Sensitivities
                    // w_high_m = (m' - m_low) / (m_high - m_low)
                    // dw_high_m = dm' / Delta_m
                    // dw_low_m = -dw_high_m
                    
                    int im_low, im_high;
                    double wm_low, wm_high, dwm_low, dwm_high;
                    calc_weight_partials(grid.m_grid, m_next, dmp, im_low, im_high, wm_low, wm_high, dwm_low, dwm_high);

                    int ia_low, ia_high;
                    double wa_low, wa_high, dwa_low, dwa_high;
                    calc_weight_partials(grid.a_grid, a_next, dap, ia_low, ia_high, wa_low, wa_high, dwa_low, dwa_high);

                    // 4. Distribute Mass Change
                    // Total weight W = wm * wa
                    // dW = dwm * wa + wm * dwa
                    
                    int z_offset = iz * grid.n_z;
                    for (int iz_next = 0; iz_next < grid.n_z; ++iz_next) {
                        double prob_z = income.Pi_flat[z_offset + iz_next];
                        if (prob_z < 1e-10) continue;
                        
                        double m_z = mass * prob_z;

                        // Target Indices
                        int idx_LL = grid.idx(im_low, ia_low, iz_next);
                        int idx_HL = grid.idx(im_high, ia_low, iz_next);
                        int idx_LH = grid.idx(im_low, ia_high, iz_next);
                        int idx_HH = grid.idx(im_high, ia_high, iz_next);
                        
                        // Weights Combinations
                        // LL: wm_low * wa_low
                        // dLL = dwm_low * wa_low + wm_low * dwa_low
                        
                        double dW_LL = dwm_low * wa_low  + wm_low * dwa_low;
                        double dW_HL = dwm_high * wa_low + wm_high * dwa_low;
                        double dW_LH = dwm_low * wa_high + wm_low * dwa_high;
                        double dW_HH = dwm_high * wa_high+ wm_high * dwa_high;

                        F[idx_LL] += m_z * dW_LL;
                        F[idx_HL] += m_z * dW_HL;
                        F[idx_LH] += m_z * dW_LH;
                        F[idx_HH] += m_z * dW_HH;
                    }
                }
            }
        }
        return F;
    }

private:
    void calc_weight_partials(const UnifiedGrid& g, double val, double dval, 
                              int& i_low, int& i_high, 
                              double& w_low, double& w_high,
                              double& dw_low, double& dw_high) const {
        // Find indices
        if (val <= g.nodes.front()) {
            i_low = 0; i_high = 0; w_low = 1.0; w_high = 0.0; dw_low = 0.0; dw_high = 0.0; return;
        }
        if (val >= g.nodes.back()) {
            i_low = g.size - 1; i_high = g.size - 1; w_low = 1.0; w_high = 0.0; dw_low = 0.0; dw_high = 0.0; return;
        }

        auto it = std::lower_bound(g.nodes.begin(), g.nodes.end(), val);
        i_high = std::distance(g.nodes.begin(), it);
        i_low = i_high - 1;
        
        double x_low = g.nodes[i_low];
        double x_high = g.nodes[i_high];
        double h = x_high - x_low;

        w_high = (val - x_low) / h;
        w_low = 1.0 - w_high;
        
        // Derivatives
        // w_high = val/h - x_low/h
        // dw_high = dval / h
        dw_high = dval / h;
        dw_low = -dw_high;
    }
};

} // namespace Monad
