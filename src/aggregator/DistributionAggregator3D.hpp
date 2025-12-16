#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <numeric>
#include "../grid/MultiDimGrid.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

class DistributionAggregator3D {
    const MultiDimGrid& grid;

public:
    DistributionAggregator3D(const MultiDimGrid& g) : grid(g) {}

    // Initialize Uniform Distribution
    std::vector<double> init_uniform() {
        std::vector<double> D(grid.total_size, 1.0 / grid.total_size);
        return D;
    }

    // Forward Iterate: D_{t+1} = T(D_t) using Policy
    // Returns max absolute difference for convergence check
    double forward_iterate(const std::vector<double>& D_curr, 
                           std::vector<double>& D_next,
                           const TwoAssetPolicy& pol,
                           const IncomeProcess& income) {
        
        // Reset Next Distribution
        std::fill(D_next.begin(), D_next.end(), 0.0);

        // Loop over current state (im, ia, iz)
        for (int iz = 0; iz < grid.n_z; ++iz) {
            for (int ia = 0; ia < grid.N_a; ++ia) {
                for (int im = 0; im < grid.N_m; ++im) {
                    
                    int idx = grid.idx(im, ia, iz);
                    double mass = D_curr[idx];
                    
                    if (mass < 1e-12) continue; // Skip empty states

                    // 1. Get Policy Choices
                    double m_next_val = pol.m_pol[idx];
                    double a_next_val = pol.a_pol[idx];

                    // 2. Find Weights for Next State (Bilinear Interpolation)
                    // We need to distribute 'mass' onto the 4 neighbors in (m, a) plane
                    
                    // --- Find m indices ---
                    int im_low, im_high;
                    double wm_low, wm_high;
                    find_weights(grid.m_grid, m_next_val, im_low, im_high, wm_low, wm_high);

                    // --- Find a indices ---
                    int ia_low, ia_high;
                    double wa_low, wa_high;
                    find_weights(grid.a_grid, a_next_val, ia_low, ia_high, wa_low, wa_high);

                    // 3. Distribute Mass to Future Income States (z -> z')
                    // Mass splits into z' based on transition probabilities
                    int z_offset = iz * grid.n_z; // Row in Pi matrix
                    
                    
                    for (int iz_next = 0; iz_next < grid.n_z; ++iz_next) {
                        double prob_z = income.Pi_flat[z_offset + iz_next];
                        if (prob_z < 1e-10) continue;

                        double mass_z = mass * prob_z;

                        // Distribute to 4 corners of (m, a) grid for this z_next
                        // Corner 1: (low, low)
                        // idx = iz * stride_z + ia * stride_a + im
                        // stride_a = N_m, stride_z = N_a * N_m
                        
                        int idx_LL = grid.idx(im_low, ia_low, iz_next);
                        D_next[idx_LL] += mass_z * wm_low * wa_low;

                        // Corner 2: (high, low) - High m, Low a
                        int idx_HL = grid.idx(im_high, ia_low, iz_next);
                        D_next[idx_HL] += mass_z * wm_high * wa_low;

                        // Corner 3: (low, high) - Low m, High a
                        int idx_LH = grid.idx(im_low, ia_high, iz_next);
                        D_next[idx_LH] += mass_z * wm_low * wa_high;

                        // Corner 4: (high, high)
                        int idx_HH = grid.idx(im_high, ia_high, iz_next);
                        D_next[idx_HH] += mass_z * wm_high * wa_high;
                    }
                }
            }
        }

        // Compute Diff & Check Sum
        double max_diff = 0.0;
        double sum = 0.0;
        for(size_t i=0; i<D_curr.size(); ++i) {
            double d = std::abs(D_next[i] - D_curr[i]);
            if(d > max_diff) max_diff = d;
            sum += D_next[i];
        }
        
        return max_diff;
    }
    
    // Compute Aggregates
    void compute_aggregates(const std::vector<double>& D, double& Agg_Liquid, double& Agg_Illiquid) {
        Agg_Liquid = 0.0;
        Agg_Illiquid = 0.0;
        for(int i=0; i<grid.total_size; ++i) {
            auto vals = grid.get_values(i);
            Agg_Liquid   += D[i] * vals.first;
            Agg_Illiquid += D[i] * vals.second;
        }
    }

private:
    // Helper: Find linear interpolation weights for value 'val' on grid 'g'
    inline void find_weights(const UnifiedGrid& g, double val, 
                             int& i_low, int& i_high, 
                             double& w_low, double& w_high) const {
        
        // Clamp
        if (val <= g.nodes.front()) {
            i_low = 0; i_high = 0; w_low = 1.0; w_high = 0.0; return;
        }
        if (val >= g.nodes.back()) {
            i_low = g.size - 1; i_high = g.size - 1; w_low = 1.0; w_high = 0.0; return;
        }

        // Binary Search
        auto it = std::lower_bound(g.nodes.begin(), g.nodes.end(), val);
        i_high = std::distance(g.nodes.begin(), it);
        i_low = i_high - 1;

        double x_low = g.nodes[i_low];
        double x_high = g.nodes[i_high];
        
        // w_high * x_high + w_low * x_low = val
        // w_high + w_low = 1
        // w_high (x_high - x_low) = val - x_low
        
        w_high = (val - x_low) / (x_high - x_low);
        w_low = 1.0 - w_high;
    }
};

} // namespace Monad
