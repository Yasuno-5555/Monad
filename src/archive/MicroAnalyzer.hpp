#pragma once
#include <Eigen/Dense>
#include <vector>
#include <algorithm>
#include "grid/MultiDimGrid.hpp"
#include "kernel/TwoAssetKernel.hpp"
#include "Params.hpp"

namespace Monad {

class MicroAnalyzer {
    const MultiDimGrid& grid;
    const TwoAssetPolicy& policy;
    const TwoAssetParam& params;
    
public:
    MicroAnalyzer(const MultiDimGrid& g, const TwoAssetPolicy& p, const TwoAssetParam& par)
        : grid(g), policy(p), params(par) {}
        
    // Compute MPC for every point in the grid
    // MPC = (C(m + dm) - C(m)) / dm
    // Or analytical using Envelope Thm: MPC = dC/dm
    // In our model, we store policy c_pol. We can finite-diff or derive.
    // In continuous time / Euler eq: u'(c) = V_m.
    // MPC = dC/dm = (u'')^-1 * V_mm. Hard.
    // Simple Finite Difference along M grid is robust.
    std::vector<double> compute_mpc_grid() {
        std::vector<double> mpc(grid.total_size, 0.0);
        
        for(int iz=0; iz<grid.n_z; ++iz) {
            for(int ia=0; ia<grid.N_a; ++ia) {
                // For each strip of M
                int offset = grid.idx(0, ia, iz);
                
                for(int im=0; im<grid.N_m; ++im) {
                    int idx = offset + im;
                    double c = policy.c_pol[idx];
                    
                    // Forward difference, backward at end
                    if (im < grid.N_m - 1) {
                        int idx_next = idx + 1;
                        double dm = grid.m_grid.nodes[im+1] - grid.m_grid.nodes[im];
                        double dc = policy.c_pol[idx_next] - c;
                        mpc[idx] = dc / dm;
                    } else {
                        // Backward
                        int idx_prev = idx - 1;
                        double dm = grid.m_grid.nodes[im] - grid.m_grid.nodes[im-1];
                        double dc = c - policy.c_pol[idx_prev];
                        mpc[idx] = dc / dm;
                    }
                }
            }
        }
        return mpc;
    }
    
    // Aggregates
    struct MpcStats {
        double avg_mpc;
        double weighted_mpc; // Weighted by wealth or mass? Usually by Mass (Aggregate MPC)
        std::vector<double> mpc_by_z; // Avg MPC per income state
    };
    
    MpcStats compute_stats(const std::vector<double>& distribution) {
        std::vector<double> mpc = compute_mpc_grid();
        
        double sum_mpc = 0.0;
        double sum_weighted = 0.0;
        double total_mass = 0.0;
        
        std::vector<double> sum_z(grid.n_z, 0.0);
        std::vector<double> mass_z(grid.n_z, 0.0);
        
        for(int i=0; i<grid.total_size; ++i) {
            sum_mpc += mpc[i];
            
            double d = distribution[i];
            sum_weighted += mpc[i] * d;
            total_mass += d;
            
            int im, ia, iz;
            grid.get_coords(i, im, ia, iz);
            sum_z[iz] += mpc[i] * d;
            mass_z[iz] += d;
        }
        
        std::vector<double> by_z(grid.n_z);
        for(int z=0; z<grid.n_z; ++z) {
            by_z[z] = (mass_z[z] > 0) ? sum_z[z] / mass_z[z] : 0.0;
        }
        
        return {
            sum_mpc / grid.total_size, // Raw average
            sum_weighted / total_mass, // Economic Average MPC
            by_z
        };
    }
};

} // namespace Monad
