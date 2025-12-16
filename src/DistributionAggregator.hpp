#pragma once
#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <type_traits>
#include "UnifiedGrid.hpp"
#include "Params.hpp"

// Forward declaration if needed, or rely on template instantiation
// We assume T is double or Dual<double>

namespace Monad {

class DistributionAggregator {
private:
    // Helper to extract double value from T
    template<typename T>
    static double val(const T& x) {
        if constexpr (std::is_arithmetic_v<T>) {
            return static_cast<double>(x);
        } else {
            return x.val;
        }
    }

public:
    // Forward iterate distribution D_t to D_{t+1} using policy a_pol
    // Uses lottery/linear interpolation logic to assign mass to grid points
    template<typename T>
    static std::vector<T> forward_iterate(const std::vector<T>& D_t, 
                                          const std::vector<T>& a_pol, 
                                          const UnifiedGrid& grid) {
        int n = grid.size;
        std::vector<T> D_next(n, T(0.0));
        
        for(int i=0; i<n; ++i) {
            T a_prime = a_pol[i];
            double a_val = val(a_prime);
            
            // 1. Find interval [x_j, x_{j+1}] containing a_prime
            // Clamp to grid boundaries
            if (a_val <= grid.nodes[0]) {
                D_next[0] = D_next[0] + D_t[i];
                continue;
            }
            if (a_val >= grid.nodes[n-1]) {
                D_next[n-1] = D_next[n-1] + D_t[i];
                continue;
            }
            
            // Binary search for index
            auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_val);
            int idx = (int)std::distance(grid.nodes.begin(), it);
            // grid[idx-1] <= a_val <= grid[idx]
            
            int j = idx - 1;
            
            double x_j = grid.nodes[j];
            double x_j1 = grid.nodes[j+1];
            double h = x_j1 - x_j;
            
            // 2. Weights
            // w is weight on j+1
            // 1-w is weight on j
            T w = (a_prime - x_j) / h;
            T one_minus_w = T(1.0) - w;
            
            // 3. Update Next Distribution
            D_next[j] = D_next[j] + D_t[i] * one_minus_w;
            D_next[j+1] = D_next[j+1] + D_t[i] * w;
        }
        
        return D_next;
    }

    // 2D Forward Iterate for Aiyagari/HANK
    // D_t: size [nz * na], flattened (row-major: z0, z1...) or vector of vectors?
    // Let's assume input is flattened std::vector<T> of size Nz*Na
    // a_pol: size [nz * na], flattened
    template<typename T>
    static std::vector<T> forward_iterate_2d(const std::vector<T>& D_t, 
                                             const std::vector<T>& a_pol, 
                                             const UnifiedGrid& grid,
                                             const IncomeProcess& income) {
        int na = grid.size;
        int nz = income.n_z;
        int size = na * nz;
        
        std::vector<T> D_next(size, T(0.0));
        
        // Loop over current state (z_j, a_i)
        for(int j=0; j<nz; ++j) {
            for(int i=0; i<na; ++i) {
                int curr_idx = j * na + i;
                T mass = D_t[curr_idx];
                
                // If mass is negligible, skip (optimization)
                if (val(mass) < 1e-16) continue;

                T a_prime = a_pol[curr_idx];
                double a_val = val(a_prime);
                
                // Asset Interpolation Weights
                int k = 0;
                T w = 0.0;
                
                if (a_val <= grid.nodes[0]) {
                    k = 0; w = 0.0;
                } else if (a_val >= grid.nodes[na - 1]) {
                    k = na - 2; w = 1.0;
                } else {
                    auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_val);
                    int idx = (int)std::distance(grid.nodes.begin(), it);
                    k = idx - 1;
                    double denom = grid.nodes[k+1] - grid.nodes[k];
                    w = (a_prime - grid.nodes[k]) / denom;
                }
                
                T weight_low = T(1.0) - w;
                T weight_high = w;
                
                // Distribute to next period z_next
                for(int next_j=0; next_j < nz; ++next_j) {
                    double prob = income.prob(j, next_j);
                    if (prob < 1e-16) continue;
                    
                    int next_base = next_j * na;
                    
                    // Add mass to (z_next, a_k) and (z_next, a_{k+1})
                    T mass_flow = mass * prob;
                    
                    D_next[next_base + k] = D_next[next_base + k] + mass_flow * weight_low;
                    D_next[next_base + k + 1] = D_next[next_base + k + 1] + mass_flow * weight_high;
                }
            }
        }
        return D_next;
    }
};
}