#pragma once

#include <vector>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include "UnifiedGrid.hpp"

// Placeholder for generated functions (in a real engine, this would be a template or included)
// For v1.1 skeleton, we assume basic CRRA
template<typename T>
struct EgmParams {
    T beta;
    T sigma;
    T r;
    T w;
};

namespace Monad {

class EgmKernel {
public:
    // Solves for policy functions c(a, z) and a'(a, z)
    // Hybrid Strategy: EGM first, if non-monotonic, fallback to VFI (Grid Search)
    template <typename T>
    static void solve_policy(const UnifiedGrid& grid, 
                             const EgmParams<T>& params,
                             const std::vector<T>& expected_mu, // E[u']
                             const std::vector<T>& expected_v,  // E[v] - Needed for VFI fallback
                             std::vector<T>& c_policy,
                             std::vector<T>& a_policy) {

        
        int Na = grid.size;
        std::vector<T> c_endo(Na);
        std::vector<T> a_endo(Na);
        
        // --- 1. Try EGM ---
        bool egm_success = true;
        
        for (int i = 0; i < Na; ++i) {
            T emu = expected_mu[i];
            
            // Euler: u'(c) = beta * (1+r) * E[u'(c')]
            T rhs = params.beta * (1.0 + params.r) * emu;
            
            // c = rhs^(-1/sigma)
            T c = pow(rhs, -1.0 / params.sigma);
            c_endo[i] = c;
            
            // Budget: c + a' = (1+r)a + w
            // => a = (c + a' - w)/(1+r)
            T a_prime = grid.nodes[i]; 
            a_endo[i] = (c + a_prime - params.w) / (1.0 + params.r); 
        }

        // Monotonicity Check
        for (int i = 1; i < Na; ++i) {
            if (a_endo[i] < a_endo[i-1]) {
                egm_success = false;
                break;
            }
        }

        // --- 2. Fallback to VFI if EGM failed ---
        if (!egm_success) {
            // Placeholder: Throw error until VFI generic code is fixed for Dual
             throw std::runtime_error("EGM Non-monotonicity detected. VFI Fallback disabled.");
        }

        // --- 3. EGM Interpolation (If successful) ---
        c_policy.resize(Na);
        a_policy.resize(Na); 
        
        int j = 0;
        for (int i = 0; i < Na; ++i) {
            double a_target = grid.nodes[i];
            
            while (j < Na - 1 && a_endo[j+1] < a_target) j++;
            
            if (a_target < a_endo[0]) {
                a_policy[i] = grid.nodes[0];
                c_policy[i] = (1.0 + params.r) * a_target + params.w - a_policy[i];
            } else if (a_target > a_endo[Na-1]) {
                a_policy[i] = grid.nodes[Na-1]; 
                c_policy[i] = (1.0 + params.r) * a_target + params.w - a_policy[i];
            } else {
                T m = (a_target - a_endo[j]) / (a_endo[j+1] - a_endo[j]);
                c_policy[i] = c_endo[j] + m * (c_endo[j+1] - c_endo[j]);
                T ap_j = grid.nodes[j];
                T ap_j1 = grid.nodes[j+1];
                a_policy[i] = ap_j + m * (ap_j1 - ap_j);
            }
        }
    }
};
}  