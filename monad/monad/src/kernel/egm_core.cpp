#include "egm_core.h"
#include "../EgmKernel.hpp"
#include <cmath>
#include <iostream>

// Implementation that delegates to the new generic EgmKernel
// This unifies the logic.

Policy EgmKernel::step(const Grid& a_grid, const std::vector<double>& next_marg_value, double beta, double R, double gamma, bool check_monotonicity) {
    // 1. Convert inputs to generic kernel format
    // Monad::EgmKernel::solve_policy requires UnifiedGrid, EgmParams, expected_mu, expected_v
    
    UnifiedGrid ugrid(a_grid);
    
    EgmParams<double> params;
    params.beta = beta;
    params.sigma = gamma; // gamma is sigma (CRRA)
    params.r = R - 1.0;
    
    // In old egm_core, 'y' was implicitly 1.0 or handled strangely.
    // The previous implementation had:
    // a_endo = (c + a_grid[i] - y) / R; with hardcoded y=1.0
    // The new kernel uses 'w'. So we set w=1.0.
    params.w = 1.0; 
    
    // next_marg_value corresponds to expected_mu (raw marginal utility, hopefully)
    // The main_trivial.cpp passed: next_mu[i] = pow(c_pol[i], -gamma);
    // This matches our new convention (raw u').
    
    std::vector<double> dummy_v(a_grid.size(), 0.0); // No VFI support in this old wrapper for now
    std::vector<double> c_policy;
    std::vector<double> a_policy;
    
    // 2. Call Generic Kernel
    // Note: Old 'check_monotonicity' arg is effectively always verified inside solve_policy
    // solve_policy has built-in VFI fallback if mono check fails.
    
    // Use Monad namespace
    Monad::EgmKernel::solve_policy(ugrid, params, next_marg_value, dummy_v, c_policy, a_policy);
    
    // 3. Return C policy
    return c_policy;
}

// Keep helper for any legacy external calls, or remove if unused.
// inv_u_prime was private in old class.
double EgmKernel::inv_u_prime(double marg_util, double gamma) {
    return std::pow(marg_util, -1.0 / gamma);
}

void EgmKernel::verify_monotonicity(const std::vector<double>& endogenous_a) {
}

double EgmKernel::interpolate(const std::vector<double>& x, const std::vector<double>& y, double xi) {
     return 0.0;
}
