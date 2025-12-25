#ifndef MONAD_EGM_CORE_H
#define MONAD_EGM_CORE_H

#include <vector>
#include <functional>

// Simple typedefs for this phase
using Grid = std::vector<double>;
using Policy = std::vector<double>;

class EgmKernel {
public:
    // Solve one step backward: u'(c_t) = beta * R * E[u'(c_{t+1})]
    // Input: next_period_marginal_value (on a_grid), R (interest), beta
    // Output: current_policy (c_t on a_grid)
    // This is a simplified interface for v1.1 proof-of-concept
    static Policy step(const Grid& a_grid, const std::vector<double>& next_marg_value, double beta, double R, double gamma, bool check_monotonicity = true);

private:
    // Inverse marginal utility for CRRA: u'(c) = x => c = x^(-1/gamma)
    static double inv_u_prime(double marg_util, double gamma);
    
    // Check if the endogenous asset grid is monotonic
    static void verify_monotonicity(const std::vector<double>& endogenous_a);
    
    // Linear interpolation
    static double interpolate(const std::vector<double>& x, const std::vector<double>& y, double xi);
};

#endif // MONAD_EGM_CORE_H
