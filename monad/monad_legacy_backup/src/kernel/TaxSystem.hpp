#pragma once
#include <cmath>
#include <iostream>
#include <vector>

namespace Monad {

struct TaxSystem {
    double lambda = 1.0;     // Tax level (multiplicative shifter for post-tax income)
    double tau = 0.0;        // Progressivity (0 = Flat, >0 = Progressive)
    double transfer = 0.0;   // Lump sum transfer

    // HSV Model applied to Y:
    // Net Income = lambda * Y^(1-tau) + transfer
    // Tax Paid = Y - Net Income
    
    // Note: For negative or zero income, we need safety.
    // Assuming Y > 0.
    
    // Generic Net Income (for Total Y = r*a + w*z)
    template <typename T>
    T get_net_income(T gross_y) const {
        // Assume gross_y > 0 for simplicity. 
        // We might want a mechanism to handle losses (zero tax? or refund?)
        // For HANK steady state, y > 0 usually.
        // If T is Dual, we need to be careful with comparisons.
        // Assuming T supports basic ops.
        
        // Using pow(T, double) - ADL will find Monad::pow for Dual, std::pow for double if included
        using std::pow; 
        return lambda * pow(gross_y, 1.0 - tau) + transfer;
    }

    // Inverse Budget Solver:
    // Solve for 'a' such that:
    // a + Net(r*a + w*z) = Resources
    // Where Net(y) = lambda * y^(1-tau) + transfer
    // Let f(a) = a + lambda * (r*a + w*z)^(1-tau) + transfer - Resources
    // f'(a) = 1 + lambda * (1-tau) * (r*a + w*z)^(-tau) * r
    
    template <typename T>
    T solve_asset_from_budget(T resources, T r, T w, double z_val) const {
        // Newton-Raphson
        // Initial guess: assume linear tax (tau=0) or just resources?
        // If tau=0, Net = lambda*(ra+wz) + transfer.
        // a + lambda*r*a + lambda*w*z + transfer = Resources
        // a(1 + lambda*r) = Resources - lambda*w*z - transfer
        // a = (Resources - NetLabor - Transfer) / (1 + lambda*r)
        // This is a good guess.
        
        using std::pow; // Enable ADL

        T Ag = (resources - get_net_income(w*z_val)) / (1.0 + r); // Approx guess
        
        for(int k=0; k<10; ++k) { // Usually converges in 2-3 steps
            T income = r * Ag + w * z_val;
            
            // Value
            // Avoiding negative base for pow.
            // In C++, we need to ensure T behaves like a scalar or Dual.
            // If T is Dual, std::pow(Dual, double) should be supported.
            
            T net_inc = lambda * pow(income, 1.0 - tau) + transfer;
            T f_val = Ag + net_inc - resources;
            
            T marg = lambda * (1.0 - tau) * pow(income, -tau);
            T f_prime = 1.0 + marg * r;
            
            T diff = f_val / f_prime;
            Ag = Ag - diff;
            
            // Convergence check? Magnitude of diff.
            // For Dual, checking .val is tricky without traits.
            // Just fixed iterations is safe for AD.
        }
        return Ag;
    }

    // Marginal Tax Rate (on labor): T'(y)
    // Net Marginal Income: 1 - T'(y) = lambda * (1-tau) * y^(-tau)
    template <typename T>
    T get_marginal_retention_rate(T gross_labor_y) const {
        using std::pow;
        return lambda * (1.0 - tau) * pow(gross_labor_y, -tau);
    }
};

} // namespace Monad
