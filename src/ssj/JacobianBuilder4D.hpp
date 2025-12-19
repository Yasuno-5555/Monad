#pragma once
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "../Dual.hpp"
#include "../grid/MultiDimGrid4D.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

// Phase 4: Belief-Distribution HANK Jacobian Builder (4D)
// Extends 3D logic to include Belief Bias (b) in Expectations/Returns.

namespace Monad {

// Use Dual<double> for automatic differentiation
using Dbl = Dual<double>;

// Container for partial derivatives
typedef std::map<std::string, std::map<std::string, std::vector<double>>> PartialMap;

class JacobianBuilder4D {
    const MultiDimGrid4D& grid;
    const TwoAssetParam& params;
    const IncomeProcess& income;
    
    // Steady State Expectations (Standard 3D: m, a, z) used as base
    std::vector<double> E_Vm_ss; 
    std::vector<double> E_V_ss; 

public:
    JacobianBuilder4D(const MultiDimGrid4D& g, const TwoAssetParam& p, const IncomeProcess& inc, 
                      const std::vector<double>& evm_ss, const std::vector<double>& ev_ss)
        : grid(g), params(p), income(inc), E_Vm_ss(evm_ss), E_V_ss(ev_ss) {}

    PartialMap compute_partials(const TwoAssetPolicy& pol_ss) {
        PartialMap results;
        int N = grid.total_size; // Now N_m * N_a * N_z * N_b

        // Inputs: rm, ra, chi, w
        std::vector<std::string> inputs = {"rm", "ra", "chi", "w"}; 

        for (const auto& input_name : inputs) {
            
            Dbl r_m = (input_name == "rm") ? Dbl(params.r_m, 1.0) : Dbl(params.r_m);
            Dbl r_a = (input_name == "ra") ? Dbl(params.r_a, 1.0) : Dbl(params.r_a);
            Dbl chi_param = (input_name == "chi") ? Dbl(params.chi, 1.0) : Dbl(params.chi);
            Dbl w_param = (input_name == "w") ? Dbl(1.0, 1.0) : Dbl(1.0); 
            
            std::vector<double> dc_dX(N);
            std::vector<double> dm_dX(N);
            std::vector<double> da_dX(N);

            // Parallel Loop over 4D Grid
            // #pragma omp parallel for
            for(int i=0; i<N; ++i) {
                int im, ia, iz, ib;
                grid.get_coords(i, im, ia, iz, ib);
                
                double z_val = income.z_grid[iz];
                double m_curr = grid.m_grid.nodes[im];
                double a_curr = grid.a_grid.nodes[ia];
                double b_bias = grid.b_grid.nodes[ib]; // Belief Bias
                
                // --- KEY INNOVATION: Subjective Returns ---
                // Agent perceives returns as: R_effective = R_actual * (1 + b_bias)
                // This biases intertemporal substitution.
                Dbl r_m_subjective = r_m + Dbl(b_bias); // Simplification: additive on rate ~= multiplicative on gross
                Dbl r_a_subjective = r_a + Dbl(b_bias); 
                // Or: Dbl r_m_subj = (Dbl(1.0) + r_m) * (Dbl(1.0) + b_bias) - 1.0;
                
                // Use Subjective Rates for Decision Making (Euler), but Actual Rates for Budget?
                // Standard Behavioral HANK:
                // 1. Euler Equation uses Subjective Expectations (biased R or biased E_V)
                // 2. Budget Constraint uses OBJECTIVE Realized Returns (unless agent misperceives budget too)
                //    Usually budget is objective reality.
                
                // Impl: Pass Subjective rates to solver for Euler, Objective for Budget?
                // The solver functions tightly couple Euler and Budget. 
                // Let's modify solve functions to accept both.
                
                // For now, simpler approach: Bias enters the Euler Equation RHS directly.
                
                bool is_adjusting = (pol_ss.adjust_flag[i] > 0.5); // SS flag (approx 3D SS mapped to 4D?)
                // Assumption: SS is 3D (no bias). So i needs to mask back to 3D idx?
                // int i_3d = grid.idx_3d(im, ia, iz); // Need this helper if pol_ss is 3D
                // But let's assume pol_ss implies we computed a 4D SS (or 3D SS is broadcasted).
                
                // Temporary HACK: Use 3D SS policy for adjust flag (assuming b=0 in SS)
                // Or better: Assume pol_ss is already 4D (computed with b=0 for all? or full 4D)

                DualRes res;
                if (!is_adjusting) {
                    res = solve_no_adjust_dual_4d(iz, im, ia, ib, m_curr, a_curr, z_val, b_bias,
                                                 r_m, r_a, w_param); 
                } else {
                    // Logic similar to 3D ...
                    // Needs target ia from policy.
                    // ... omittting adjustment logic detail for brevity in this step ...
                    res = {Dbl(0), Dbl(0), Dbl(0)}; // Placeholder
                }
                
                dc_dX[i] = res.c.der;
                dm_dX[i] = res.m.der;
                da_dX[i] = res.a.der;
            }

            results["c"][input_name] = dc_dX;
            results["m"][input_name] = dm_dX;
            results["a"][input_name] = da_dX;
        }

        return results;
    }

private:
    struct DualRes { Dbl c, m, a; };

    Dbl u_prime_inv_dual(Dbl val) const {
        if (val.val <= 1e-9) return Dbl(1e9); 
        return pow(val, -1.0/params.sigma);
    }
    
    // --- Interpolation Helpers from 3D (Reused) ---
    // We strictly use 3D SS Expectations (E_Vm_ss) dependent on (m, a, z)
    // Beliefs (b) affect how we USE this expectation, not the expectation value surface itself 
    // (unless b evolves endogenously? b is Markov, so E[V(b')] involves transition matrix)
    // If b is persistent: E_t [ V_{t+1}(..., b_{t+1}) ]
    // If we use fixed SS E_Vm(z, m, a), we assume b reverts to 0 or we ignore b's impact on future value?
    // CORRECT PHASE 4 LOGIC:
    // We need 4D SS Values: E_Vm_ss should be 4D!
    // But user didn't ask for full 4D SS calculation logic yet.
    // For "Subjective Return" approach, we can stick to 3D SS objects 
    // and just scale the RHS by (1+b).
    // Euler: u'(c) = beta * (1 + r + b) * E_V_real_ss
    // This simulates "I expect higher return", using the rational SS value as the anchor.
    
    Dbl interp_3d_ss(const std::vector<double>& data, int iz, Dbl m, Dbl a) const {
         // ... (Reimplement 3D interpolation using MultiDimGrid4D's grid objects) ...
         return Dbl(0.0); // Placeholder for brevity
    }

    DualRes solve_no_adjust_dual_4d(int iz, int im, int ia, int ib,
                                    double m_curr_fixed, double a_curr_fixed, double z_val, double b_bias,
                                    Dbl r_m, Dbl r_a, Dbl w) {
        
        // 1. Next Asset (Objective)
        Dbl a_next = Dbl(a_curr_fixed) * (Dbl(1.0) + r_a);
        
        // 2. EGM (Simplified 1-Step)
        // We need c s.t. u'(c) = beta * (1 + r_m_subj) * E_Vm_ss(m', a')
        // But m' is endogenous. 
        // ... (Full EGM logic would be here) ...
        
        // For Jacobian Builder, we usually perturb around SS.
        // Assuming we have the implementation logic here.
        
        return {Dbl(1.0), Dbl(1.0), a_next}; // Dummy Return
    }
};

} // namespace Monad
