#pragma once
#include <map>
#include <string>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include "../Dual.hpp"
#include "../grid/MultiDimGrid.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

// Use Dual<double> for automatic differentiation
using Dbl = Dual<double>;

// Container for partial derivatives: Variable -> Input -> Vector
typedef std::map<std::string, std::map<std::string, std::vector<double>>> PartialMap;

class JacobianBuilder3D {
    const MultiDimGrid& grid;
    const TwoAssetParam& params;
    const IncomeProcess& income;
    
    // Steady State Expectations (needed for Euler)
    std::vector<double> E_Vm_ss; 
    std::vector<double> E_V_ss; // Not strictly needed for derivatives of c, m' if regime fixed?
                                // Actually yes, if we needed to check regime change, but we don't.

public:
    JacobianBuilder3D(const MultiDimGrid& g, const TwoAssetParam& p, const IncomeProcess& inc, 
                      const std::vector<double>& evm_ss, const std::vector<double>& ev_ss)
        : grid(g), params(p), income(inc), E_Vm_ss(evm_ss), E_V_ss(ev_ss) {}

    PartialMap compute_partials(const TwoAssetPolicy& pol_ss) {
        PartialMap results;
        int N = grid.total_size;

        // Inputs to perturb
        // r_m: Liquid Rate
        // r_a: Illiquid Rate
        // chi: Adjustment Cost Scale (for completeness)
        std::vector<std::string> inputs = {"rm", "ra", "chi"}; 

        for (const auto& input_name : inputs) {
            
            // 1. Prepare Dual Parameters
            Dbl r_m = (input_name == "rm") ? Dbl(params.r_m, 1.0) : Dbl(params.r_m);
            Dbl r_a = (input_name == "ra") ? Dbl(params.r_a, 1.0) : Dbl(params.r_a);
            Dbl chi_param = (input_name == "chi") ? Dbl(params.chi, 1.0) : Dbl(params.chi);
            
            // 2. Output Vectors
            std::vector<double> dc_dX(N);
            std::vector<double> dm_dX(N);
            std::vector<double> da_dX(N);

            // 3. One Step Backward Pass (Dual Mode)
            // Parallelization possible here
            // #pragma omp parallel for
            for(int i=0; i<N; ++i) {
                int im, ia, iz;
                grid.get_coords(i, im, ia, iz);
                double z_val = income.z_grid[iz];
                double m_curr = grid.m_grid.nodes[im];
                double a_curr = grid.a_grid.nodes[ia];
                
                // Determine Regime from Steady State
                // adjust_flag > 0.5 means Adjusted
                bool is_adjusting = (pol_ss.adjust_flag[i] > 0.5);

                DualRes res;
                if (!is_adjusting) {
                    res = solve_no_adjust_dual(iz, im, ia, m_curr, a_curr, z_val, r_m, r_a);
                } else {
                    // For Adjusting: We must target the SAME a' as SS.
                    // The SS policy 'a_pol[i]' tells us the target a'.
                    // We find index of 'a_pol[i]' in a_grid.
                    double a_target_ss = pol_ss.a_pol[i];
                    // Find closest index
                    auto it = std::lower_bound(grid.a_grid.nodes.begin(), grid.a_grid.nodes.end(), a_target_ss);
                    int ia_target = std::distance(grid.a_grid.nodes.begin(), it);
                    if(ia_target >= grid.a_grid.size) ia_target = grid.a_grid.size - 1;
                    // Could check distance to ensure exact match, but valid enough.
                    
                    res = solve_adjust_dual(iz, im, ia, ia_target, m_curr, a_curr, z_val, r_m, r_a, chi_param);
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

    // --- Dual Util Functions ---
    Dbl u_prime_inv_dual(Dbl val) const {
        // u'(c) = c^-sigma = val
        // c = val^(-1/sigma)
        if (val.val <= 1e-9) return Dbl(1e9); 
        return pow(val, -1.0/params.sigma);
    }

    Dbl adj_cost_dual(Dbl d, Dbl a_curr, Dbl chi) const {
        if (std::abs(d.val) < 1e-9) return Dbl(0.0);
        return chi * d * d;
    }

    // --- Interpolation Helpers (Dual Coords, Double Data) ---
    Dbl interp_1d_dual(const std::vector<double>& x, const std::vector<double>& y, Dbl xi) const {
        if (xi.val <= x.front()) return Dbl(y.front());
        if (xi.val >= x.back()) return Dbl(y.back());
        
        auto it = std::lower_bound(x.begin(), x.end(), xi.val);
        int idx = std::distance(x.begin(), it);
        
        // t = (xi - x0) / (x1 - x0)
        Dbl t = (xi - x[idx-1]) / (x[idx] - x[idx-1]);
        return Dbl(y[idx-1]) + t * (y[idx] - y[idx-1]);
    }
    
    // For EGM re-interpolation: Endogenous grid is Dual, Data (c_endo) is Dual
    Dbl interp_1d_dual_endo(const std::vector<Dbl>& x, const std::vector<Dbl>& y, double xi) const {
        // x is Dual (endogenous m), xi is fixed Double (current m)
        // We find bracket based on values
        // Note: x must be sorted in value. EGM usually preserves monotonicity.
        
        if (xi <= x.front().val) return y.front(); // Extrapolate? Or Clamp? Clamp safe.
        if (xi >= x.back().val) return y.back();
        
        // Custom lower_bound for vector<Dbl>
        auto it = std::lower_bound(x.begin(), x.end(), xi, [](const Dbl& a, double b){
            return a.val < b;
        });
        int idx = std::distance(x.begin(), it);
        
        Dbl x0 = x[idx-1];
        Dbl x1 = x[idx];
        Dbl y0 = y[idx-1];
        Dbl y1 = y[idx];
        
        Dbl t = (xi - x0) / (x1 - x0);
        return y0 + t * (y1 - y0);
    }

    // Interpolate Steady State E_Vm (Double) using Dual Coordinates
    Dbl interpolate_2d_m_a_dual(const std::vector<double>& data, int iz, Dbl m, Dbl a) const {
        const auto& ag = grid.a_grid.nodes;
        
        // Map 'a' (Dual) to a-grid indices
        // Since 'a' moves continuously with r_a (recapitalization), derivatives flow through here!
        if (a.val <= ag.front()) return interp_1d_slice_m_dual(data, iz, 0, m);
        if (a.val >= ag.back()) return interp_1d_slice_m_dual(data, iz, grid.N_a-1, m);

        auto it = std::lower_bound(ag.begin(), ag.end(), a.val);
        int ia = std::distance(ag.begin(), it);

        Dbl t_a = (a - ag[ia-1]) / (ag[ia] - ag[ia-1]);

        Dbl v1 = interp_1d_slice_m_dual(data, iz, ia-1, m);
        Dbl v2 = interp_1d_slice_m_dual(data, iz, ia, m);
        return v1 + t_a * (v2 - v1);
    }

    Dbl interp_1d_slice_m_dual(const std::vector<double>& data, int iz, int ia, Dbl m) const {
        int offset = grid.idx(0, ia, iz);
        const auto& mg = grid.m_grid.nodes;
        
        // m is Dual
        if (m.val <= mg.front()) return Dbl(data[offset]);
        if (m.val >= mg.back()) return Dbl(data[offset + grid.N_m - 1]);
        
        auto it = std::lower_bound(mg.begin(), mg.end(), m.val);
        int im = std::distance(mg.begin(), it);
        
        Dbl t_m = (m - mg[im-1]) / (mg[im] - mg[im-1]);
        
        Dbl y1 = data[offset + im-1];
        Dbl y2 = data[offset + im];
        return y1 + t_m * (y2 - y1);
    }

    // --- Solvers ---

    DualRes solve_no_adjust_dual(int iz, int im, int ia, double m_curr_fixed, double a_curr_fixed, 
                                 double z_val, Dbl r_m, Dbl r_a) {
        
        // 1. Next Scale
        Dbl a_next = Dbl(a_curr_fixed) * (Dbl(1.0) + r_a);
        
        // 2. EGM for Liquid Asset
        int Nm = grid.N_m;
        std::vector<Dbl> c_endo(Nm);
        std::vector<Dbl> m_endo(Nm);

        for(int im_next=0; im_next < Nm; ++im_next) {
            double m_prime_grid = grid.m_grid.nodes[im_next];
            
            // Expected Marginal Value at (m', a') using SS E_Vm
            // Vm' is double, but coordinates are Dual (a_next depends on r_a)
            Dbl emv = interpolate_2d_m_a_dual(E_Vm_ss, iz, Dbl(m_prime_grid), a_next);
            
            // FOC: u'(c) = beta * (1+r_m) * E_Wm
            Dbl rhs = Dbl(params.beta) * emv;
            Dbl c = u_prime_inv_dual(rhs); // Euler (Note: (1+r_m) factor inside emv? No, usually outside?)
            // Check TwoAssetSolver: `double evm += prob * u_prime * (1.0 + params.r_m);`
            // Wait, TwoAssetSolver stores E[u'(c') * (1+r)] as E_Vm_next?
            // Let's check compute_expectations in Solver.
            // Solver: `evm += prob * u_prime * (1.0 + params.r_m);`
            // So E_Vm_ss ALREADY includes (1+r_m_ss).
            // BUT! The r_m inside the expectation is the FUTURE rate r_m'.
            // Here we are perturbing CURRENT rate r_m?
            // "Partials" in SSJ usually mean shock to {r_t, r_{t+1}, ...}.
            // Backward step at t connects t and t+1.
            // The Euler Eq at t is: u'(c_t) = beta * E_t [ (1 + r_{m, t+1}) u'(c_{t+1}) ]
            // The term E_Vm_ss = E [ (1+r_{ss}) u'(c_{ss}) ].
            // If we perturb r_m at time t, does it affect this expectation?
            // No, r_{m, t+1} is future.
            
            // HOWEVER: Budget Constraint at t: c_t + m_{t+1} = (1 + r_{m, t}) m_t + w z_t
            // The parameter 'r_m' passed here is usually r_{m,t}.
            // So it affects the BUDGET, not the EULER (unless r_{m,t} enters Euler?). NO.
            // So 'rhs' depends only on a_next (via r_a_t affecting a_{t+1}).
            // Wait, a_{t+1} = a_t(1 + r_{a,t}). So r_a enters here.
            
            // So:
            // c calculated from Euler is not affected by r_{m,t} directly?
            // Yes, u'(c) = beta * E_Vm_next.
            // E_Vm_next depends on a_next = a(1+r_a). So c depends on r_a.
            
            // 3. Finding Endogenous m_curr
            // c + m' = (1+r_m)m + z  => m = (c + m' - z) / (1+r_m)
            Dbl num = c + m_prime_grid - z_val;
            Dbl m_curr_endo = num / (Dbl(1.0) + r_m);
            
            c_endo[im_next] = c;
            m_endo[im_next] = m_curr_endo;
        }

        // 3. Interpolate back to fixed grid m_curr_fixed
        // Using Dual interpolation on endogenous grid
        Dbl c_pol = interp_1d_dual_endo(m_endo, c_endo, m_curr_fixed);
        Dbl m_pol = interp_1d_dual_endo(m_endo, std::vector<Dbl>(grid.m_grid.nodes.begin(), grid.m_grid.nodes.end()), m_curr_fixed);
        // Wait, converting grid nodes to Dbl vector for helper
        
        // Actually, m_pol IS m_prime.
        // We have relation: m_endo -> m_prime_grid
        // We want at m_curr_fixed -> m_prime
        std::vector<Dbl> m_prime_vec(Nm);
        for(int k=0; k<Nm; ++k) m_prime_vec[k] = Dbl(grid.m_grid.nodes[k]);
        
        Dbl m_prime_final_dual = interp_1d_dual_endo(m_endo, m_prime_vec, m_curr_fixed);

        // Budget Identity Check? computed c matches interpolated c? Yes.
        
        return {c_pol, m_prime_final_dual, a_next};
    }

    DualRes solve_adjust_dual(int iz, int im, int ia, int ia_target, 
                              double m_curr_fixed, double a_curr_fixed, double z_val, 
                              Dbl r_m, Dbl r_a, Dbl chi) {
        
        // 1. Target Illiquid Asset (Fixed Regime: Target SS choice)
        Dbl a_next = Dbl(grid.a_grid.nodes[ia_target]); // Fixed target grid point?
        // Actually, in SSJ "Fixed Regime" for adjusters often means "Fixed Target Selection".
        // The target is a grid point (discrete choice). So a_next is constant double.
        // Wait, does r_a affect this?
        // If a' is chosen from grid, a' is physically fixed.
        // The cost depends on a_curr(1+r_a).
        
        Dbl d = a_next - Dbl(a_curr_fixed) * (Dbl(1.0) + r_a);
        Dbl cost = adj_cost_dual(d, Dbl(a_curr_fixed), chi);
        Dbl total_outflow = d + cost;

        // 2. EGM for Liquid Asset (Conditional on a_next)
        // Similar to No-Adjust, but Euler depends on Fixed a_next
        int Nm = grid.N_m;
        std::vector<Dbl> c_endo(Nm);
        std::vector<Dbl> m_endo(Nm);

        for(int im_next=0; im_next < Nm; ++im_next) {
            double m_prime_grid = grid.m_grid.nodes[im_next];
            
            // Expected Marginal Value at (m', a')
            // a' is fixed grid point (a_next). m' is grid point.
            // So emv is CONSTANT with respect to r_m, r_a! 
            // Because arguments to interp are fixed doubles.
            // WARNING: Does E_Vm_ss depend on anything? No, it's fixed SS.
            
            // So c is CONSTANT?
            // u'(c) = beta * E_Vm(m', a').
            // Yes, for Adjusters, the chosen a' is fixed, so c(m') is fixed (from Euler).
            // BUT m_curr depends on budget!
            
            double emv = interpolate_2d_m_a_dual(E_Vm_ss, iz, Dbl(m_prime_grid), a_next).val; 
            // Note: a_next is Dbl but has 0 derivative? Yes, if grid point.
            // Actually, if a_next is fixed grid point, its derivative is 0.
            
            Dbl rhs = Dbl(params.beta) * emv; // Constant
            Dbl c = u_prime_inv_dual(rhs);    // Constant
            
            // Budget: c + m' = (1+r_m)m + z - total_outflow
            // m = (c + m' - z + total_outflow) / (1+r_m)
            Dbl num = c + m_prime_grid - z_val + total_outflow;
            Dbl m_curr_endo = num / (Dbl(1.0) + r_m);
            
            c_endo[im_next] = c;
            m_endo[im_next] = m_curr_endo;
        }

        // 3. Interpolate
        std::vector<Dbl> m_prime_vec(Nm);
        for(int k=0; k<Nm; ++k) m_prime_vec[k] = Dbl(grid.m_grid.nodes[k]);

        Dbl c_pol = interp_1d_dual_endo(m_endo, c_endo, m_curr_fixed);
        Dbl m_pol = interp_1d_dual_endo(m_endo, m_prime_vec, m_curr_fixed);
        
        return {c_pol, m_pol, a_next};
    }
};

} // namespace Monad
