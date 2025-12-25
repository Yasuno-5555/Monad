#pragma once
#include <algorithm>
#include <iostream>
#include <cmath>
#include <vector>
#include "../grid/MultiDimGrid.hpp"
#include "../kernel/TwoAssetKernel.hpp"
#include "../Params.hpp"

namespace Monad {

class OneAssetSolver {
    const MultiDimGrid& grid;
    const TwoAssetParam& params;
    
public:
    // Internal buffer for expected values E[V(m', z')] and E[Vm]
    std::vector<double> E_V_next; 
    std::vector<double> E_Vm_next; // Marginal Value E[dV/dm]

    OneAssetSolver(const MultiDimGrid& g, const TwoAssetParam& p) 
        : grid(g), params(p), E_V_next(g.total_size), E_Vm_next(g.total_size) {
        // Enforce Na=1 for OneAsset mode optimization
        if (grid.N_a != 1) {
            std::cerr << "Warning: OneAssetSolver initialized with Na=" << grid.N_a 
                      << ". Forcing 1-asset logic (ignoring 'a' grid)." << std::endl;
        }
    }

    // --- Core Solver Method ---
    // Returns max diff
    double solve_bellman(const TwoAssetPolicy& guess, TwoAssetPolicy& result, 
                         const IncomeProcess& income) {
        
        // 1. Expectation Step
        compute_expectations(guess, income);

        double max_diff = 0.0;
        
        // 2. Solve Conditional Value Functions (Loop over z)
        // Ignoring 'a' dimension (assuming ia=0 always)
        for (int iz = 0; iz < grid.n_z; ++iz) {
            double z_val = income.z_grid[iz];
            double net_income = params.fiscal.tax_rule.after_tax(z_val);
            solve_slice(iz, net_income, result);
        }

        // 3. Convergence Check
        for(int k=0; k<grid.total_size; ++k) {
            double d = std::abs(result.value[k] - guess.value[k]);
            if(d > max_diff) max_diff = d;
        }

        return max_diff;
    }

private:
    double u(double c) const {
        if (c <= 1e-9) return -1e9;
        if (std::abs(params.sigma - 1.0) < 1e-5) return std::log(c);
        return std::pow(c, 1.0 - params.sigma) / (1.0 - params.sigma);
    }

    double inv_u_prime(double val) const {
        if (val <= 1e-9) return 1e9;
        return std::pow(val, -1.0/params.sigma);
    }

    void compute_expectations(const TwoAssetPolicy& next_pol, const IncomeProcess& income) {
        // Flattened indexing over m, (a=0), z
        for(int iz = 0; iz < grid.n_z; ++iz) {
            for(int im = 0; im < grid.N_m; ++im) {
                int curr_idx = grid.idx(im, 0, iz);
                
                double ev = 0.0;
                double evm = 0.0;
                
                for(int iz_next = 0; iz_next < grid.n_z; ++iz_next) {
                    double prob = income.Pi_flat[iz * grid.n_z + iz_next];
                    if(prob > 1e-10) {
                        int next_idx = grid.idx(im, 0, iz_next);
                        ev += prob * next_pol.value[next_idx];
                        
                        double c_next = next_pol.c_pol[next_idx];
                        double u_prime_next = std::pow(c_next, -params.sigma);
                        evm += prob * u_prime_next * (1.0 + params.r_m);
                    }
                }
                
                E_V_next[curr_idx] = ev;
                E_Vm_next[curr_idx] = evm;
            }
        }
    }

    void solve_slice(int iz, double net_income, TwoAssetPolicy& res) {
        // Standard EGM on m-grid
        int Nm = grid.N_m;
        std::vector<double> c_endo(Nm);
        std::vector<double> m_endo(Nm);
        
        for(int im_next=0; im_next < Nm; ++im_next) {
            double m_prime = grid.m_grid.nodes[im_next];
            
            // Expected Marginal Value at m_prime
            // We need to interp E_Vm_next at m_prime
            double rhs = params.beta * interp_1d(E_Vm_next, iz, m_prime);
            double c = inv_u_prime(rhs);
            
            // Budget: c + m' = (1+r)m + y
            // m = (c + m' - y) / (1+r)
            double m_curr = (c + m_prime - net_income) / (1.0 + params.r_m);
            
            c_endo[im_next] = c;
            m_endo[im_next] = m_curr;
        }
        
        // Re-interpolate to fixed grid
        for(int im=0; im < Nm; ++im) {
            double m_fixed = grid.m_grid.nodes[im];
            double c_val, m_prime_val;
            
            // Constrained region
            if (m_fixed < m_endo[0]) {
                m_prime_val = grid.m_grid.nodes[0]; // Binds at min
                c_val = (1.0 + params.r_m) * m_fixed + net_income - m_prime_val;
            } else {
                c_val = interp_1d_vec(m_endo, c_endo, m_fixed);
                m_prime_val = interp_1d_vec(m_endo, grid.m_grid.nodes, m_fixed);
            }
            
            // Value Update
            // V = u(c) + beta * E V(m')
            double ev = interp_1d(E_V_next, iz, m_prime_val);
            double val = u(c_val) + params.beta * ev;
            
            int idx = grid.idx(im, 0, iz);
            res.value[idx] = val;
            res.c_pol[idx] = c_val;
            res.m_pol[idx] = m_prime_val;
            res.a_pol[idx] = 0.0; // No illiquid
            res.d_pol[idx] = 0.0;
            res.adjust_flag[idx] = 0.0;
        }
    }
    
    // Interpolate E_Vm_next[iz, m]
    double interp_1d(const std::vector<double>& data, int iz, double m) const {
        int offset = iz * grid.N_m;
        const auto& nodes = grid.m_grid.nodes;
        
        if (m <= nodes.front()) {
            // Extrapolation warning could be added here if critical
            return data[offset];
        }
        if (m >= nodes.back()) {
             // std::cerr << "Warning: Extrapolating beyond max 'm' grid." << std::endl;
             return data[offset + grid.N_m - 1];
        }
        
        auto it = std::lower_bound(nodes.begin(), nodes.end(), m);
        int i = std::distance(nodes.begin(), it);
        
        double t = (m - nodes[i-1]) / (nodes[i] - nodes[i-1]);
        return data[offset + i-1] + t * (data[offset + i] - data[offset + i-1]);
    }
    
    double interp_1d_vec(const std::vector<double>& x, const std::vector<double>& y, double xi) const {
        if (xi <= x.front()) return y.front();
        if (xi >= x.back()) return y.back();
        auto it = std::lower_bound(x.begin(), x.end(), xi);
        int i = std::distance(x.begin(), it);
        double t = (xi - x[i-1]) / (x[i] - x[i-1]);
        return y[i-1] + t * (y[i] - y[i-1]);
    }
};

} // namespace Monad
