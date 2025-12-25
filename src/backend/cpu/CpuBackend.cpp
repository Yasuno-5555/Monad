#include "backend/cpu/CpuBackend.hpp"
#include <cmath>
#include <iostream>
#include <algorithm>
#include <iterator>

namespace Monad {

CpuBackend::CpuBackend(const MultiDimGrid& g, const TwoAssetParam& p) 
    : grid(g), params(p), E_V_next(g.total_size), E_Vm_next(g.total_size) {}

double CpuBackend::solve_bellman_iteration(const TwoAssetPolicy& guess, 
                                           TwoAssetPolicy& result, 
                                           const IncomeProcess& income) {
    // 1. Expectation Step
    compute_expectations(guess, income);

    double max_diff = 0.0;

    // 2. Solve Conditional Value Functions
    for (int iz = 0; iz < grid.n_z; ++iz) {
        for (int ia = 0; ia < grid.N_a; ++ia) {
            
            double z_val = income.z_grid[iz];
            double net_income = params.fiscal.tax_rule.after_tax(z_val);
            
            // Problem A: No Adjustment
            solve_no_adjust_slice(iz, ia, net_income, result);

            // Problem B: Adjustment
            solve_adjust_slice(iz, ia, net_income, result);
        }
    }

    // 3. Convergence Check
    finalize_policy(guess, result, max_diff);
    return max_diff;
}

// --- Helper implementations ---

void CpuBackend::compute_expectations(const TwoAssetPolicy& next_pol, const IncomeProcess& income) {
    bool nan_found = false;
    for(int flat_idx = 0; flat_idx < grid.total_size; ++flat_idx) {
        int im, ia, iz;
        grid.get_coords(flat_idx, im, ia, iz);
        
        double ev = 0.0;
        double evm = 0.0;
        
        for(int iz_next = 0; iz_next < grid.n_z; ++iz_next) {
            double prob = income.Pi_flat[iz * grid.n_z + iz_next];
            if(prob > 1e-10) {
                int next_idx = grid.idx(im, ia, iz_next);
                ev += prob * next_pol.value[next_idx];
                
                double c_next = next_pol.c_pol[next_idx];
                if (c_next < 1e-9) c_next = 1e-9;

                // V_m = u'(c) * (1+r_m)
                double u_prime = std::pow(c_next, -params.sigma);
                evm += prob * u_prime * (1.0 + params.r_m);
            }
        }
        E_V_next[flat_idx] = ev;
        E_Vm_next[flat_idx] = evm;
        
        if (!nan_found && (std::isnan(ev) || std::isnan(evm))) {
             // std::cout << "[WARN] NaN detected" << std::endl;
             nan_found = true;
        }
    }
}

double CpuBackend::u(double c) const {
    if (c <= 1e-9) return -1e9;
    if (std::abs(params.sigma - 1.0) < 1e-5) return std::log(c);
    return std::pow(c, 1.0 - params.sigma) / (1.0 - params.sigma);
}

double CpuBackend::inv_u_prime(double val) const {
    if (val <= 1e-9) return 1e9;
    return std::pow(val, -1.0/params.sigma);
}

double CpuBackend::adj_cost(double d, double a_curr) const {
    if (std::abs(d) < 1e-9) return 0.0;
    return params.chi * d * d;
}

double CpuBackend::interp_1d(const std::vector<double>& x, const std::vector<double>& y, double xi) const {
    if (xi <= x.front()) return y.front();
    if (xi >= x.back()) return y.back();
    auto it = std::lower_bound(x.begin(), x.end(), xi);
    int idx = static_cast<int>(std::distance(x.begin(), it));
    double t = (xi - x[idx-1]) / (x[idx] - x[idx-1]);
    return y[idx-1] + t * (y[idx] - y[idx-1]);
}

double CpuBackend::interp_1d_slice_m(const std::vector<double>& data, int iz, int ia, double m) const {
    int offset = grid.idx(0, ia, iz);
    const auto& mg = grid.m_grid.nodes;
    if(m <= mg.front()) return data[offset];
    if(m >= mg.back()) return data[offset + grid.N_m - 1];
    auto it = std::lower_bound(mg.begin(), mg.end(), m);
    int im = static_cast<int>(std::distance(mg.begin(), it));
    double t_m = (m - mg[im-1]) / (mg[im] - mg[im-1]);
    return data[offset + im - 1] + t_m * (data[offset + im] - data[offset + im - 1]);
}

double CpuBackend::interpolate_2d_m_a(const std::vector<double>& data, int iz, double m, double a) const {
    const auto& ag = grid.a_grid.nodes;
    if(a <= ag.front()) return interp_1d_slice_m(data, iz, 0, m);
    if(a >= ag.back()) return interp_1d_slice_m(data, iz, grid.N_a-1, m);
    auto it = std::lower_bound(ag.begin(), ag.end(), a);
    int ia = static_cast<int>(std::distance(ag.begin(), it));
    double t_a = (a - ag[ia-1]) / (ag[ia] - ag[ia-1]);
    double v1 = interp_1d_slice_m(data, iz, ia-1, m);
    double v2 = interp_1d_slice_m(data, iz, ia, m);
    return v1 + t_a * (v2 - v1);
}

void CpuBackend::solve_no_adjust_slice(int iz, int ia, double z_val, TwoAssetPolicy& res) {
    double a_curr = grid.a_grid.nodes[ia];
    double a_next_no_adjust = a_curr * (1.0 + params.r_a);
    int Nm = grid.N_m;
    
    std::vector<double> c_endo(Nm), m_endo(Nm);
    for(int im_next=0; im_next < Nm; ++im_next) {
        double m_next = grid.m_grid.nodes[im_next];
        double emv = interpolate_2d_m_a(E_Vm_next, iz, m_next, a_next_no_adjust);
        double c = inv_u_prime(params.beta * emv);
        m_endo[im_next] = (c + m_next - z_val) / (1.0 + params.r_m);
        c_endo[im_next] = c;
    }
    
    for(int im=0; im < Nm; ++im) {
        double m_fixed = grid.m_grid.nodes[im];
        double c_val, m_prime_val;
        if(m_fixed < m_endo[0]) {
            m_prime_val = grid.m_grid.nodes[0];
            c_val = (1.0 + params.r_m) * m_fixed + z_val - m_prime_val;
        } else {
            c_val = interp_1d(m_endo, c_endo, m_fixed);
            m_prime_val = interp_1d(m_endo, grid.m_grid.nodes, m_fixed);
        }
        if(c_val < 1e-9) c_val = 1e-9;
        
        double ev = interpolate_2d_m_a(E_V_next, iz, m_prime_val, a_next_no_adjust);
        
        int idx = grid.idx(im, ia, iz);
        res.value[idx] = u(c_val) + params.beta * ev;
        res.c_pol[idx] = c_val;
        res.m_pol[idx] = m_prime_val;
        res.a_pol[idx] = a_next_no_adjust;
        res.d_pol[idx] = 0.0;
        res.adjust_flag[idx] = 0.0;
    }
}

void CpuBackend::solve_adjust_slice(int iz, int ia, double z_val, TwoAssetPolicy& res) {
     double a_curr = grid.a_grid.nodes[ia];
     int Nm = grid.N_m;
     
     for (int ia_next = 0; ia_next < grid.N_a; ++ia_next) {
         double a_next = grid.a_grid.nodes[ia_next];
         double d = a_next - a_curr * (1.0 + params.r_a);
         // double cost = adj_cost(d, a_curr);
         double total_outflow = d + adj_cost(d, a_curr);
         
         std::vector<double> c_endo(Nm), m_endo(Nm);
         for(int im_next=0; im_next < Nm; ++im_next) {
              double m_next = grid.m_grid.nodes[im_next];
              double emv = interp_1d_slice_m(E_Vm_next, iz, ia_next, m_next);
              double c = inv_u_prime(params.beta * emv);
              m_endo[im_next] = (c + m_next - z_val + total_outflow) / (1.0 + params.r_m);
              c_endo[im_next] = c;
         }
         
         for(int im=0; im < Nm; ++im) {
             double m_fixed = grid.m_grid.nodes[im];
             double c_adj, m_prime_adj;
             
             if(m_fixed < m_endo[0]) {
                 m_prime_adj = grid.m_grid.nodes[0];
                 c_adj = (1.0 + params.r_m) * m_fixed + z_val - total_outflow - m_prime_adj;
             } else {
                 c_adj = interp_1d(m_endo, c_endo, m_fixed);
                 m_prime_adj = interp_1d(m_endo, grid.m_grid.nodes, m_fixed);
             }
             
             if(c_adj > 1e-9) {
                 double ev = interp_1d_slice_m(E_V_next, iz, ia_next, m_prime_adj);
                 double val_adj = u(c_adj) + params.beta * ev;
                 
                 int idx = grid.idx(im, ia, iz);
                 if (val_adj > res.value[idx]) {
                     res.value[idx] = val_adj;
                     res.c_pol[idx] = c_adj;
                     res.m_pol[idx] = m_prime_adj;
                     res.a_pol[idx] = a_next;
                     res.d_pol[idx] = d;
                     res.adjust_flag[idx] = 1.0;
                 }
             }
         }
     }
}

void CpuBackend::finalize_policy(const TwoAssetPolicy& old, TwoAssetPolicy& res, double& diff) {
    for(int k=0; k<grid.total_size; ++k) {
        double d = std::abs(res.value[k] - old.value[k]);
        if(d > diff) diff = d;
    }
}

} // namespace Monad
