#pragma once
#include <vector>
#include <Eigen/Sparse>
#include <algorithm>
#include "../UnifiedGrid.hpp"
#include "../Params.hpp"
#include "../Dual.hpp"

namespace Monad {

class JacobianBuilder {
public:
    using SparseMat = Eigen::SparseMatrix<double>;

    struct PolicyPartials {
        // r changes
        Eigen::VectorXd da_dr; // da'(a)/dr
        Eigen::VectorXd dc_dr; // dc(a)/dr
        
        // w changes
        Eigen::VectorXd da_dw; // da'(a)/dw
        Eigen::VectorXd dc_dw; // dc(a)/dw
    };

    struct EgmResultDual { std::vector<Duald> c_pol, a_pol; };

    // Step 1: Dual to get partials
    static PolicyPartials compute_partials(
        const UnifiedGrid& grid,
        const MonadParams& params,
        const std::vector<double>& mu_ss, // Steady State Marginal Utility
        double r_ss, double w_ss
    ) {
        PolicyPartials out;
        int n = grid.size;
        out.da_dr.resize(n); out.dc_dr.resize(n);
        out.da_dw.resize(n); out.dc_dw.resize(n);

        // 1. Sensitivities to Interest Rate r
        // Dual seed: r = r_ss + epsilon, w = w_ss
        Duald r_dual(r_ss, 1.0);
        Duald w_dual(w_ss, 0.0);
        
        auto res_r = run_egm_one_step_dual(grid, params, mu_ss, r_dual, w_dual);
        
        for(int i=0; i<n; ++i) {
            out.da_dr[i] = res_r.a_pol[i].der;
            out.dc_dr[i] = res_r.c_pol[i].der;
        }

        // 2. Sensitivities to Wage w
        // Dual seed: r = r_ss, w = w_ss + epsilon
        // Note: Reset r_dual derivative to 0
        r_dual = Duald(r_ss, 0.0);
        w_dual = Duald(w_ss, 1.0);
        
        auto res_w = run_egm_one_step_dual(grid, params, mu_ss, r_dual, w_dual);
        
        for(int i=0; i<n; ++i) {
            out.da_dw[i] = res_w.a_pol[i].der;
            out.dc_dw[i] = res_w.c_pol[i].der;
        }

        return out;
    }

    // Output: Sparse Transition Matrix (Lambda)
    static SparseMat build_transition_matrix(const std::vector<double>& a_pol, const UnifiedGrid& grid) {
        int n = grid.size;
        SparseMat Lambda(n, n);
        std::vector<Eigen::Triplet<double>> tripletList;
        tripletList.reserve(n * 2); 

        for (int i = 0; i < n; ++i) {
            double a_next = a_pol[i];
            
            auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_next);
            
            int j = 0;
            if (it == grid.nodes.begin()) {
                j = 0;
            } else if (it == grid.nodes.end()) {
                j = n - 2;
            } else {
                j = (int)std::distance(grid.nodes.begin(), it) - 1;
            }
            
            if (j < 0) j = 0;
            if (j >= n - 1) j = n - 2;

            double dx = grid.nodes[j+1] - grid.nodes[j];
            double weight_right = (a_next - grid.nodes[j]) / dx;
            double weight_left = 1.0 - weight_right;

            if (weight_right > 1.0) { weight_right = 1.0; weight_left = 0.0; }
            if (weight_right < 0.0) { weight_right = 0.0; weight_left = 1.0; }

            tripletList.push_back(Eigen::Triplet<double>(i, j, weight_left));
            tripletList.push_back(Eigen::Triplet<double>(i, j+1, weight_right));
        }

        Lambda.setFromTriplets(tripletList.begin(), tripletList.end());
        return Lambda;
    }

private:
    static EgmResultDual run_egm_one_step_dual(
        const UnifiedGrid& grid, const MonadParams& params,
        const std::vector<double>& mu_ss, 
        Duald r, Duald w
    ) {
        int n = grid.size;
        EgmResultDual res;
        res.c_pol.resize(n); res.a_pol.resize(n);
        
        double beta = params.get_required("beta");
        double sigma = params.get("sigma", 2.0);

        std::vector<Duald> c_endo(n);
        std::vector<Duald> a_endo(n);
        
        // 1. Endogenous Grid
        for(int i=0; i<n; ++i) {
             // Expectation is FIXED at steady state mu_ss
             // Euler: u'(c) = beta * (1+r) * E[u'(c')]
             Duald rhs = beta * (1.0 + r) * mu_ss[i]; // mu_ss is double, promoted to Dual
             
             Duald exponent = -1.0 / sigma; 
             Duald c = pow(rhs, exponent);
             c_endo[i] = c;
             
             // Budget: a = (c + a' - w) / (1+r)
             Duald a_prime = grid.nodes[i];
             a_endo[i] = (c + a_prime - w) / (1.0 + r);
        }
        
        // 2. Interpolation (Linear)
        int j = 0; 
        for(int i=0; i<n; ++i) {
            double a_target = grid.nodes[i];
            
            if (a_target <= a_endo[0].val) {
                res.a_pol[i] = grid.nodes[0];
                res.c_pol[i] = (1.0 + r) * a_target + w - res.a_pol[i];
                continue;
            }
            
            if (a_target >= a_endo[n-1].val) {
                 Duald slope_c = (c_endo[n-1] - c_endo[n-2]) / (a_endo[n-1] - a_endo[n-2]);
                 res.c_pol[i] = c_endo[n-1] + slope_c * (a_target - a_endo[n-1]);
                 res.a_pol[i] = (1.0 + r) * a_target + w - res.c_pol[i];
                 continue;
            }
            
            while(j < n - 2 && a_target > a_endo[j+1].val) {
                j++;
            }
            
            Duald denom = a_endo[j+1] - a_endo[j];
            Duald weight = (a_target - a_endo[j]) / denom;
            
            res.c_pol[i] = c_endo[j] * (1.0 - weight) + c_endo[j+1] * weight;
            res.a_pol[i] = (1.0 + r) * a_target + w - res.c_pol[i];
        }
        
        return res;
    }
};

} // namespace Monad
