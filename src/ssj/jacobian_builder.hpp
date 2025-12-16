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

    // 2D Transition Matrix Construction (Sparse)
    // Indexes: Row (Today) -> Col (Tomorrow)
    // Row k = (iz * Na + ia)
    // Col k' = (iz_next * Na + ia_next)
    static Eigen::SparseMatrix<double> build_transition_matrix_2d(
        const std::vector<double>& a_pol, // Flattened 2D Policy
        const UnifiedGrid& grid,
        const IncomeProcess& income
    ) {
        int na = grid.size;
        int nz = income.n_z;
        int size = na * nz;
        
        Eigen::SparseMatrix<double> Lambda(size, size);
        std::vector<Eigen::Triplet<double>> triplets;
        triplets.reserve(size * nz * 2); // Roughly 2 neighbors per asset * Nz income shocks

        for(int iz = 0; iz < nz; ++iz) {
            for(int ia = 0; ia < na; ++ia) {
                int row_idx = iz * na + ia;
                
                // 1. Asset Transition (Lorentz/Linear weight)
                double a_val = a_pol[row_idx];
                
                // Find bracket [nodes[p], nodes[p+1]]
                auto it = std::lower_bound(grid.nodes.begin(), grid.nodes.end(), a_val);
                int p = 0;
                if (it == grid.nodes.begin()) p = 0;
                else if (it == grid.nodes.end()) p = na - 2;
                else p = (int)std::distance(grid.nodes.begin(), it) - 1;
                
                if(p < 0) p = 0;
                if(p >= na - 1) p = na - 2;

                double dx = grid.nodes[p+1] - grid.nodes[p];
                double w_right = (a_val - grid.nodes[p]) / dx;
                double w_left = 1.0 - w_right;
                
                // Clamp
                if (w_right > 1.0) { w_right = 1.0; w_left = 0.0; }
                if (w_right < 0.0) { w_right = 0.0; w_left = 1.0; }

                // 2. Income Transition
                for(int iz_next = 0; iz_next < nz; ++iz_next) {
                    double prob = income.prob(iz, iz_next);
                    if (prob < 1e-12) continue;

                    // Combined transition weights
                    int col_left = iz_next * na + p;
                    int col_right = iz_next * na + (p+1);
                    
                    triplets.push_back(Eigen::Triplet<double>(row_idx, col_left, prob * w_left));
                    triplets.push_back(Eigen::Triplet<double>(row_idx, col_right, prob * w_right));
                }
            }
        }
        
        Lambda.setFromTriplets(triplets.begin(), triplets.end());
        return Lambda;
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

    // Step 1: Dual to get partials (2D Version)
    static PolicyPartials compute_partials_2d(
        const UnifiedGrid& grid,
        const MonadParams& params,
        const std::vector<double>& mu_ss, // Steady State Marginal Utility (Flattened)
        double r_ss, double w_ss
    ) {
        PolicyPartials out;
        int size = grid.size * params.income.n_z;
        
        out.da_dr.resize(size); out.dc_dr.resize(size);
        out.da_dw.resize(size); out.dc_dw.resize(size);

        // 1. Sensitivities to Interest Rate r
        Duald r_dual(r_ss, 1.0);
        Duald w_dual(w_ss, 0.0);
        
        auto res_r = run_egm_one_step_2d_dual(grid, params, mu_ss, r_dual, w_dual);
        
        for(int i=0; i<size; ++i) {
            out.da_dr[i] = res_r.a_pol[i].der;
            out.dc_dr[i] = res_r.c_pol[i].der;
        }

        // 2. Sensitivities to Wage w
        r_dual = Duald(r_ss, 0.0);
        w_dual = Duald(w_ss, 1.0);
        
        auto res_w = run_egm_one_step_2d_dual(grid, params, mu_ss, r_dual, w_dual);
        
        for(int i=0; i<size; ++i) {
            out.da_dw[i] = res_w.a_pol[i].der;
            out.dc_dw[i] = res_w.c_pol[i].der;
        }

        return out;
    }

private:
    static EgmResultDual run_egm_one_step_2d_dual(
        const UnifiedGrid& grid, const MonadParams& params,
        const std::vector<double>& mu_ss, 
        Duald r, Duald w
    ) {
        int na = grid.size;
        int nz = params.income.n_z;
        int size = na * nz;

        EgmResultDual res;
        res.c_pol.resize(size); res.a_pol.resize(size);
        
        double beta = params.get_required("beta");
        double sigma = params.get("sigma", 2.0);

        std::vector<Duald> c_endo(size);
        std::vector<Duald> a_endo(size);
        
        // 1. Endogenous Grid
        // Requires Expectation E[u']
        std::vector<double> expected_mu_ss(size);
        for(int j=0; j<nz; ++j) {
            for(int i=0; i<na; ++i) {
                double sum_mu = 0.0;
                for(int next=0; next<nz; ++next) {
                    sum_mu += params.income.prob(j, next) * mu_ss[next*na + i];
                }
                expected_mu_ss[j*na + i] = sum_mu;
            }
        }

        for(int j=0; j<nz; ++j) {
            double z = params.income.z_grid[j];
            for(int i=0; i<na; ++i) {
                int idx = j*na + i;
                
                // Euler: u'(c) = beta * (1+r) * E[u'(c')]
                Duald rhs = beta * (1.0 + r) * expected_mu_ss[idx]; 
                
                Duald exponent = -1.0 / sigma; 
                Duald c = pow(rhs, exponent);
                if (c.val < 1e-4) c = 1e-4; // Safety
                c_endo[idx] = c;
                
                // Budget: c + a' = (1+r)a + wz
                // a = (c + a' - wz) / (1+r)
                Duald a_prime = grid.nodes[i];
                a_endo[idx] = (c + a_prime - w * z) / (1.0 + r);
            }
        }
        
        // 2. Interpolation (Linear) for each z
        for(int j=0; j<nz; ++j) {
            double z = params.income.z_grid[j];
            int offset = j * na;
            int p = 0;
            
            for(int i=0; i<na; ++i) {
                int idx = offset + i;
                double a_target = grid.nodes[i];
                
                // Binding constraint or Extrapolation
                if (a_target <= a_endo[offset].val) {
                    res.a_pol[idx] = grid.nodes[0];
                    res.c_pol[idx] = (1.0 + r) * a_target + w * z - res.a_pol[idx];
                    continue;
                }
                
                if (a_target >= a_endo[offset + na -1].val) {
                     Duald slope_c = (c_endo[offset+na-1] - c_endo[offset+na-2]) / (a_endo[offset+na-1] - a_endo[offset+na-2]);
                     res.c_pol[idx] = c_endo[offset+na-1] + slope_c * (a_target - a_endo[offset+na-1]);
                     res.a_pol[idx] = (1.0 + r) * a_target + w * z - res.c_pol[idx];
                     continue;
                }
                
                while(p < na - 2 && a_target > a_endo[offset + p + 1].val) {
                    p++;
                }
                
                Duald denom = a_endo[offset + p + 1] - a_endo[offset + p];
                Duald weight = (a_target - a_endo[offset + p]) / denom;
                
                res.c_pol[idx] = c_endo[offset + p] * (1.0 - weight) + c_endo[offset + p + 1] * weight;
                res.a_pol[idx] = (1.0 + r) * a_target + w * z - res.c_pol[idx];
            }
        }
        
        return res;
    }
};

} // namespace Monad
