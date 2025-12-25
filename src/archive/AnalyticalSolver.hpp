#include "ZigenAdapter.hpp"
#include "solver/ZigenNewtonSolver.hpp"
#include "UnifiedGrid.hpp"
#include "DistributionAggregator.hpp"
#include "Params.hpp"
#include "kernel/TaxSystem.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

class AnalyticalSolver {
public:
    /**
     * @brief High-precision Steady State Solver using Zigen Newton Method.
     */
    static void solve_steady_state_zigen(const UnifiedGrid& grid, double& r_guess, const MonadParams& m_params) {
        std::cout << "\n--- Zigen-Accelerated Newton Solver (HANK SS) ---" << std::endl;
        
        using ZigenDualV = Monad::ZigenBridge::ZigenDualV;
        
        // Define Objective Function for Newton Solver
        auto objective = [&](const std::vector<ZigenDualV>& prices) -> std::vector<ZigenDualV> {
            // prices[0] is interest rate r
            ZigenDualV r_dual = prices[0];
            
            // 1. Production side (Firm Block)
            double beta = m_params.get_required("beta");
            double sigma = m_params.get("sigma", 2.0);
            double alpha = m_params.get("alpha", 0.33);
            double A = m_params.get("A", 1.0);
            
            // K_demand = (r / alpha*A)^(1/(alpha-1))
            ZigenDualV term = r_dual / (alpha * A);
            ZigenDualV K_dem = pow(term, 1.0 / (alpha - 1.0));
            ZigenDualV w_dual = (1.0 - alpha) * A * pow(K_dem, alpha);

            // Tax System
            Monad::TaxSystem tax_sys;
            tax_sys.lambda = m_params.get("tax_lambda", 1.0);
            tax_sys.tau = m_params.get("tax_tau", 0.0);
            tax_sys.transfer = m_params.get("tax_transfer", 0.0);

            // 2. Household Block (EGM)
            int na = (int)grid.size;
            int nz = m_params.income.n_z;
            int size = na * nz;

            // Initialize expected marginal utility
            std::vector<ZigenDualV> expected_mu(size);
            for(int j=0; j<nz; ++j) {
                double z = m_params.income.z_grid[j];
                ZigenDualV gross_lab = w_dual * z;
                for(int i=0; i<na; ++i) {
                    ZigenDualV gross = r_dual * grid.nodes[i] + gross_lab;
                    ZigenDualV c = tax_sys.get_net_income(gross);
                    if (c.val < 1e-5) c = 1e-5;
                    expected_mu[j*na + i] = pow(c, -sigma);
                }
            }

            std::vector<ZigenDualV> c_pol(size), a_pol(size);
            
            // EGM Inner Loop (Converge Policy)
            for(int k=0; k<200; ++k) {
                std::vector<ZigenDualV> c_endo(size), a_endo(size);
                std::vector<ZigenDualV> EU(size);
                for(int j=0; j<nz; ++j) {
                    for(int i=0; i<na; ++i) {
                        ZigenDualV sum_mu = 0.0;
                        for(int next=0; next<nz; ++next) {
                             sum_mu = sum_mu + m_params.income.prob(j, next) * expected_mu[next*na + i];
                        }
                        EU[j*na + i] = sum_mu;
                    }
                }

                for(int j=0; j<nz; ++j) {
                    double z = m_params.income.z_grid[j];
                    for(int i=0; i<na; ++i) {
                        int idx = j*na + i;
                        ZigenDualV rhs = beta * (1.0 + r_dual) * EU[idx];
                        ZigenDualV c = pow(rhs, -1.0/sigma);
                        c_endo[idx] = c;
                        double a_p_val = grid.nodes[i];
                        ZigenDualV res = c + a_p_val;
                        a_endo[idx] = tax_sys.solve_asset_from_budget(res, r_dual, w_dual, z);
                    }
                }

                // Interpolation
                for(int j=0; j<nz; ++j) {
                    double z = m_params.income.z_grid[j];
                    int offset = j * na;
                    int p = 0;
                    for(int i=0; i<na; ++i) {
                        int idx = offset + i;
                        double a0 = grid.nodes[i];
                        if (a0 <= a_endo[offset].val) {
                            a_pol[idx] = ZigenDualV(grid.nodes[0]);
                        } else if (a0 >= a_endo[offset + na - 1].val) {
                            ZigenDualV slope = (c_endo[offset+na-1] - c_endo[offset+na-2]) / (a_endo[offset+na-1] - a_endo[offset+na-2]);
                            ZigenDualV c = c_endo[offset+na-1] + slope * (a0 - a_endo[offset+na-1]);
                            a_pol[idx] = ZigenDualV(a0) + tax_sys.get_net_income(r_dual * a0 + w_dual * z) - c;
                        } else {
                            while(p < na - 2 && a0 > a_endo[offset + p + 1].val) p++;
                            ZigenDualV weight = (ZigenDualV(a0) - a_endo[offset + p]) / (a_endo[offset + p + 1] - a_endo[offset + p]);
                            ZigenDualV c = c_endo[offset + p] * (ZigenDualV(1.0) - weight) + c_endo[offset + p + 1] * weight;
                            a_pol[idx] = ZigenDualV(a0) + tax_sys.get_net_income(r_dual * a0 + w_dual * z) - c;
                        }
                    }
                }

                // Update MU and Check Convergence
                double diff = 0;
                for(int i=0; i<size; ++i) {
                    ZigenDualV c = ZigenDualV(grid.nodes[i % na]) + tax_sys.get_net_income(r_dual * grid.nodes[i % na] + w_dual * m_params.income.z_grid[i / na]) - a_pol[i];
                    ZigenDualV mu_new = pow(c, -sigma);
                    diff += std::abs(mu_new.val - expected_mu[i].val);
                    expected_mu[i] = mu_new;
                }
                if (diff < 1e-10) break;
            } // End EGM Loop

            // 3. Distribution Aggregation
            std::vector<ZigenDualV> D(size, ZigenDualV(1.0 / size));
            for(int t=0; t<2000; ++t) {
                std::vector<ZigenDualV> D_next = Monad::DistributionAggregator::forward_iterate_2d(D, a_pol, grid, m_params.income);
                double d_diff = 0;
                for(int i=0; i<size; ++i) d_diff += std::abs(D_next[i].val - D[i].val);
                D = D_next;
                if(d_diff < 1e-10) break;
            }

            // 4. Market Clearing Condition
            ZigenDualV K_sup = 0.0;
            for(int i=0; i<size; ++i) K_sup = K_sup + D[i] * grid.nodes[i % na];
            
            return { K_dem - K_sup };
        };

        // Run Newton Solver
        std::vector<double> initial_guess = { r_guess };
        Monad::ZigenNewtonSolver::SolverParams params;
        params.verbose = true;
        params.tol = 1e-7;

        auto solution = Monad::ZigenNewtonSolver::solve(objective, initial_guess, params);
        r_guess = solution[0];
    }

    static void solve_steady_state(const UnifiedGrid& grid, double& r_guess, const MonadParams& m_params) {
        // Redundant or legacy, can call zigen-version
        solve_steady_state_zigen(grid, r_guess, m_params);
    }

    // Helper to retrieve steady state objects for SSJ (Updated for 2D)
    // NOTE: For now, we return flattened vectors.
    static void get_steady_state_policy(
        const UnifiedGrid& grid, double r, const MonadParams& m_params,
        std::vector<double>& c_out, std::vector<double>& mu_out, std::vector<double>& a_out, std::vector<double>& D_out
    ) {
        // Re-run minimal EGM step at fixed r
        int na = grid.size;
        int nz = m_params.income.n_z;
        int size = na * nz;

        double beta = m_params.get_required("beta");
        double sigma = m_params.get("sigma", 2.0);
        double alpha = m_params.get("alpha", 0.33);
        double A = m_params.get("A", 1.0);
        
        double K_dem = std::pow(r / (alpha * A), 1.0/(alpha-1.0));
        double w = (1.0 - alpha) * A * std::pow(K_dem, alpha);
        
        // Initial Guess
        // Tax System
        Monad::TaxSystem tax_sys;
        tax_sys.lambda = m_params.get("tax_lambda", 1.0);
        tax_sys.tau = m_params.get("tax_tau", 0.0); 
        tax_sys.transfer = m_params.get("tax_transfer", 0.0);

        std::vector<double> expected_mu(size);
        for(int j=0; j<nz; ++j) {
            double z = m_params.income.z_grid[j];
            for(int i=0; i<na; ++i) {
                double gross = r * grid.nodes[i] + w * z;
                double c = tax_sys.get_net_income(gross); // Initial guess: consumes all income
                if(c < 1e-5) c = 1e-5;
                expected_mu[j*na+i] = std::pow(c, -sigma);
            }
        }
        
        std::vector<double> c_pol(size), a_pol(size);
        
        // Converge Policy
        for(int k=0; k<200; ++k) {
             std::vector<double> c_endo(size), a_endo(size);
             std::vector<double> EU(size);
             
             // Expectation Step
             for(int j=0; j<nz; ++j) {
                 for(int i=0; i<na; ++i) {
                     double sum_mu = 0.0;
                     for(int next=0; next<nz; ++next) {
                          sum_mu += m_params.income.prob(j, next) * expected_mu[next*na+i];
                     }
                     EU[j*na+i] = sum_mu;
                 }
             }

             // Endogenous Grid
             for(int j=0; j<nz; ++j) {
                 double z = m_params.income.z_grid[j];

                 
                 for(int i=0; i<na; ++i) {
                     int idx = j*na+i;
                     double rhs = beta * (1.0 + r) * EU[idx];
                     double c = std::pow(rhs, -1.0/sigma);
                     c_endo[idx] = c;
                     // Budget Inversion
                     double a_prime = grid.nodes[i];
                     double resources = c + a_prime;
                     a_endo[idx] = tax_sys.solve_asset_from_budget(resources, r, w, z);
                 }
             }
             
             // Interpolation
             for(int j=0; j<nz; ++j) {
                 double z = m_params.income.z_grid[j];
                 
                 int offset = j*na;
                 int p=0;
                 for(int i=0; i<na; ++i) {
                     int idx = offset + i;
                     double a_target = grid.nodes[i];
                     
                     if (a_target <= a_endo[offset]) {
                         a_pol[idx] = grid.nodes[0];
                         c_pol[idx] = a_target + tax_sys.get_net_income(r*a_target + w*z) - a_pol[idx];
                         continue;
                     }
                     if (a_target >= a_endo[offset + na -1]) {
                         double slope = (c_endo[offset+na-1] - c_endo[offset+na-2])/(a_endo[offset+na-1] - a_endo[offset+na-2]);
                         c_pol[idx] = c_endo[offset+na-1] + slope*(a_target - a_endo[offset+na-1]);
                         a_pol[idx] = a_target + tax_sys.get_net_income(r*a_target + w*z) - c_pol[idx];
                         continue;
                     }
                     while(p < na-2 && a_target > a_endo[offset + p + 1]) p++;
                     double wgt = (a_target - a_endo[offset+p])/(a_endo[offset+p+1] - a_endo[offset+p]);

                     c_pol[idx] = c_endo[offset+p]*(1.0-wgt) + c_endo[offset+p+1]*wgt;
                     a_pol[idx] = a_target + tax_sys.get_net_income(r*a_target + w*z) - c_pol[idx];
                 }
             }
             
             // Update Mu
             std::vector<double> next_mu(size);
             double mad = 0;
             for(int i=0; i<size; ++i) {
                 next_mu[i] = std::pow(c_pol[i], -sigma);
                 mad += std::abs(next_mu[i] - expected_mu[i]);
             }
             expected_mu = next_mu;
             if(mad < 1e-10) break;
        }
        

        
        // Compute Invariant Distribution (2D)
        std::vector<double> D(size);
        for(int idx=0; idx<size; ++idx) D[idx] = 1.0 / size; 
        
        for(int t=0; t<5000; ++t) {
             std::vector<double> D_next = Monad::DistributionAggregator::forward_iterate_2d(D, a_pol, grid, m_params.income);
             double dist_diff = 0.0;
             for(int idx=0; idx<size; ++idx) dist_diff += std::abs(D_next[idx] - D[idx]);
             D = D_next;
             if(dist_diff < 1e-10) break;
        }

        c_out = c_pol;
        a_out = a_pol;
        mu_out = expected_mu;
        D_out = D;
    }
};
