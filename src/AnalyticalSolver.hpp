#pragma once
#define NOMINMAX

#include "Dual.hpp"
#include "UnifiedGrid.hpp"
#include "DistributionAggregator.hpp"
#include "Params.hpp"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

class AnalyticalSolver {
public:
    static void solve_steady_state(const UnifiedGrid& grid, double& r_guess, const MonadParams& m_params) {
        
        std::cout << "--- Analytical Newton Solver (v1.2 HANK) ---" << std::endl;
        
        double r = r_guess;
        const int max_iter = 20;
        const double tol = 1e-6;
        int na = grid.size;
        int nz = m_params.income.n_z;
        int size = na * nz; // Flattened size
        
        double beta = m_params.get_required("beta");
        double sigma = m_params.get("sigma", 2.0);
        double alpha = m_params.get("alpha", 0.33);
        double A = m_params.get("A", 1.0);
        
        for(int iter=0; iter<max_iter; ++iter) {
            Duald r_dual(r, 1.0); 
            
            Duald term = r_dual / (alpha * A);
            Duald K_dem = pow(term, 1.0/(alpha - 1.0)); // Capital Demand
            Duald w_dual = (1.0 - alpha) * A * pow(K_dem, alpha);
            
            // 1. Initialize Expected Marginal Utility E[u'(c')]
            // Guess c = r*a + w*z
            std::vector<Duald> expected_mu(size);
            for(int j=0; j<nz; ++j) {
                double z = m_params.income.z_grid[j];
                for(int i=0; i<na; ++i) {
                    Duald c = r_dual * grid.nodes[i] + w_dual * z;
                    if (c.val < 1e-4) c = 1e-4; // Safe
                    expected_mu[j*na + i] = pow(c, -sigma);
                }
            }
            
            std::vector<Duald> c_pol(size), a_pol(size);
            
            // 2. EGM Fixed Point Loop
            for(int k=0; k<1000; ++k) {
                std::vector<Duald> c_endo(size);
                std::vector<Duald> a_endo(size); // Endogenous asset grid
                bool egm_success = true;
                
                // A. Compute Expected MU for current states using Matrix Product
                // EU[j, i] = Sum_next ( Pi[j, next] * expected_mu[next, i] )
                // Here 'i' represents a' choice (which corresponds to grid nodes today)
                std::vector<Duald> EU(size);
                for(int j=0; j<nz; ++j) {
                    for(int i=0; i<na; ++i) {
                        Duald sum_mu = 0.0;
                        for(int next=0; next<nz; ++next) {
                             sum_mu = sum_mu + m_params.income.prob(j, next) * expected_mu[next*na + i];
                        }
                        EU[j*na + i] = sum_mu;
                    }
                }

                // B. Invert Euler & Budget
                for(int j=0; j<nz; ++j) {
                    double z = m_params.income.z_grid[j];
                    for(int i=0; i<na; ++i) {
                        int idx = j*na + i;
                        Duald emu = EU[idx]; 
                        Duald rhs = beta * (1.0 + r_dual) * emu;
                        Duald c = pow(rhs, -1.0/sigma);
                        
                        c_endo[idx] = c;
                        // Budget: c + a' = (1+r)a + wz
                        // a = (c + a' - wz) / (1+r)
                        Duald a_prime = grid.nodes[i];
                        a_endo[idx] = (c + a_prime - w_dual * z) / (1.0 + r_dual);
                    }
                }

                // C. Interpolation (for each z layer)
                for(int j=0; j<nz; ++j) {
                    double z = m_params.income.z_grid[j];
                    int offset = j * na;
                    
                    // Slice for this z
                    // Check monotonicity? (omitted for speed, critical)
                    
                    int p = 0; // Pointer for interpolation
                    for(int i=0; i<na; ++i) {
                        int idx = offset + i;
                        double a_target = grid.nodes[i];
                        Duald a_min = a_endo[offset];
                        Duald a_max = a_endo[offset + na - 1];
                        
                        // Borrowing Constraint / Boundary
                        if (a_target <= a_min.val) {
                             a_pol[idx] = grid.nodes[0]; // Binding constraint a' = a_min (0)
                             c_pol[idx] = (1.0 + r_dual)*a_target + w_dual*z - a_pol[idx];
                             continue;
                        }
                        if (a_target >= a_max.val) {
                             // Extrapolate slope from last segment
                             Duald slope_c = (c_endo[offset+na-1] - c_endo[offset+na-2]) / (a_endo[offset+na-1] - a_endo[offset+na-2]);
                             c_pol[idx] = c_endo[offset+na-1] + slope_c * (a_target - a_endo[offset+na-1]);
                             a_pol[idx] = (1.0 + r_dual)*a_target + w_dual*z - c_pol[idx];
                             continue;
                        }

                        // Find bracket
                        while(p < na - 2 && a_target > a_endo[offset + p + 1].val) {
                            p++;
                        }
                        
                        Duald denom = a_endo[offset + p + 1] - a_endo[offset + p];
                        Duald weight = (a_target - a_endo[offset + p]) / denom;
                        
                        c_pol[idx] = c_endo[offset + p] * (1.0 - weight) + c_endo[offset + p + 1] * weight;
                        a_pol[idx] = (1.0 + r_dual)*a_target + w_dual*z - c_pol[idx];
                    }
                }
                
                // D. Update mu_next
                std::vector<Duald> expected_mu_next(size);
                for(int idx=0; idx<size; ++idx) {
                    expected_mu_next[idx] = pow(c_pol[idx], -sigma);
                }

                double mu_diff = 0.0;
                for(int idx=0; idx<size; ++idx) {
                    mu_diff += std::abs(expected_mu_next[idx].val - expected_mu[idx].val);
                }
                expected_mu = expected_mu_next;
                if (mu_diff < 1e-9) break;
            } // End EGM Loop
            
            // 3. Aggregation (2D)
            std::vector<Duald> D(size);
            for(int idx=0; idx<size; ++idx) D[idx] = 1.0 / size; // Uniform init
            
            for(int t=0; t<5000; ++t) {
                // Use 2D Iterator
                std::vector<Duald> D_next = Monad::DistributionAggregator::forward_iterate_2d(D, a_pol, grid, m_params.income);
                
                double dist_diff = 0.0;
                for(int idx=0; idx<size; ++idx) dist_diff += std::abs(D_next[idx].val - D[idx].val);
                D = D_next;
                if(dist_diff < 1e-10) break;
            }
            
            // 4. Market Clearing
            Duald K_sup = 0.0;
            for(int idx=0; idx<size; ++idx) {
                // Need a' or a? Steady state K = Sum D(a,z) * a
                // But D is distribution over grid points (a, z).
                int a_idx = idx % na;
                K_sup = K_sup + D[idx] * grid.nodes[a_idx];
            }
            
            Duald resid = K_dem - K_sup;
            std::cout << "Iter " << iter << ": r=" << r << ", Resid=" << resid.val << ", J=" << resid.der << std::endl;
            
            if(std::abs(resid.val) < tol) {
                std::cout << "Converged!" << std::endl;
                r_guess = r;
                return;
            }
            
            double step = resid.val / resid.der;
            if (std::abs(step) > 0.005) step = (step > 0 ? 0.005 : -0.005); // Smaller step for stability
            r = r - step;
            
            // Bounds check
            if (r < 0.0001) r = 0.0001;
            if (r > 0.15) r = 0.15;
        }
        std::cout << "Max iter reached." << std::endl;
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
        std::vector<double> expected_mu(size);
        for(int j=0; j<nz; ++j) {
            double z = m_params.income.z_grid[j];
            for(int i=0; i<na; ++i) {
                double c = r * grid.nodes[i] + w * z;
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
                     a_endo[idx] = (c + grid.nodes[i] - w * z) / (1.0 + r);
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
                         c_pol[idx] = (1.0 + r)*a_target + w*z - a_pol[idx];
                         continue;
                     }
                     if (a_target >= a_endo[offset + na -1]) {
                         double slope = (c_endo[offset+na-1] - c_endo[offset+na-2])/(a_endo[offset+na-1] - a_endo[offset+na-2]);
                         c_pol[idx] = c_endo[offset+na-1] + slope*(a_target - a_endo[offset+na-1]);
                         a_pol[idx] = (1.0 + r)*a_target + w*z - c_pol[idx];
                         continue;
                     }
                     while(p < na-2 && a_target > a_endo[offset + p + 1]) p++;
                     double wgt = (a_target - a_endo[offset+p])/(a_endo[offset+p+1] - a_endo[offset+p]);
                     c_pol[idx] = c_endo[offset+p]*(1.0-wgt) + c_endo[offset+p+1]*wgt;
                     a_pol[idx] = (1.0 + r)*a_target + w*z - c_pol[idx];
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
