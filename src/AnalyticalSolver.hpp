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
        
        std::cout << "--- Analytical Newton Solver (Phase 3) ---" << std::endl;
        
        double r = r_guess;
        const int max_iter = 20;
        const double tol = 1e-6;
        int n = grid.size;
        
        double beta = m_params.get_required("beta");
        double sigma = m_params.get("sigma", 2.0);
        double alpha = m_params.get("alpha", 0.33);
        double A = m_params.get("A", 1.0);
        
        for(int iter=0; iter<max_iter; ++iter) {
            Duald r_dual(r, 1.0); 
            
            Duald term = r_dual / (alpha * A);
            Duald K_dem = pow(term, 1.0/(alpha - 1.0));
            Duald w_dual = (1.0 - alpha) * A * pow(K_dem, alpha);
            
            // EGM Initialization
            std::vector<Duald> expected_mu(n);
             
            // Initial guess
            for(int i=0; i<n; ++i) {
                Duald c = r_dual * grid.nodes[i] + w_dual;
                expected_mu[i] = pow(c, -sigma);
            }
            
            std::vector<Duald> c_pol(n), a_pol(n);
            
            // EGM Fixed Point Loop
            for(int k=0; k<500; ++k) {
                std::vector<Duald> c_endo(n);
                std::vector<Duald> a_endo(n);
                bool egm_success = true;
                
                for(int i=0; i<n; ++i) {
                     Duald emu = expected_mu[i];
                     Duald rhs = beta * (1.0 + r_dual) * emu;
                     Duald exponent = -1.0 / sigma; 
                     Duald c = pow(rhs, exponent);
                     c_endo[i] = c;
                     Duald a_prime = grid.nodes[i];
                     a_endo[i] = (c + a_prime - w_dual) / (1.0 + r_dual);
                }
                
                // Monotonicity Check
                for (int i = 1; i < n; ++i) {
                    if (a_endo[i] < a_endo[i-1]) {
                        egm_success = false;
                        break;
                    }
                }
                
                if (!egm_success) {
                    std::cerr << "Non-monotonicity in EGM. Aborting." << std::endl;
                    break;
                }
                
                // Interpolation
                int j = 0; 
                for(int i=0; i<n; ++i) {
                    double a_target = grid.nodes[i];
                    
                    if (a_target <= a_endo[0].val) {
                        a_pol[i] = grid.nodes[0];
                        c_pol[i] = (1.0 + r_dual) * a_target + w_dual - a_pol[i];
                        continue;
                    }
                    
                    if (a_target >= a_endo[n-1].val) {
                         Duald slope_c = (c_endo[n-1] - c_endo[n-2]) / (a_endo[n-1] - a_endo[n-2]);
                         c_pol[i] = c_endo[n-1] + slope_c * (a_target - a_endo[n-1]);
                         a_pol[i] = (1.0 + r_dual) * a_target + w_dual - c_pol[i];
                         continue;
                    }
                    
                    while(j < n - 2 && a_target > a_endo[j+1].val) {
                        j++;
                    }
                    
                    Duald denom = a_endo[j+1] - a_endo[j];
                    Duald weight = (a_target - a_endo[j]) / denom;
                    
                    c_pol[i] = c_endo[j] * (1.0 - weight) + c_endo[j+1] * weight;
                    a_pol[i] = (1.0 + r_dual) * a_target + w_dual - c_pol[i];
                }
                
                // Update expected_mu
                std::vector<Duald> expected_mu_next(n);
                for(int i=0; i<n; ++i) {
                    expected_mu_next[i] = pow(c_pol[i], -sigma);
                }
                
                double mu_diff = 0.0;
                for(int i=0; i<n; ++i) {
                    mu_diff += std::abs(expected_mu_next[i].val - expected_mu[i].val);
                }
                expected_mu = expected_mu_next;
                
                if (mu_diff < 1e-10) break;
            }
            
            // Aggregation
            std::vector<Duald> D(n);
            for(int i=0; i<n; ++i) D[i] = 1.0 / n;
            
            for(int t=0; t<2000; ++t) {
                std::vector<Duald> D_next = Monad::DistributionAggregator::forward_iterate(D, a_pol, grid);
                double diff = 0.0;
                for(int i=0; i<n; ++i) diff += std::abs(D_next[i].val - D[i].val);
                D = D_next;
                if(diff < 1e-10) break;
            }
            
            Duald K_sup = 0.0;
            for(int i=0; i<n; ++i) K_sup = K_sup + D[i] * grid.nodes[i];
            
            Duald resid = K_dem - K_sup;
            std::cout << "Iter " << iter << ": r=" << r << ", Resid=" << resid.val << ", J=" << resid.der << std::endl;
            
            if(std::abs(resid.val) < tol) {
                std::cout << "Converged!" << std::endl;
                r_guess = r;
                return;
            }
            
            double step = resid.val / resid.der;
            if (std::abs(step) > 0.01) step = (step > 0 ? 0.01 : -0.01);
            r = r - step;
            
            if (r < 0.001) r = 0.001;
            if (r > 0.2) r = 0.2;
        }
        std::cout << "Max iter reached." << std::endl;
    }

    // Helper to retrieve steady state objects for SSJ
    static void get_steady_state_policy(
        const UnifiedGrid& grid, double r, const MonadParams& m_params,
        std::vector<double>& c_out, std::vector<double>& mu_out, std::vector<double>& a_out
    ) {
        // Re-run one EGM step at fixed r
        int n = grid.size;
        double beta = m_params.get_required("beta");
        double sigma = m_params.get("sigma", 2.0);
        double alpha = m_params.get("alpha", 0.33);
        double A = m_params.get("A", 1.0);
        
        // Prices
        double K_dem = std::pow(r / (alpha * A), 1.0/(alpha-1.0));
        double w = (1.0 - alpha) * A * std::pow(K_dem, alpha);
        
        // 1. Initial Guess (same as solver)
        std::vector<double> expected_mu(n);
        for(int i=0; i<n; ++i) {
            double c = r * grid.nodes[i] + w;
            expected_mu[i] = std::pow(c, -sigma);
        }
        
        std::vector<double> c_pol(n), a_pol(n);
        
        // Converge Policy
        for(int k=0; k<200; ++k) {
             std::vector<double> c_endo(n), a_endo(n);
             
             for(int i=0; i<n; ++i) {
                 double rhs = beta * (1.0 + r) * expected_mu[i];
                 double c = std::pow(rhs, -1.0/sigma);
                 c_endo[i] = c;
                 a_endo[i] = (c + grid.nodes[i] - w) / (1.0 + r);
             }
             
             // Interp
             int j=0;
             for(int i=0; i<n; ++i) {
                 double a_target = grid.nodes[i];
                 if (a_target <= a_endo[0]) {
                     a_pol[i] = grid.nodes[0];
                     c_pol[i] = (1.0 + r)*a_target + w - a_pol[i];
                     continue;
                 }
                 if (a_target >= a_endo[n-1]) {
                     double slope = (c_endo[n-1] - c_endo[n-2])/(a_endo[n-1] - a_endo[n-2]);
                     c_pol[i] = c_endo[n-1] + slope*(a_target - a_endo[n-1]);
                     a_pol[i] = (1.0 + r)*a_target + w - c_pol[i];
                     continue;
                 }
                 while(j < n-2 && a_target > a_endo[j+1]) j++;
                 double wgt = (a_target - a_endo[j])/(a_endo[j+1] - a_endo[j]);
                 c_pol[i] = c_endo[j]*(1.0-wgt) + c_endo[j+1]*wgt;
                 a_pol[i] = (1.0 + r)*a_target + w - c_pol[i];
             }
             
             std::vector<double> next_mu(n);
             double mu_diff=0;
             for(int i=0; i<n; ++i) {
                 next_mu[i] = std::pow(c_pol[i], -sigma);
                 mu_diff += std::abs(next_mu[i] - expected_mu[i]);
             }
             expected_mu = next_mu;
             if(mu_diff < 1e-10) break;
        }
        
        c_out = c_pol;
        a_out = a_pol;
        mu_out = expected_mu;
    }
};
