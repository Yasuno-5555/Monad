#pragma once
#include <Eigen/Dense>
#include <map>
#include <vector>
#include <string>
#include "SsjSolver3D.hpp"
#include "../blocks/SimpleBlocks.hpp"


namespace Monad {

class GeneralEquilibrium {
    SsjSolver3D& solver;
    int T; // Horizon

public:
    GeneralEquilibrium(SsjSolver3D& s, int horizon) : solver(s), T(horizon) {}

    // Solve for General Equilibrium Impulse Response
    // Input: Shock path to r_m (monetary policy)
    // Output: Paths for Y, C, etc.
    std::map<std::string, Eigen::VectorXd> solve_monetary_shock(const Eigen::VectorXd& dr_m) {
        
        // 1. Get Partial Jacobians from Household Block
        // J_C_rm: Direct effect of r_m on C
        // J_C_Y:  Effect of Income (Y) on C (via w*N)
        auto J = solver.compute_block_jacobians(T);
        
        // Ensure keys exist
        if(J.find("C") == J.end()) throw std::runtime_error("Jacobian for C missing");
        if(J["C"].find("rm") == J["C"].end()) throw std::runtime_error("Partials for rm missing");
        if(J["C"].find("w") == J["C"].end()) throw std::runtime_error("Partials for w missing"); // mapped to Y
        
        Eigen::MatrixXd J_C_rm = J["C"]["rm"];
        Eigen::MatrixXd J_C_Y  = J["C"]["w"]; // Assumption: wage rate w moves 1-to-1 with Output Y (linear production)

        // 2. Define GE System
        // Goods Market Clearing: dY = dC
        // dC = J_C_rm * dr_m + J_C_Y * dY
        // => dY - J_C_Y * dY = J_C_rm * dr_m
        // => (I - J_C_Y) * dY = J_C_rm * dr_m
        // => dY = (I - J_C_Y)^-1 * (J_C_rm * dr_m)

        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(T, T);
        Eigen::MatrixXd A = I - J_C_Y;
        
        Eigen::VectorXd dC_partial = J_C_rm * dr_m;
        Eigen::VectorXd dY = A.colPivHouseholderQr().solve(dC_partial); // Linear Solve

        // 3. Recover other variables
        // dC = dY (Market Clearing)
        Eigen::VectorXd dC = dY;
        
        // decomposition: dC = Direct + Indirect(GE)
        // Direct = J_C_rm * dr_m
        // Indirect = J_C_Y * dY
        
        // Pack results
        std::map<std::string, Eigen::VectorXd> results;
        results["dr_m"] = dr_m;
        results["dY"] = dY;
        results["dC"] = dC;
        results["dC_direct"] = dC_partial;
        results["dC_indirect"] = J_C_Y * dY;
        
        return results;
    }

    // v2.2/2.3: ZLB Solver with Forward Guidance Support
    // forced_binding: set of periods where i=0 is enforced regardless of Taylor Rule
    std::map<std::string, Eigen::VectorXd> solve_with_zlb(const Eigen::VectorXd& dr_star, 
                                                          const std::vector<int>& forced_binding = {}) {
        
        // 1. Get Canonical Jacobians (Unconstrained)
        auto J = solver.compute_block_jacobians(T);
        
        // Household Partials
        Eigen::MatrixXd J_C_r = J["C"]["rm"]; // Consumption wrt real rate (rm)
        Eigen::MatrixXd J_C_Y = J["C"]["w"];  // Consumption wrt Income (Y)
        
        // NKPC & Taylor Rule Blocks (SimpleBlocks)
        double beta = 0.97;
        double kappa = 0.1; 
        double phi_pi = 1.5;
        
        // Build NKPC Matrices
        Eigen::MatrixXd M_pi_Y = SimpleBlocks::build_nkpc_jacobian(T, beta, kappa);
        
        // Build Fisher & Taylor
        Eigen::MatrixXd L_inv = Eigen::MatrixXd::Zero(T, T);
        for(int t=0; t<T-1; ++t) L_inv(t, t+1) = 1.0;
        
        Eigen::MatrixXd Term_r_Y = (phi_pi * Eigen::MatrixXd::Identity(T,T) - L_inv) * M_pi_Y;
        
        // Substitute into IS Curve: Y = J_C_r * r + J_C_Y * Y
        Eigen::MatrixXd A_GE = Eigen::MatrixXd::Identity(T, T) - J_C_Y - J_C_r * Term_r_Y;
        Eigen::MatrixXd J_Y_eps = A_GE.colPivHouseholderQr().solve(J_C_r); // Maps eps -> Y
        Eigen::MatrixXd J_Y_rstar = J_Y_eps; // Symmetric map
        
        // Map epsilon -> i_star (Shadow Rate)
        Eigen::MatrixXd J_istar_eps = phi_pi * M_pi_Y * J_Y_eps;
        Eigen::VectorXd istar_base = phi_pi * M_pi_Y * (J_Y_rstar * dr_star) + dr_star;
        
        // 2. Regime Iteration Loop
        Eigen::VectorXd eps = Eigen::VectorXd::Zero(T);
        std::vector<int> binding_idx = forced_binding; // Initialize with forced periods
        
        for(int iter=0; iter<20; ++iter) {
            Eigen::VectorXd istar = istar_base + J_istar_eps * eps;
            std::vector<int> next_binding;
            
            // Add forced periods first
            for(int t : forced_binding) next_binding.push_back(t);
            
            for(int t=0; t<T; ++t) {
                bool is_forced = false;
                for(int fb : forced_binding) if(fb == t) is_forced = true;
                if(is_forced) continue; // Already added

                double i_val = istar[t] + eps[t]; 
                
                // Binding Condition: i < 0
                if (i_val < -1e-6) {
                    next_binding.push_back(t);
                } 
                // Unbinding Check: eps > 0 (Bound) but i_star > 0 (Release)
                // We keep it in binding set if we still need eps > 0 to maintain i=0
                // Wait, if we keep checks simple:
                // Check if index was in previous binding_idx. 
                // If so, we check sign of eps.
                // But simplified heuristic:
                else if (eps[t] > 1e-6) {
                     next_binding.push_back(t); 
                }
            }
            
            // Deduplicate
            std::sort(next_binding.begin(), next_binding.end());
            next_binding.erase(std::unique(next_binding.begin(), next_binding.end()), next_binding.end());
            
            if (next_binding == binding_idx && iter > 0) break; // Converged
            
            binding_idx = next_binding;
            
            // Solve Sub-system
            eps = Eigen::VectorXd::Zero(T); 
            int n_bind = binding_idx.size();
            
            if (n_bind > 0) {
                Eigen::MatrixXd H_sub(n_bind, n_bind);
                Eigen::VectorXd rhs_sub(n_bind);
                Eigen::MatrixXd H = J_istar_eps + Eigen::MatrixXd::Identity(T, T);
                
                for(int r=0; r<n_bind; ++r) {
                    int tr = binding_idx[r];
                    rhs_sub(r) = -istar_base(tr);
                    for(int c=0; c<n_bind; ++c) {
                        int tc = binding_idx[c];
                        H_sub(r, c) = H(tr, tc);
                    }
                }
                
                Eigen::VectorXd eps_sub = H_sub.colPivHouseholderQr().solve(rhs_sub);
                
                // Enforce eps >= 0 (Release if negative), EXCEPT for forced periods
                std::vector<int> final_binding;
                for(int r=0; r<n_bind; ++r) {
                    int t_idx = binding_idx[r];
                    bool is_forced = false;
                    for(int fb : forced_binding) if(fb == t_idx) is_forced = true;

                    if (eps_sub(r) > 0.0 || is_forced) { // Keep if eps>0 OR Forced
                         eps(t_idx) = eps_sub(r); // Note: eps might be negative if forced!
                         final_binding.push_back(t_idx);
                    }
                }
                binding_idx = final_binding; 
            }
        }
        
        // Final Computation
        Eigen::VectorXd Y = J_Y_rstar * dr_star + J_Y_eps * eps;
        Eigen::VectorXd istar = istar_base + J_istar_eps * eps;
        Eigen::VectorXd i_nom = istar + eps; 
        
        std::map<std::string, Eigen::VectorXd> final_res;
        final_res["dY"] = Y;
        final_res["i"] = i_nom;
        final_res["i_star"] = istar;
        final_res["eps"] = eps;
        final_res["r_star"] = dr_star; // Add r_star for check
        
        return final_res;
    }
};

} // namespace Monad
