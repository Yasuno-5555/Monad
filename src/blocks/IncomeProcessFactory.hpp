#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <utility>
#include <Sokudo/Core/ou_process.hpp>

namespace Monad {

class IncomeProcessFactory {
public:
    struct DiscretizedProcess {
        std::vector<double> z_grid;
        std::vector<double> Pi_flat;
        int n_z;
    };

    /**
     * @brief Generates a Rouwenhorst discretization of an AR(1) process.
     * 
     * Uses Sokudo's OU process definition (conceptually) but implements 
     * the specific discrete approximation here.
     * 
     * @param rho Persistence parameter
     * @param sigma_eps Standard deviation of shock
     * @param acc Accuracy related param? No, usually rho and sigma_e or sigma_uncond.
     *            Let's assume inputs are rho and sigma_unconditional for now.
     * @param n_points Number of grid points
     * @return DiscretizedProcess
     */
    static DiscretizedProcess make_rouwenhorst(double rho, double sigma_uncond, int n_points) {
        
        // Rouwenhorst method
        // 1. Construct Grid
        double psi = std::sqrt(n_points - 1.0) * sigma_uncond;
        double min_z = -psi;
        double max_z = psi;
        double step = (max_z - min_z) / (n_points - 1);
        
        std::vector<double> z_grid(n_points);
        for(int i=0; i<n_points; ++i) {
            z_grid[i] = std::exp(min_z + i * step); // Log-normal process usually
        }

        // 2. Construct Transition Matrix (p, q)
        double p = (1.0 + rho) / 2.0;
        double q = p; 

        // Base case N=2
        std::vector<double> Pi_prev = {p, 1-p, 1-q, q};
        
        // Recursive construction
        for(int n=3; n<=n_points; ++n) {
            int dim_prev = n-1;
            int dim_curr = n;
            std::vector<double> Pi_curr(dim_curr * dim_curr, 0.0);
            
            // Fill Pi_curr based on Pi_prev
            for(int i=0; i<dim_prev; ++i) {
                for(int j=0; j<dim_prev; ++j) {
                    double val = Pi_prev[i*dim_prev + j];
                    
                    // Top-Left block * p
                    Pi_curr[i*dim_curr + j] += p * val;
                    
                    // Top-Right block * (1-p)
                    Pi_curr[i*dim_curr + (j+1)] += (1.0-p) * val;
                    
                    // Bottom-Left block * (1-q)
                    Pi_curr[(i+1)*dim_curr + j] += (1.0-q) * val;
                    
                    // Bottom-Right block * q
                    Pi_curr[(i+1)*dim_curr + (j+1)] += q * val;
                }
            }
            
            // Re-normalize rows for safety
             for(int i=0; i<dim_curr; ++i) {
                double sum = 0;
                for(int j=0; j<dim_curr; ++j) sum += Pi_curr[i*dim_curr + j];
                if(sum > 1e-9) {
                    for(int j=0; j<dim_curr; ++j) Pi_curr[i*dim_curr + j] /= sum;
                }
            }
            
            Pi_prev = Pi_curr;
        }
        
        return {z_grid, Pi_prev, n_points};
    }
};

} // namespace Monad
