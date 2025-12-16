#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <vector>
#include <cmath>

// Core Components
#include "Params.hpp"
#include "io/json_loader.hpp"
#include "UnifiedGrid.hpp"
#include "AnalyticalSolver.hpp"
#include "ssj/jacobian_builder.hpp"
#include "ssj/aggregator.hpp"
#include "ssj/ssj_solver.hpp"

// Simple CSV Writer Helpers
void write_csv_ss(const std::string& filename, const UnifiedGrid& grid, 
                  const std::vector<double>& c, const std::vector<double>& a_pol, 
                  const Eigen::VectorXd& D) {
    std::ofstream f(filename);
    f << "asset,consumption,next_asset,distribution\n";
    for(int i=0; i<grid.size; ++i) {
        f << grid.nodes[i] << "," << c[i] << "," << a_pol[i] << "," << D[i] << "\n";
    }
    f.close();
    std::cout << "[IO] Wrote " << filename << std::endl;
}

void write_csv_trans(const std::string& filename, const Eigen::VectorXd& dr, 
                     const Eigen::VectorXd& dZ, int T) {
    std::ofstream f(filename);
    f << "period,dZ,dr\n";
    for(int t=0; t<T; ++t) {
        double dz_val = (t < dZ.size()) ? dZ[t] : 0.0;
        double dr_val = (t < dr.size()) ? dr[t] : 0.0;
        f << t << "," << dz_val << "," << dr_val << "\n";
    }
    f.close();
    std::cout << "[IO] Wrote " << filename << std::endl;
}

int main(int argc, char* argv[]) {
    try {
        // 1. Setup & Load Config
        std::string config_path = (argc > 1) ? argv[1] : "model_ir.json";
        std::cout << "=== Monad Engine v1.1 ===" << std::endl;
        std::cout << "Config: " << config_path << std::endl;

        if (!std::filesystem::exists(config_path)) {
            std::cerr << "Error: Config file not found: " << config_path << std::endl;
            return 1;
        }

        UnifiedGrid grid;
        MonadParams params;
        
        // Load Grid & Params
        JsonLoader::load_model(config_path, grid, params);

        // 2. Solve Steady State
        std::cout << "\n--- Step 1: Solving Steady State ---" << std::endl;
        
        double r_guess = params.get("r_guess", 0.02);
        
        // Run Analytical Solver
        AnalyticalSolver::solve_steady_state(grid, r_guess, params);
        
        double r_ss = r_guess;
        double beta = params.get_required("beta");
        double sigma = params.get("sigma", 2.0);
        double alpha = params.get("alpha", 0.33); 
        double A = params.get("A", 1.0);
        
        double K_dem = std::pow(r_ss / (alpha * A), 1.0 / (alpha - 1.0));
        double w_ss = (1.0 - alpha) * A * std::pow(K_dem, alpha);
        
        std::cout << "  -> Equilibrium r = " << r_ss << " (" << r_ss * 100.0 << "%)" << std::endl;

        // Retrieve Full Steady State Policy & Distribution
        // NOTE: Implemented helper in AnalyticalSolver to retrieve these after convergence
        std::vector<double> c_ss, mu_ss, a_pol_ss;
        AnalyticalSolver::get_steady_state_policy(grid, r_ss, params, c_ss, mu_ss, a_pol_ss);

        // Build Steady State Transition Matrix and Distribution properly
        auto Lambda_ss = Monad::JacobianBuilder::build_transition_matrix(a_pol_ss, grid);
        
        // Solve for invariant distribution D_ss
        Eigen::VectorXd D_ss(grid.size);
        std::vector<double> D_std(grid.size, 1.0/grid.size);
        for(int t=0; t<5000; ++t) {
            Eigen::VectorXd D_curr = Eigen::Map<Eigen::VectorXd>(D_std.data(), grid.size);
            Eigen::VectorXd D_next = Lambda_ss.transpose() * D_curr;
            double diff = (D_next - D_curr).cwiseAbs().sum();
            std::copy(D_next.data(), D_next.data() + grid.size, D_std.begin());
            if(diff < 1e-11) break;
        }
        D_ss = Eigen::Map<Eigen::VectorXd>(D_std.data(), grid.size);

        // Output Steady State
        write_csv_ss("steady_state.csv", grid, c_ss, a_pol_ss, D_ss);

        
        // 3. Prepare for SSJ (Partials)
        std::cout << "\n--- Step 2: Building Jacobian Partials ---" << std::endl;
        
        auto partials = Monad::JacobianBuilder::compute_partials(
            grid, params, mu_ss, r_ss, w_ss
        );
        
        // 4. Check for Shocks & Solve Transition
        bool run_shock = true; 
        
        if (run_shock) {
            std::cout << "\n--- Step 3: Solving Transition Path (SSJ) ---" << std::endl;
            int T = 200;
            
            // Define Shock dZ (e.g., 1% TFP shock AR(1))
            // We interpret dZ as the exogenous shift in Capital Demand
            Eigen::VectorXd dZ = Eigen::VectorXd::Zero(T);
            double rho = 0.9;
            double shock_size = 0.01; 
            
            for(int t=0; t<T; ++t) {
                 dZ[t] = shock_size * std::pow(rho, t); 
            }

            // Solve Linear System
            Eigen::VectorXd dr_path = Monad::SsjSolver::solve_linear_transition(
                T, grid, D_ss, Lambda_ss, a_pol_ss, partials, dZ
            );
            
            // Output Results
            write_csv_trans("transition.csv", dr_path, dZ, T);
        }

        std::cout << "\n=== Finished Successfully ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
